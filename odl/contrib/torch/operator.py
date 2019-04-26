# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for converting ODL operators to pytorch layers.

This requires the ``torch`` module from the ``pytorch`` package,
see `the pytorch installation guide
<https://github.com/pytorch/pytorch#installation>`_ for instructions.
"""

from __future__ import division
from packaging.version import parse as parse_version
import warnings

import numpy as np
import torch

if parse_version(torch.__version__) < parse_version('0.4'):
    warnings.warn("This interface is designed to work with Pytorch >= 0.4",
                  RuntimeWarning)

__all__ = ('OperatorAsAutogradFunction', 'OperatorAsModule')

# TODO: ProductSpaceOperator as implementation of channels_in and channels_out?


class OperatorAsAutogradFunction(torch.autograd.Function):

    """Wrapper of an ODL operator as a ``torch.autograd.Function``.

    This wrapper exposes an ``odl.Operator`` object to ``pytorch``'s
    autograd machinery by implementing custom ``forward()`` and
    ``backward()`` methods.

    Do not use this directly in a pytorch ``Module`` since backpropagation
    with batches and channels will not work as expected. For that
    purpose, use `OperatorAsModule` instead.
    """

    def __init__(self, operator):
        """Initialize a new instance.

        Parameters
        ----------
        operator : `Operator`
            The ODL operator to be wrapped. For gradient computations to
            work, ``operator.derivative(x).adjoint`` must be implemented.

        Examples
        --------
        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype='float32')
        >>> odl_op = odl.MatrixOperator(matrix)
        >>> torch_op = OperatorAsAutogradFunction(odl_op)
        >>> torch_op.operator is odl_op
        True
        """
        super(OperatorAsAutogradFunction, self).__init__()
        self.operator = operator

    def forward(self, input):
        """Evaluate forward pass on the input.

        Parameters
        ----------
        input : `torch.tensor._TensorBase`
            Point at which to evaluate the operator.

        Returns
        -------
        result : `torch.autograd.variable.Variable`
            Variable holding the result of the evaluation.

        Examples
        --------
        Perform a matrix multiplication:

        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype='float32')
        >>> odl_op = odl.MatrixOperator(matrix)
        >>> torch_op = OperatorAsAutogradFunction(odl_op)
        >>> x = torch.Tensor([1, 2, 3])
        >>> x_var = torch.autograd.Variable(x)
        >>> torch_op(x_var)
        Variable containing:
         4
         5
        [torch.FloatTensor of size 2]

        Evaluate a functional, i.e., an operator with scalar output:

        >>> odl_func = odl.solvers.L2NormSquared(odl.rn(3, dtype='float32'))
        >>> torch_func = OperatorAsAutogradFunction(odl_func)
        >>> x = torch.Tensor([1, 2, 3])
        >>> x_var = torch.autograd.Variable(x)
        >>> torch_func(x_var)
        Variable containing:
         14
        [torch.FloatTensor of size 1]
        """
        # TODO: batched evaluation
        if not self.operator.is_linear:
            # Only needed for nonlinear operators
            self.save_for_backward(input)

        # TODO: use GPU memory directly if possible
        input_arr = input.cpu().detach().numpy()
        if any(s == 0 for s in input_arr.strides):
            # TODO: remove when Numpy issue #9165 is fixed
            # https://github.com/numpy/numpy/pull/9177
            input_arr = input_arr.copy()

        op_result = self.operator(input_arr)
        if np.isscalar(op_result):
            # For functionals, the result is funnelled through `float`,
            # so we wrap it into a Numpy array with the same dtype as
            # `operator.domain`
            op_result = np.array(op_result, ndmin=1,
                                 dtype=self.operator.domain.dtype)
        tensor = torch.from_numpy(np.array(op_result, copy=False, ndmin=1))
        tensor = tensor.to(input.device)
        return tensor

    def backward(self, grad_output):
        r"""Apply the adjoint of the derivative at ``grad_output``.

        This method is usually not called explicitly but as a part of the
        ``cost.backward()`` pass of a backpropagation step.

        Parameters
        ----------
        grad_output : `torch.tensor._TensorBase`
            Tensor to which the Jacobian should be applied. See Notes
            for details.

        Returns
        -------
        result : `torch.autograd.variable.Variable`
            Variable holding the result of applying the Jacobian to
            ``grad_output``. See Notes for details.

        Examples
        --------
        Compute the Jacobian adjoint of the matrix operator, which is the
        operator of the transposed matrix. We compose with the ``sum``
        functional to be able to evaluate ``grad``:

        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype='float32')
        >>> odl_op = odl.MatrixOperator(matrix)
        >>> torch_op = OperatorAsAutogradFunction(odl_op)
        >>> x = torch.Tensor([1, 2, 3])
        >>> x_var = torch.autograd.Variable(x, requires_grad=True)
        >>> op_x_var = torch_op(x_var)
        >>> cost = op_x_var.sum()
        >>> cost.backward()
        >>> x_var.grad  # should be matrix.T.dot([1, 1])
        Variable containing:
         1
         1
         2
        [torch.FloatTensor of size 3]

        Compute the gradient of a custom functional:

        >>> odl_func = odl.solvers.L2NormSquared(odl.rn(3, dtype='float32'))
        >>> torch_func = OperatorAsAutogradFunction(odl_func)
        >>> x = torch.Tensor([1, 2, 3])
        >>> x_var = torch.autograd.Variable(x, requires_grad=True)
        >>> func_x_var = torch_func(x_var)
        >>> func_x_var
        Variable containing:
         14
        [torch.FloatTensor of size 1]

        >>> func_x_var.backward()
        >>> x_var.grad  # Should be 2 * x
        Variable containing:
         2
         4
         6
        [torch.FloatTensor of size 3]

        Notes
        -----
        This method applies the contribution of this node, i.e., the
        transpose of the Jacobian of its outputs with respect to its inputs,
        to the gradients of some cost function with respect to the outputs
        of this node.

        Example: Assume that this node computes :math:`x \mapsto C(f(x))`,
        where :math:`x` is a tensor variable and :math:`C` is a scalar-valued
        function. In ODL language, what ``backward`` should compute is

        .. math::
            \nabla(C \circ f)(x) = f'(x)^*\big(\nabla C (f(x))\big)

        according to the chain rule. In ODL code, this corresponds to ::

            f.derivative(x).adjoint(C.gradient(f(x))).

        Hence, the parameter ``grad_output`` is a tensor variable containing
        :math:`y = \nabla C(f(x))`. Then, ``backward`` boils down to
        computing ``[f'(x)^*(y)]`` using the input ``x`` stored during
        the previous `forward` pass.
        """
        # TODO: implement directly for GPU data
        if not self.operator.is_linear:
            input_arr = self.saved_variables[0].data.cpu().numpy()
            if any(s == 0 for s in input_arr.strides):
                # TODO: remove when Numpy issue #9165 is fixed
                # https://github.com/numpy/numpy/pull/9177
                input_arr = input_arr.copy()

        grad = None

        # ODL weights spaces, pytorch doesn't, so we need to handle this
        try:
            dom_weight = self.operator.domain.weighting.const
        except AttributeError:
            dom_weight = 1.0

        try:
            ran_weight = self.operator.range.weighting.const
        except AttributeError:
            ran_weight = 1.0

        scaling = dom_weight / ran_weight

        if self.needs_input_grad[0]:
            grad_output_arr = grad_output.cpu().numpy()
            if any(s == 0 for s in grad_output_arr.strides):
                # TODO: remove when Numpy issue #9165 is fixed
                # https://github.com/numpy/numpy/pull/9177
                grad_output_arr = grad_output_arr.copy()

            if self.operator.is_linear:
                adjoint = self.operator.adjoint
            else:
                adjoint = self.operator.derivative(input_arr).adjoint

            grad_odl = adjoint(grad_output_arr)

            if scaling != 1.0:
                grad_odl *= scaling

            grad = torch.from_numpy(np.array(grad_odl, copy=False, ndmin=1))
            grad = grad.to(grad_output.device)

        return grad

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}(\n    {!r}    \n)'.format(self.__class__.__name__,
                                             self.operator)


class OperatorAsModule(torch.nn.Module):

    """Wrapper of an ODL operator as a ``torch.nn.Module``.

    This wrapper can be used as a layer in ``pytorch`` Neural Networks.
    It works with arbitrary batches and channels and supports
    backpropagation.

    .. note::
        Currently, batches and channels are supported by simply looping
        over them and stacking the results.
    """

    def __init__(self, operator):
        """Initialize a new instance.

        Parameters
        ----------
        operator : `Operator`
            The ODL operator to be wrapped. For gradient computations to
            work, ``operator.derivative(x).adjoint`` must be implemented.

        Examples
        --------
        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype='float32')
        >>> odl_op = odl.MatrixOperator(matrix)
        >>> torch_op = OperatorAsModule(odl_op)
        >>> torch_op.op_func.operator is odl_op
        True
        """
        super(OperatorAsModule, self).__init__()
        self.op_func = OperatorAsAutogradFunction(operator)

    @property
    def operator(self):
        """The wrapped ODL operator."""
        return self.op_func.operator

    def forward(self, x):
        """Compute forward-pass of this module on ``x``.

        Parameters
        ----------
        x : `torch.autograd.variable.Variable`
            Input of this layer. The contained tensor must have shape
            ``extra_shape + operator.domain.shape``, and
            ``len(extra_shape)`` must be at least 1 (batch axis).

        Returns
        -------
        out : `torch.autograd.variable.Variable`
            The computed output. Its tensor will have shape
            ``extra_shape + operator.range.shape``, where ``extra_shape``
            are the extra axes of ``x``.

        Examples
        --------
        Evaluating on a 2D tensor, where the operator expects a 1D input,
        i.e., with extra batch axis only:

        >>> matrix = np.array([[1, 0, 0],
        ...                    [0, 1, 1]], dtype='float32')
        >>> odl_op = odl.MatrixOperator(matrix)
        >>> odl_op.domain.shape
        (3,)
        >>> odl_op.range.shape
        (2,)
        >>> op_mod = OperatorAsModule(odl_op)
        >>> t = torch.ones(3)
        >>> x = autograd.Variable(t[None, :]) # "fake" batch axis
        >>> op_mod(x)
        Variable containing:
         1  2
        [torch.FloatTensor of size 1x2]
        >>> t = torch.ones(3)
        >>> x_tensor = torch.stack([0 * t, 1 * t])
        >>> x = autograd.Variable(x_tensor)  # batch of 2 inputs
        >>> op_mod(x)
        Variable containing:
         0  0
         1  2
        [torch.FloatTensor of size 2x2]

        An arbitrary number of axes is supported:

        >>> x = autograd.Variable(t[None, None, :])  # "fake" batch and channel
        >>> op_mod(x)
        Variable containing:
        (0 ,.,.) =
          1  2
        [torch.FloatTensor of size 1x1x2]
        >>> x_tensor = torch.stack([torch.stack([0 * t, 1 * t]),
        ...                         torch.stack([2 * t, 3 * t]),
        ...                         torch.stack([4 * t, 5 * t])])
        >>> x = autograd.Variable(x_tensor)  # batch of 3x2 inputs
        >>> op_mod(x)
        Variable containing:
        (0 ,.,.) =
           0   0
           1   2
        <BLANKLINE>
        (1 ,.,.) =
           2   4
           3   6
        <BLANKLINE>
        (2 ,.,.) =
           4   8
           5  10
        [torch.FloatTensor of size 3x2x2]
        """
        in_shape = x.data.shape
        op_in_shape = self.op_func.operator.domain.shape
        op_out_shape = self.op_func.operator.range.shape

        extra_shape = in_shape[:-len(op_in_shape)]

        if in_shape[-len(op_in_shape):] != op_in_shape or not extra_shape:
            shp_str = str(op_in_shape).strip('()')
            raise ValueError('expected input of shape (N, *, {}), got input '
                             'with shape {}'.format(shp_str, in_shape))

        # Flatten extra axes, then do one entry at a time
        newshape = (int(np.prod(extra_shape)),) + op_in_shape
        x_flat_xtra = x.reshape(*newshape)
        results = []
        for i in range(x_flat_xtra.data.shape[0]):
            results.append(self.op_func(x_flat_xtra[i]))

        # Reshape the resulting stack to the expected output shape
        stack_flat_xtra = torch.stack(results)
        return stack_flat_xtra.view(extra_shape + op_out_shape)

    def __repr__(self):
        """Return ``repr(self)``."""
        op_name = self.op_func.operator.__class__.__name__
        op_dom_shape = self.op_func.operator.domain.shape
        if len(op_dom_shape) == 1:
            op_dom_shape = op_dom_shape[0]
        op_ran_shape = self.op_func.operator.range.shape
        if len(op_ran_shape) == 1:
            op_ran_shape = op_ran_shape[0]

        return '{}({}) ({} -> {})'.format(self.__class__.__name__,
                                          op_name, op_dom_shape, op_ran_shape)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    import odl
    from torch import autograd, nn
    run_doctests(extraglobs={'np': np, 'odl': odl, 'torch': torch,
                             'nn': nn, 'autograd': autograd})
