# Copyright 2014-2017 The ODL contributors
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

import numpy as np
import torch

__all__ = ('TorchOperator',)

# TODO: ProductSpaceOperator as multi-input, multi-output construction?


class TorchOperator(torch.autograd.Function):

    """Wrap an ODL operator as a Torch autograd Function."""

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
        ...                    [0, 1, 1]], dtype=float)
        >>> odl_op = odl.MatrixOperator(matrix)
        >>> torch_op = TorchOperator(odl_op)
        >>> torch_op.operator is odl_op
        True
        """
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
        ...                    [0, 1, 1]], dtype=float)
        >>> odl_op = odl.MatrixOperator(matrix)
        >>> torch_op = TorchOperator(odl_op)
        >>> x = torch.DoubleTensor([1, 2, 3])
        >>> x_var = torch.autograd.Variable(x)
        >>> torch_op(x_var)
        Variable containing:
         4
         5
        [torch.DoubleTensor of size 2]

        Evaluate a functional, i.e., an operator with scalar output:

        >>> odl_func = odl.solvers.L2NormSquared(odl.rn(3))
        >>> torch_func = TorchOperator(odl_func)
        >>> x = torch.DoubleTensor([1, 2, 3])
        >>> x_var = torch.autograd.Variable(x)
        >>> torch_func(x_var)
        Variable containing:
         14
        [torch.DoubleTensor of size 1]
        """
        if not self.operator.is_linear:
            # Only needed for nonlinear operators
            self.save_for_backward(input)

        input_arr = input.numpy()
        if any(s == 0 for s in input_arr.strides):
            # TODO: remove when Numpy issue #9165 is fixed
            # https://github.com/numpy/numpy/pull/9177
            input_arr = input_arr.copy()

        op_result = self.operator(input_arr)
        return torch.from_numpy(np.array(op_result, copy=False, ndmin=1))

    def backward(self, grad_output):
        """Apply the adjoint of the derivative at ``grad_output``.

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
        ...                    [0, 1, 1]], dtype=float)
        >>> odl_op = odl.MatrixOperator(matrix)
        >>> torch_op = TorchOperator(odl_op)
        >>> x = torch.DoubleTensor([1, 2, 3])
        >>> x_var = torch.autograd.Variable(x, requires_grad=True)
        >>> op_x_var = torch_op(x_var)
        >>> cost = op_x_var.sum()
        >>> cost.backward()
        >>> x_var.grad  # should be matrix.T.dot([1, 1])
        Variable containing:
         1
         1
         2
        [torch.DoubleTensor of size 3]

        Compute the gradient of a custom functional:

        >>> odl_func = odl.solvers.L2NormSquared(odl.rn(3))
        >>> torch_func = TorchOperator(odl_func)
        >>> x = torch.DoubleTensor([1, 2, 3])
        >>> x_var = torch.autograd.Variable(x, requires_grad=True)
        >>> func_x_var = torch_func(x_var)
        >>> func_x_var
        Variable containing:
         14
        [torch.DoubleTensor of size 1]

        >>> func_x_var.backward()
        >>> x_var.grad  # Should be 2 * x
        Variable containing:
         2
         4
         6
        [torch.DoubleTensor of size 3]

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
            \\nabla(C \circ f)(x) = f'(x)^*\\big(\\nabla C (f(x))\\big)

        according to the chain rule. In ODL code, this corresponds to ::

            f.derivative(x).adjoint(C.gradient(f(x))).

        Hence, the parameter ``grad_output`` is a tensor variable containing
        :math:`y = \\nabla C(f(x))`. Then, ``backward`` boils down to
        computing ``[f'(x)^*(y)]`` using the input ``x`` stored during
        the previous `forward` pass.
        """
        if not self.operator.is_linear:
            input_arr = self.saved_variables[0].data.numpy()
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
            grad_output_arr = grad_output.numpy()
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

        return grad


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
