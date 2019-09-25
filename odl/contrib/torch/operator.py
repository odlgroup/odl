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

import warnings

import numpy as np
import torch
from packaging.version import parse as parse_version

from odl import Operator

if parse_version(torch.__version__) < parse_version('0.4'):
    warnings.warn("This interface is designed to work with Pytorch >= 0.4",
                  RuntimeWarning, stacklevel=2)

__all__ = ('OperatorFunction', 'OperatorModule')


class OperatorFunction(torch.autograd.Function):

    """Wrapper of an ODL operator as a ``torch.autograd.Function``.

    This wrapper exposes an `Operator` object to the PyTorch autograd
    machinery by implementing custom ``forward()`` and ``backward()``
    methods.

    These methods should not be used directly. Instead, in a ``Module``,
    the call ``OperatorFunction.apply(operator, input_tensor)`` will
    apply the ``forward()`` method correctly and register gradients
    for the ``backward()`` step during backpropagation.

    The application of ``op`` to multiple inputs is done automatically
    in the background. The only requirement is that the shape of an
    input *ends with* the input shape that ``op`` expects, see below.

    Examples
    --------
    Simple example with of evaluating the ODL ``MatrixOperator`` on an
    input tensor of matching shape:

    >>> matrix = np.array([[1, 0, 1],
    ...                    [0, 1, 1]], dtype='float32')
    >>> odl_op = odl.MatrixOperator(matrix)
    >>> odl_op.domain.shape
    (3,)
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> OperatorFunction.apply(odl_op, x)
    tensor([4., 5.])

    It is possible to pass tensors with extra axes "left" of the ones
    corresponding to the input shape expected by the operator:

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> xs = x[None, None, :]  # shape (1, 1, 3)
    >>> OperatorFunction.apply(odl_op, xs)  # result shape (1, 1, 2)
    tensor([[[4., 5.]]])
    >>> xs = torch.stack([x, 2 * x], dim=0)  # shape (2, 3)
    >>> OperatorFunction.apply(odl_op, xs)  # result shape (2, 2)
    tensor([[ 4.,  5.],
            [ 8., 10.]])

    Functionals, i.e., operators with scalar output, are also supported:

    >>> odl_func = odl.solvers.L2NormSquared(odl.rn(3, dtype='float32'))
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> OperatorFunction.apply(odl_func, x)
    tensor(14.)

    With multiple inputs:

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> xs = torch.stack([x, 2 * x], dim=0)
    >>> OperatorFunction.apply(odl_func, xs)
    tensor([14., 56.])

    Backpropagation makes use of the Jacobian adjoint of the given matrix
    operator, which is transposed matrix operator. We mark the input
    tensor as requiring gradient, and compose the operator with the
    ``sum`` function to be able to backpropagate and get access to
    gradients:

    >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> loss = OperatorFunction.apply(odl_op, x).sum()
    >>> loss
    tensor(9., grad_fn=<SumBackward0>)
    >>> loss.backward()
    >>> x.grad  # should be matrix.T.dot([1, 1])
    tensor([1., 1., 2.])

    With multiple inputs:

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> xs = torch.stack([x, 2 * x], dim=0).requires_grad_(True)
    >>> loss = OperatorFunction.apply(odl_op, xs).sum()
    >>> loss
    tensor(27., grad_fn=<SumBackward0>)
    >>> loss.backward()
    >>> xs.grad
    tensor([[1., 1., 2.],
            [1., 1., 2.]])

    We can again use a custom functional, with single or multiple
    inputs:

    >>> odl_func = odl.solvers.L2NormSquared(odl.rn(3, dtype='float32'))
    >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> loss = OperatorFunction.apply(odl_func, x)
    >>> loss
    tensor(14., grad_fn=<OperatorFunctionBackward>)
    >>> loss.backward()
    >>> x.grad  # should be 2 * x
    tensor([2., 4., 6.])
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> xs = torch.stack([x, 2 * x], dim=0).requires_grad_(True)
    >>> loss = OperatorFunction.apply(odl_func, xs).sum()
    >>> loss
    tensor(70., grad_fn=<SumBackward0>)
    >>> loss.backward()
    >>> xs.grad  # should be 2 * xs
    tensor([[ 2.,  4.,  6.],
            [ 4.,  8., 12.]])

    Note, however, that the functional does not automatically reduce over
    extra axes, hence it cannot be used directly as a loss function. In
    addition, ODL functionals always take a single input.
    Loss functions of type ``loss_func(input, target)`` with reduction can
    be implemented e.g. as follows:

    >>> l2sq = odl.solvers.L2NormSquared(odl.rn(3, dtype='float32'))
    >>>
    >>> def my_mse(input, target, reduction='mean'):
    ...     val = OperatorFunction.apply(l2sq, input - target)
    ...     if reduction == 'mean':
    ...         return val.mean()
    ...     elif reduction == 'sum':
    ...         return val.sum()
    ...     elif reduction == 'none':
    ...         return val
    ...     else:
    ...         raise ValueError('bad reduction')
    ...
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> xs = torch.stack([x, 2 * x], dim=0).requires_grad_(True)
    >>> ys = torch.stack([x, 2 * x], dim=0) + 1
    >>> loss = my_mse(xs, ys, reduction='sum')
    >>> loss
    tensor(6., grad_fn=<SumBackward0>)
    >>> loss.backward()
    >>> xs.grad
    tensor([[-2., -2., -2.],
            [-2., -2., -2.]])
    """

    @staticmethod
    def forward(ctx, operator, input):
        """Evaluate forward pass on the input.

        Parameters
        ----------
        ctx : context object
            Object to communicate information between forward and backward
            passes.
        operator : `Operator`
            ODL operator to be wrapped. For gradient computations to
            work, ``operator.derivative(x).adjoint`` must be implemented.
        input : `torch.Tensor`
            Point at which to evaluate the operator.

        Returns
        -------
        result : `torch.Tensor`
            Tensor holding the result of the evaluation.
        """
        if not isinstance(operator, Operator):
            raise TypeError(
                "`operator` must be an `Operator` instance, got {!r}"
                "".format(operator)
            )

        # Save operator for backward; input only needs to be saved if
        # the operator is nonlinear (for `operator.derivative(input)`)
        ctx.operator = operator

        if not operator.is_linear:
            # Only needed for nonlinear operators
            ctx.save_for_backward(input)

        # TODO(kohr-h): use GPU memory directly when possible
        # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
        # is required
        input_arr = copy_if_zero_strides(input.cpu().detach().numpy())

        # Determine how to loop over extra shape "left" of the operator
        # domain shape
        in_shape = input_arr.shape
        op_in_shape = operator.domain.shape
        if operator.is_functional:
            op_out_shape = ()
            op_out_dtype = operator.domain.dtype
        else:
            op_out_shape = operator.range.shape
            op_out_dtype = operator.range.dtype

        extra_shape = in_shape[:-len(op_in_shape)]
        if in_shape[-len(op_in_shape):] != op_in_shape:
            shp_str = str(op_in_shape).strip('(,)')
            raise ValueError(
                'input tensor has wrong shape: expected (*, {}), got {}'
                ''.format(shp_str, in_shape)
            )

        # Store some information on the context object
        ctx.op_in_shape = op_in_shape
        ctx.op_out_shape = op_out_shape
        ctx.extra_shape = extra_shape
        ctx.op_in_dtype = operator.domain.dtype
        ctx.op_out_dtype = op_out_dtype

        # Evaluate the operator on all inputs in a loop
        if extra_shape:
            # Multiple inputs: flatten extra axes, then do one entry at a time
            input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
            results = []
            for inp in input_arr_flat_extra:
                results.append(operator(inp))

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_out_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_out_shape)
        else:
            # Single input: evaluate directly
            result_arr = np.asarray(
                operator(input_arr)
            ).astype(op_out_dtype, copy=False)

        # Convert back to tensor
        tensor = torch.from_numpy(result_arr).to(input.device)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        r"""Apply the adjoint of the derivative at ``grad_output``.

        This method is usually not called explicitly but as a part of the
        ``backward()`` pass of a backpropagation step.

        Parameters
        ----------
        ctx : context object
            Object to communicate information between forward and backward
            passes.
        grad_output : `torch.Tensor`
            Tensor to which the Jacobian should be applied. See Notes
            for details.

        Returns
        -------
        gradients : tuple
            Tuple ``(None, grad_input)``, where the ``None`` part is due to
            the first argument of ``forward`` being the ODL operator that
            does not require a gradient. The ``grad_input`` tensor is the
            result of applying the Jacobian to ``grad_output``.
            See Notes for details.

        Notes
        -----
        This method applies the contribution of this node, i.e., the
        transpose of the Jacobian of its outputs with respect to its inputs,
        to the gradients of some cost function with respect to the outputs
        of this node.

        **Example:** Assume that this node computes :math:`x \mapsto C(f(x))`,
        where :math:`x` is a tensor and :math:`C` is a scalar-valued
        function. In ODL language, what ``backward`` should compute is

        .. math::
            \nabla(C \circ f)(x) = f'(x)^*\big(\nabla C (f(x))\big)

        according to the chain rule. In ODL code, this corresponds to ::

            f.derivative(x).adjoint(C.gradient(f(x))).

        Hence, the parameter ``grad_output`` is a tensor containing
        :math:`y = \nabla C(f(x))`. Then, ``backward`` boils down to
        computing ``[f'(x)^*(y)]`` using the input ``x`` stored during
        the previous `forward` pass.
        """
        # Return early if there's nothing to do
        if not ctx.needs_input_grad[1]:
            return None, None

        operator = ctx.operator

        # Get `operator` and `input` from the context object (the input
        # is only needed for nonlinear operators)
        if not operator.is_linear:
            # TODO: implement directly for GPU data
            # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
            # is required
            input_arr = copy_if_zero_strides(
                ctx.saved_tensors[0].detach().cpu().numpy()
            )

        # ODL weights spaces, pytorch doesn't, so we need to handle this
        try:
            dom_weight = operator.domain.weighting.const
        except AttributeError:
            dom_weight = 1.0
        try:
            ran_weight = operator.range.weighting.const
        except AttributeError:
            ran_weight = 1.0
        scaling = dom_weight / ran_weight

        # Convert `grad_output` to NumPy array
        grad_output_arr = copy_if_zero_strides(
            grad_output.detach().cpu().numpy()
        )

        # Get shape information from the context object
        op_in_shape = ctx.op_in_shape
        op_out_shape = ctx.op_out_shape
        extra_shape = ctx.extra_shape
        op_in_dtype = ctx.op_in_dtype

        # Check if `grad_output` is consistent with `extra_shape` and
        # `op_out_shape`
        if grad_output_arr.shape != extra_shape + op_out_shape:
            raise ValueError(
                'expected tensor of shape {}, got shape {}'
                ''.format(extra_shape + op_out_shape, grad_output_arr.shape)
            )

        # Evaluate the (derivative) adjoint on all inputs in a loop
        if extra_shape:
            # Multiple gradients: flatten extra axes, then do one entry
            # at a time
            grad_output_arr_flat_extra = grad_output_arr.reshape(
                (-1,) + op_out_shape
            )

            results = []
            if operator.is_linear:
                for ograd in grad_output_arr_flat_extra:
                    results.append(np.asarray(operator.adjoint(ograd)))
            else:
                # Need inputs, flattened in the same way as the gradients
                input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
                for ograd, inp in zip(
                    grad_output_arr_flat_extra, input_arr_flat_extra
                ):
                    results.append(
                        np.asarray(operator.derivative(inp).adjoint(ograd))
                    )

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_in_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_in_shape)
        else:
            # Single gradient: evaluate directly
            if operator.is_linear:
                result_arr = np.asarray(
                    operator.adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)
            else:
                result_arr = np.asarray(
                    operator.derivative(input_arr).adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)

        # Apply scaling, convert to tensor and return
        if scaling != 1.0:
            result_arr *= scaling
        grad_input = torch.from_numpy(result_arr).to(grad_output.device)
        return None, grad_input  # return `None` for the `operator` part


class OperatorModule(torch.nn.Module):

    """Wrapper of an ODL operator as a ``torch.nn.Module``.

    This wrapper can be used as a layer in ``pytorch`` Neural Networks.
    It works with arbitrary batches and channels and supports
    backpropagation.

    Parameters
    ----------
    operator : `Operator`
        The ODL operator to be wrapped. For gradient computations to work,
        ``operator.derivative(x).adjoint`` must be implemented.

    Examples
    --------
    Simple example of using wrapping a ``MatrixOperator`` as a ``Module``.
    The input must have at least one extra dimension (batch axis), i.e.,
    in this case must be a 2D tensor:

    >>> matrix = np.array([[1, 0, 0],
    ...                    [0, 1, 1]], dtype='float32')
    >>> odl_op = odl.MatrixOperator(matrix)
    >>> odl_op.domain.shape
    (3,)
    >>> odl_op.range.shape
    (2,)
    >>> op_mod = OperatorModule(odl_op)
    >>> x = torch.ones((1, 3))  # with trivial batch axis
    >>> op_mod(x)
    tensor([[1., 2.]])
    >>> t = torch.ones(3)
    >>> x = torch.stack([0 * t, 1 * t])  # batch size 2
    >>> op_mod(x)
    tensor([[0., 0.],
            [1., 2.]])

    An arbitrary number of axes is supported:

    >>> x = t[None, None, :]  # trivial batch and channel
    >>> op_mod(x)
    tensor([[[1., 2.]]])
    >>> x = torch.stack([torch.stack([0 * t, 1 * t]),
    ...                  torch.stack([2 * t, 3 * t]),
    ...                  torch.stack([4 * t, 5 * t])])
    >>> op_mod(x)
    tensor([[[ 0.,  0.],
             [ 1.,  2.]],
    <BLANKLINE>
            [[ 2.,  4.],
             [ 3.,  6.]],
    <BLANKLINE>
            [[ 4.,  8.],
             [ 5., 10.]]])

    Backpropagation works autmatically by means of the
    ``operator.derivative(x).adjoint`` machinery. To trigger it, the
    input tensor must be marked as requiring gradient:

    >>> x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    >>> loss = op_mod(x).sum()
    >>> loss
    tensor(6., grad_fn=<SumBackward0>)
    >>> loss.backward()
    >>> x.grad
    tensor([[1., 1., 1.]])
    """

    def __init__(self, operator):
        """Initialize a new instance."""
        super(OperatorModule, self).__init__()
        self.operator = operator

    def forward(self, x):
        """Compute forward-pass of this module on ``x``.

        Parameters
        ----------
        x : `torch.Tensor`
            Input of this layer. The contained tensor must have shape
            ``extra_shape + operator.domain.shape``, and
            ``len(extra_shape)`` must be at least 1 (batch axis).

        Returns
        -------
        out : `torch.Tensor`
            The computed output. Its tensor will have shape
            ``extra_shape + operator.range.shape``, where ``extra_shape``
            are the extra axes of ``x``.

        Examples
        --------
        """
        in_shape = tuple(x.shape)
        in_ndim = len(in_shape)
        op_in_shape = self.operator.domain.shape
        op_in_ndim = len(op_in_shape)
        if in_ndim <= op_in_ndim or in_shape[-op_in_ndim:] != op_in_shape:
            shp_str = str(op_in_shape).strip('()')
            raise ValueError(
                'input tensor has wrong shape: expected (N, *, {}), got {}'
                ''.format(shp_str, in_shape)
            )
        return OperatorFunction.apply(self.operator, x)

    def __repr__(self):
        """Return ``repr(self)``."""
        op_name = self.operator.__class__.__name__
        op_in_shape = self.operator.domain.shape
        if len(op_in_shape) == 1:
            op_in_shape = op_in_shape[0]
        op_out_shape = self.operator.range.shape
        if len(op_out_shape) == 1:
            op_out_shape = op_out_shape[0]

        return '{}({}) ({} -> {})'.format(
            self.__class__.__name__, op_name, op_in_shape, op_out_shape
        )


def copy_if_zero_strides(arr):
    """Workaround for NumPy issue #9165 with 0 in arr.strides."""
    assert isinstance(arr, np.ndarray)
    return arr.copy() if 0 in arr.strides else arr


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    import odl
    from torch import autograd, nn
    run_doctests(extraglobs={'np': np, 'odl': odl, 'torch': torch,
                             'nn': nn, 'autograd': autograd})
