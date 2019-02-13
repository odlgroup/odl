# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for converting ODL operators to Theano operators."""

from __future__ import print_function, division, absolute_import
from future.utils import native

import theano
import numpy as np

from odl.solvers import Functional


__all__ = ('TheanoOperator',)


class TheanoOperator(theano.Op):

    """Wrap an ODL operator as a Theano operator.

    The operator works with Theanos automatic differentiation if the operator
    implements `Operator.derivative` and `Operator.adjoint`.
    """

    # Properties used by Theano for __eq__, __hash__ and __repr__
    __props__ = ('operator',)

    def __init__(self, operator):
        """Initialize an instance.

        Parameters
        ----------
        operator : `Operator`
            The operator that should be wrapped, must map from a
            `TensorSpace` to a `TensorSpace`.

        Examples
        --------
        Make a vector-to-vector operator:

        >>> space = odl.rn(3)
        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype=float)
        >>> op = odl.MatrixOperator(matrix, domain=space)
        >>> matrix_op = TheanoOperator(op)
        >>> matrix_op.operator is op
        True

        Create a functional, i.e., an operator with scalar output:

        >>> space = odl.rn(3)
        >>> functional = odl.solvers.L2NormSquared(space)
        >>> func_op = TheanoOperator(functional)
        >>> x = theano.tensor.dvector()
        >>> apply = func_op.make_node(x)
        >>> apply.outputs[0].type()
        <TensorType(float64, scalar)>
        """
        self.operator = operator

    def make_node(self, x):
        """Create a node for the computation graph.

        Parameters
        ----------
        x : `theano.tensor.var.TensorVariable`
            Input to the node.

        Returns
        -------
        node : `theano.gof.graph.Apply`
            Node for the Theano expression graph. Its only input is ``x``,
            and the output is of the same type.
        """
        x = theano.tensor.as_tensor_variable(x)

        # Create tensor type with correct dtype.
        # The second argument specifies the number of dimensions of the output.
        # False means that we do not support broadcasting.
        if isinstance(self.operator, Functional):
            # Make scalar out type
            out_type = theano.tensor.TensorVariable(
                theano.tensor.TensorType(self.operator.domain.dtype, ()))
        else:
            out_type = theano.tensor.TensorVariable(
                theano.tensor.TensorType(
                    self.operator.range.dtype,
                    [False] * len(self.operator.range.shape)))

        return theano.Apply(self, [x], [out_type.type()])

    def perform(self, node, inputs, output_storage):
        """Evaluate this node's computation.

        Parameters
        ----------
        node : `theano.gof.graph.Apply`
            The node of this Op in the computation graph.
        inputs : 1-element list of arrays
            Contains an array (usually `numpy.ndarray`) of concrete values
            supplied for the symbolic input variable ``x``.
        output_storage : 1-element list of 1-element lists
            The single 1-element list contained in ``output_storage``
            by default contains only ``None``. This value must be replaced
            by the result of the application of `odl_op`.

        Examples
        --------
        Perform a matrix multiplication:

        >>> space = odl.rn(3)
        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype=float)
        >>> op = odl.MatrixOperator(matrix, domain=space)
        >>> matrix_op = TheanoOperator(op)
        >>> x = theano.tensor.dvector()
        >>> op_x = matrix_op(x)
        >>> op_func = theano.function([x], op_x)
        >>> op_func([1, 2, 3])
        array([ 4.,  5.])

        Evaluate a functional, i.e., an operator with scalar output:

        >>> space = odl.rn(3)
        >>> functional = odl.solvers.L2NormSquared(space)
        >>> func_op = TheanoOperator(functional)
        >>> x = theano.tensor.dvector()
        >>> op_x = func_op(x)
        >>> op_func = theano.function([x], op_x)
        >>> op_func([1, 2, 3])
        array(14.0)
        """
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.asarray(self.operator(x))

    def infer_shape(self, node, input_shapes):
        """Return a list of output shapes based on ``input_shapes``.

        This method is optional. It allows to compute the shape of the
        output without having to evaluate.

        Parameters
        ----------
        node : `theano.gof.graph.Apply`
            The node of this Op in the computation graph.
        input_shapes : 1-element list of `theano.compile.ops.Shape`
            Symbolic shape of the input.

        Returns
        -------
        output_shapes : 1-element list of tuples
            Fixed shape of the output determined by `odl_op`.
        """
        if isinstance(self.operator, Functional):
            return [()]
        else:
            # Need to convert to native to avoid error in Theano from
            # future.int
            return [tuple(native(si) for si in self.operator.range.shape)]

    def grad(self, inputs, output_grads):
        r"""Apply adjoint of the Jacobian at ``inputs`` to ``output_grads``.

        Parameters
        ----------
        inputs : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic input to the gradient, the point at which the
            Jacobian is computed.
        output_grads : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic gradient from the subsequent node received during
            backpropagation. The adjoint of the Jacobian is applied to
            this variable.

        Examples
        --------
        Compute the Jacobian adjoint of the matrix operator, which is the
        operator of the transposed matrix. We compose with the ``sum``
        functional to be able to evaluate ``grad``:

        >>> space = odl.rn(3)
        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype=float)
        >>> op = odl.MatrixOperator(matrix, domain=space)
        >>> matrix_op = TheanoOperator(op)
        >>> x = theano.tensor.dvector()
        >>> op_x = matrix_op(x)
        >>> cost = op_x.sum()
        >>> cost_grad = theano.grad(cost, x)
        >>> cost_grad_func = theano.function([x], cost_grad)
        >>> cost_grad_func([1, 2, 3])
        array([ 1.,  1.,  2.])
        >>> sum_grad = np.array([1.0, 1.0])
        >>> np.allclose(cost_grad_func([1, 2, 3]), matrix.T.dot(sum_grad))
        True

        Compute the gradient of a custom functional:

        >>> space = odl.rn(3)
        >>> functional = odl.solvers.L2NormSquared(space)
        >>> func_op = TheanoOperator(functional)
        >>> x = theano.tensor.dvector()
        >>> op_x = func_op(x)
        >>> grad_x = theano.grad(op_x, x)
        >>> grad_func = theano.function([x], grad_x)
        >>> grad_func([1, 2, 3])  # should be 2 * input
        array([ 2.,  4.,  6.])

        Notes
        -----
        This method applies the contribution of this node, i.e., the Jacobian
        of its outputs with respect to its inputs, to the gradients of some
        cost function with respect to the outputs of this node.

        Example: Assume that this node computes :math:`x \mapsto f(x)`, and
        its output is connected to a cost function that computes
        :math:`y --> C(y)`. Here, :math:`x`, :math:`y` and :math:`f(x)` are
        tensor variables and :math:`C(y)` is a scalar variable.
        In ODL language, what ``grad`` should compute is

            .. math::
                \nabla(C \circ f)(x) = f'(x)^*\big(\nabla C (f(x))\big)

        according to the chain rule. In ODL code, this corresponds to ::

            f.derivative(x).adjoint(C.gradient(f(x))).

        Then, the parameter ``output_grads`` contains a single tensor
        variable ``y`` that stands for :math:`\nabla C(f(x))`. Thus,
        ``grad`` boils down to taking the ``output_grads`` ``[y]`` and
        return ``[f'(x)^*(y)]`` symbolically, where ``inputs == [x]``.

        This turns out to be just a special case of `R_op`, which is the
        exact same operation, only for arbitrary ``eval_points`` instead of
        ``output_grads``.
        """
        return self.R_op(inputs, output_grads)

    def R_op(self, inputs, eval_points):
        """Apply the adjoint of the Jacobian at ``inputs`` to ``eval_points``.

        This is the symbolic counterpart of ODL's ::

            op.derivative(x).adjoint(v)

        See `grad` for its usage.

        Parameters
        ----------
        inputs : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic input to the gradient, the point at which the
            Jacobian is computed.
        eval_points : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic input to the adjoint of the Jacobian, i.e., the
            variable to which the Jacobian adjoint should be applied.

        Returns
        -------
        outputs : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic result of the application of the Jacobian adjoint.
            It uses a wrapper class ``OdlDerivativeAdjointAsTheanoROp``
            for ``(x, v) --> op.derivative(x).adjoint(v)``.
        """
        # ODL weights spaces, Theano does not. We need to handle this
        try:
            dom_weight = self.operator.domain.weighting.const
        except AttributeError:
            dom_weight = 1.0

        try:
            ran_weight = self.operator.range.weighting.const
        except AttributeError:
            ran_weight = 1.0

        scale = dom_weight / ran_weight

        op = self

        class TheanoJacobianAdjoint(theano.Op):

            __props__ = ()

            """Wrap ``op.derivative(x).adjoint(v)`` into a Theano Op.

            This Op has two inputs, ``x`` and ``v``, where ``x``
            is the point at which the Jacobian is taken, and ``v`` the
            tensor to which its adjoint is applied. There is only one output,
            which is of the same type as ``v`` (and ``x``).
            """

            def make_node(self, x, v):
                """Create a node for the computation graph."""
                x = theano.tensor.as_tensor_variable(x)
                v = theano.tensor.as_tensor_variable(v)
                return theano.Apply(self, [x, v], [x.type()])

            def perform(self, node, inputs_storage, output_storage):
                """Evaluate this node's computation.

                This method computes ::

                    op.derivative(x).adjoint(v)
                """
                x = inputs_storage[0]
                v = inputs_storage[1]
                out = output_storage[0]
                out[0] = np.asarray(op.operator.derivative(x).adjoint(v))
                if scale != 1.0:
                    out[0] *= scale

            def infer_shape(self, node, input_shapes):
                """Return a list of output shapes based on ``input_shapes``."""
                # Need to convert to native to avoid error in theano from
                # future.int
                return [tuple(native(si) for si in op.operator.domain.shape)]

        r_op = TheanoJacobianAdjoint()
        r_op_apply = r_op(inputs[0], eval_points[0])
        return [r_op_apply]


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
