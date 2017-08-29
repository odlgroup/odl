# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for converting ODL operators to theano operators."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
from future.utils import native
standard_library.install_aliases()

import theano
import numpy as np


__all__ = ('TheanoOperator',)


class TheanoOperator(theano.Op):
    """Wraps ODL operator to use with theano.

    The operator works with theanos automatic differentiation if the operator
    implements `Operator.derivative` and `Operator.adjoint`.
    """

    # Properties used by theano to check for equality
    __props__ = ('operator',)

    def __init__(self, operator):
        """Initialize an instance.

        Parameters
        ----------
        operator : Operator
            The operator that should be wrapped, must map `FnBase` spaces to
            `FnBase` spaces.
        """
        self.operator = operator

    def make_node(self, x):
        """Create node in for theanos computational graph."""
        x = theano.tensor.as_tensor_variable(x)

        # Create tensortype with correct dtype.
        # The second argument specifies the number of dimensions of the output
        # False means that we do not support broadcasting.
        out_type = theano.tensor.TensorType(
            self.operator.range.dtype,
            [False] * len(self.operator.range.shape))
        return theano.Apply(self, [x], [out_type()])

    def perform(self, node, inputs, output_storage):
        """Compute the result."""
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.asarray(self.operator(x))

    def infer_shape(self, node, i0_shapes):
        """Infer the output shape from input shape."""
        # Need to convert to native to avoid error in theano from future.int
        return [[native(si) for si in self.operator.range.shape]]

    def grad(self, inputs, output_grads):
        """Compute gradient (adjoint of derivative)."""
        if not self.operator.is_linear:
            raise NotImplementedError('Derivative of non-linear operators not '
                                      'yet implemented')

        # ODL weights spaces, theano does not. We need to handle this
        try:
            dom_weight = self.operator.domain.weighting.const
        except AttributeError:
            dom_weight = 1.0

        try:
            ran_weight = self.operator.range.weighting.const
        except AttributeError:
            ran_weight = 1.0

        scale = dom_weight / ran_weight
        if scale == 1.0:
            adjoint_operator = self.operator.adjoint
        else:
            adjoint_operator = scale * self.operator.adjoint

        return [TheanoOperator(adjoint_operator)(output_grads[0])]


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
