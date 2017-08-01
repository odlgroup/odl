# Copyright 2014-2017 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Utilities for converting ODL spaces to tensorflow layers."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import tensorflow as tf
import numpy as np
import odl


__all__ = ('TensorflowOperator',)


class TensorflowOperator(odl.Operator):
    def __init__(self, input_tensor, output_tensor, linear=False, sess=None):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

        # TODO: Fix with tensors
        domain = odl.fn(np.prod(input_tensor.shape.as_list()),
                        dtype=input_tensor.dtype.as_numpy_dtype)
        range = odl.fn(np.prod(output_tensor.shape.as_list()),
                       dtype=output_tensor.dtype.as_numpy_dtype)

        self.dx = tf.placeholder(input_tensor.dtype,
                                 shape=input_tensor.shape)
        self.dy = tf.placeholder(output_tensor.dtype,
                                 shape=output_tensor.shape)

        adjoint_of_derivative_tensor = tf.gradients(
            self.output_tensor, [self.input_tensor], [self.dy])[0]
        self.adjoint_of_derivative_tensor = (range.weighting.const *
                                             adjoint_of_derivative_tensor)

        # Since tensorflow does not support forward differentiation, use trick
        # that adjoint of the derivative of adjoint of the derivative is simply
        # the derivative.
        derivative_tensor = tf.gradients(
            adjoint_of_derivative_tensor, [self.dy], [self.dx])[0]
        self.derivative_tensor = (range.weighting.const *
                                  derivative_tensor)

        if sess is None:
            self.sess = tf.get_default_session()
        else:
            self.sess = sess

        odl.Operator.__init__(self, domain, range, linear=linear)

    def _call(self, x):
        result = self.sess.run(self.output_tensor,
                               feed_dict={self.input_tensor: np.asarray(x)})

        return result

    def derivative(self, x):
        op = self

        class TensorflowOperatorDerivative(odl.Operator):
            def _call(self, dx):
                result = op.sess.run(op.derivative_tensor,
                                     feed_dict={op.input_tensor: np.asarray(x),
                                                op.dx: np.asarray(dx)})

                return result

            @property
            def adjoint(self):
                class TensorflowOperatorDerivativeAdjoint(odl.Operator):
                    def _call(self, y):
                        result = op.sess.run(
                            op.adjoint_of_derivative_tensor,
                            feed_dict={op.input_tensor: np.asarray(x),
                                       op.dy: np.asarray(y)})

                        return result

                return TensorflowOperatorDerivativeAdjoint(self.range,
                                                           self.domain,
                                                           linear=True)

        return TensorflowOperatorDerivative(self.domain,
                                            self.range,
                                            linear=True)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
