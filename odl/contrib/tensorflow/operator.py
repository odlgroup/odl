# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for converting ODL spaces to tensorflow layers."""

from __future__ import absolute_import, division, print_function

import numpy as np

import odl
import tensorflow as tf

__all__ = ('TensorflowOperator',)


class TensorflowOperator(odl.Operator):
    def __init__(self, input_tensor, output_tensor, linear=False, sess=None):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

        domain = odl.tensor_space(input_tensor.shape.as_list(),
                                  dtype=input_tensor.dtype.as_numpy_dtype)
        range = odl.tensor_space(output_tensor.shape.as_list(),
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

        super(TensorflowOperator, self).__init__(domain, range, linear=linear)

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
    from odl.util.testutils import run_doctests
    run_doctests()
