# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for converting ODL spaces to tensorflow layers."""

from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np

import odl


__all__ = ('TensorflowOperator',)


class TensorflowOperator(odl.Operator):
    def __init__(self, input_tensor, output_tensor,
                 domain=None, range=None,
                 linear=False, sess=None):
        """Wrap Tensorflow layers in ODL operator.

        Parameters
        ----------
        input_tensor : `tf.Tensor`
            Input to the tensorflow graph (values will be fed to this).
        output_tensor : `tf.Tensor`
            Output node from the graph.
        domain : `TensorSpace`, optional
            Domain of the wrapping operator.
            Default: `tensor_space` with same shape and dtype as
            ``input_tensor``.
        range : `TensorSpace`, optional
            Range of the wrapping operator.
            Default: `tensor_space` with same shape and dtype as
            ``output_tensor``.
        linear : bool, optional
            If the created operator should be linear
        sess : `tf.Session`, optional
            Session to evaluate the graph in.
            Default: `tf.get_default_session`

        Notes
        -----
        Both `Operator.derivative` and `Operator.adjoint` are implemented
        using automatic differentiation in tensorflow.
        """
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

        if domain is None:
            domain = odl.tensor_space(input_tensor.shape.as_list(),
                                      dtype=input_tensor.dtype.as_numpy_dtype)
        if range is None:
            range = odl.tensor_space(output_tensor.shape.as_list(),
                                     dtype=output_tensor.dtype.as_numpy_dtype)

        # TODO: replace with #1177
        self.adjoint_weight = domain.weighting.const / range.weighting.const

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
        """Return ``self(x)``."""
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

                        # Compensate for weighted spaces in ODL
                        result *= op.adjoint_weight
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
