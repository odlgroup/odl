# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for converting ODL operators to tensorflow layers."""

from __future__ import print_function, division, absolute_import
import numpy as np
import odl
import uuid
import tensorflow as tf
from tensorflow.python.framework import ops


__all__ = ('as_tensorflow_layer',)


def as_tensorflow_layer(odl_op, name='ODLOperator',
                        differentiable=True):
    """Convert `Operator` or `Functional` to a tensorflow layer.

    Parameters
    ----------
    odl_op : `Operator` or `Functional`
        The operator that should be wrapped as a tensorflow layer.
    name : str
        Default name for tensorflow layers created.
    differentiable : boolean
        ``True`` if the layer should be differentiable, in which case
        ``odl_op`` should implement `Operator.derivative` which in turn
        implements `Operator.adjoint`. In this case, the adjoint of the
        derivative is properly wrapped in ``tensorflow_layer``, and
        gradients propagate as expected.

        If ``False``, the gradient is defined as everywhere zero.

    Returns
    -------
    tensorflow_layer : callable
        Callable that, when called with a `tensorflow.Tensor` of shape
        ``(n,) + odl_op.domain.shape + (1,)`` where ``n`` is the batch size,
        returns another `tensorflow.Tensor` which is a lazy evaluation of
        ``odl_op``.

        If ``odl_op`` is an `Operator`, the shape of the returned tensor is
        ``(n,) + odl_op.range.shape + (1,)``.

        If ``odl_op`` is an `Functional`, the shape of the returned tensor is
        ``(n,)``.

        The ``dtype`` of the tensor is ``odl_op.range.dtype``.
    """
    default_name = name

    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
        """Define custom py_func which takes also a grad op as argument.

        We need to overwrite this function since the default tensorflow
        `tf.py_func` does not support custom gradients.

        See tensorflow `issue #1095`_ for more information.

        Parameters
        ----------
        func : callable
            Python function that takes and returns numpy arrays.
        inp : sequence of `tensorflow.Tensor`
            Input tensors for the function
        Tout : sequence of `tensorflow.dtype`
            Datatype of the output(s).
        stateful : bool, optional
            If the function has internal state, i.e. if calling the function
            with a given input repeatedly could give different output.
        name : string, optional
            Name of the python function.
        grad : callbable, optional
            Gradient of the function.

        References
        ----------
        .. _issue #1095: https://github.com/tensorflow/tensorflow/issues/1095
        """
        if grad is None:
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        else:
            if stateful:
                override_name = 'PyFunc'
            else:
                override_name = 'PyFuncStateless'

            # Need to generate a unique name to avoid duplicates:
            rnd_name = override_name + 'Grad' + str(uuid.uuid4())

            tf.RegisterGradient(rnd_name)(grad)
            g = tf.get_default_graph()

            with g.gradient_override_map({override_name: rnd_name}):
                return tf.py_func(func, inp, Tout, stateful=stateful,
                                  name=name)

    def tensorflow_layer_grad_impl(x, dy, name):
        """Implementation of the tensorflow gradient.

        Gradient in tensorflow is equivalent to the adjoint of the derivative
        in ODL.

        Returns a `tensorflow.Tensor` that represents a lazy application of ::

            odl_op.derivative(x).adjoint(dy)

        Parameters
        ----------
        x : `numpy.ndarray`
            Point(s) in which the derivative should be taken.
            If ``odl_op`` is an `Operator` the axes are:
                0 : batch id. This is a constant if ``fixed_size`` is
                    ``True``, otherwise it is dynamic.
                1, ..., n-2 : spatial dimensions of data.
                n-1 : (currently) unused data channel.
            If ``odl_op`` is a `Functional` the axes are:
                0 : batch id.
        dy : `tensorflow.Tensor`
            Point(s) in which the adjoint of the derivative of the
            operator should be evaluated.
            The axes are:
                0 : batch id. Should be pairwise matched with ``x``.
                1, ..., m-2 : spatial dimensions of data.
                m-1 : (currently) unused data channel.
        name : string
            Name of the tensor.

        Returns
        -------
        result : `tensorflow.Tensor`
            Lazy result of the computation.
            If ``odl_op`` is an `Operator` the axes are:
                0 : batch id.
                1, ..., n-2 : spatial dimensions of data.
                n-1 : (currently) unused data channel.
            If ``odl_op`` is a `Functional` the axes are:
                0 : batch id.
        """
        with tf.name_scope(name):
            # Validate the input/output shape
            x_shape = x.get_shape()
            dy_shape = dy.get_shape()
            try:
                # Lazy check if the first dimension is dynamic
                n_x = int(x_shape[0])
                fixed_size = True
            except TypeError:
                n_x = x_shape[0]
                fixed_size = False

            if odl_op.is_functional:
                in_shape = (n_x,)
            else:
                in_shape = (n_x,) + space_shape(odl_op.range) + (1,)
            out_shape = (n_x,) + space_shape(odl_op.domain) + (1,)

            assert x_shape[1:] == space_shape(odl_op.domain) + (1,)
            if odl_op.is_functional:
                assert dy_shape[1:] == ()
            else:
                assert dy_shape[1:] == space_shape(odl_op.range) + (1,)

            def _impl(x, dy):
                """Implementation of the adjoint of the derivative.

                Returns ::

                    odl_op.derivative(x).adjoint(dy)

                Parameters
                ----------
                x : `numpy.ndarray`
                    Point(s) in which the derivative should be taken.
                    If ``odl_op`` is an `Operator` the axes are:
                        0 : batch id. This is a constant if ``fixed_size`` is
                            true, otherwise it is dynamic.
                        1, ..., n-2 : spatial dimensions of data.
                        n-1 : (currently) unused data channel.
                    If ``odl_op`` is a `Functional` the axes are:
                        0 : batch id.
                dy : `numpy.ndarray`
                    Point(s) in which the adjoint of the derivative of the
                    operator should be evaluated.
                    The axes are:
                        0 : batch id. Should be pairwise matched with ``x``.
                        1, ..., m-2 : spatial dimensions of data.
                        m-1 : (currently) unused data channel.

                Returns
                -------
                result : `numpy.ndarray`
                    Result of the computation.

                    If ``odl_op`` is an `Operator` the axes are:
                        0 : batch id.
                        1, ..., n-2 : spatial dimensions of data.
                        n-1 : (currently) unused data channel.
                    If ``odl_op`` is a `Functional` the axes are:
                        0 : batch id.
                """
                # Validate the shape of the given input
                if fixed_size:
                    x_out_shape = out_shape
                    assert x.shape == out_shape
                    assert dy.shape == in_shape
                else:
                    x_out_shape = (x.shape[0],) + out_shape[1:]
                    assert x.shape[1:] == out_shape[1:]
                    assert dy.shape[1:] == in_shape[1:]

                # Evaluate the operator on all inputs in the batch.
                out = np.empty(x_out_shape, odl_op.domain.dtype)
                out_element = odl_op.domain.element()
                for i in range(x_out_shape[0]):
                    if odl_op.is_functional:
                        xi = x[i, ..., 0]
                        dyi = dy[i]
                        out[i, ..., 0] = np.asarray(odl_op.gradient(xi)) * dyi
                    else:
                        xi = x[i, ..., 0]
                        dyi = dy[i, ..., 0]
                        odl_op.derivative(xi).adjoint(dyi,
                                                      out=out_element)
                        out[i, ..., 0] = np.asarray(out_element)

                # Rescale the domain/range according to the weighting since
                # tensorflow does not have weighted spaces.
                try:
                    dom_weight = odl_op.domain.weighting.const
                except AttributeError:
                    dom_weight = 1.0

                try:
                    ran_weight = odl_op.range.weighting.const
                except AttributeError:
                    ran_weight = 1.0

                scale = dom_weight / ran_weight
                out *= scale

                return out

            with ops.name_scope(name + '_pyfunc', values=[x, dy]) as name_call:
                result = py_func(_impl,
                                 [x, dy],
                                 [odl_op.domain.dtype],
                                 name=name_call,
                                 stateful=False)

                # We must manually set the output shape since tensorflow cannot
                # figure it out
                result = result[0]
                result.set_shape(out_shape)
                return result

    def tensorflow_layer(x, name=None):
        """Implementation of the tensorflow call.

        Returns a `tensorflow.Tensor` that represents a lazy application of
        ``odl_op`` to ``x``.

        Parameters
        ----------
        x : `tensorflow.Tensor`
            Point(s) to which the layer should be applied.
            The axes are:
                0 : batch id. Can be fixed or dynamic.
                1, ..., n-2 : spatial dimensions of data.
                n-1 : (currently) unused data channel.
        name : string
            Name of the tensor. Default: Defaultname.

        Returns
        -------
        result : `tensorflow.Tensor`
            Lazy result of the computation.
            If ``odl_op`` is an `Operator` the axes are:
                0 : batch id.
                1, ..., m-2 : spatial dimensions of data.
                m-1 : (currently) unused data channel.
            If ``odl_op`` is a `Functional` the axes are:
                0 : batch id.
        """
        if name is None:
            name = default_name

        with tf.name_scope(name):
            # Validate input shape
            x_shape = x.get_shape()
            try:
                # Lazy check if the first dimension is dynamic
                n_x = int(x_shape[0])
                fixed_size = True
            except TypeError:
                n_x = x_shape[0]
                fixed_size = False

            in_shape = (n_x,) + space_shape(odl_op.domain) + (1,)
            if odl_op.is_functional:
                out_shape = (n_x,)
            else:
                out_shape = (n_x,) + space_shape(odl_op.range) + (1,)

            assert x_shape[1:] == space_shape(odl_op.domain) + (1,)

            out_dtype = getattr(odl_op.range, 'dtype',
                                odl_op.domain.dtype)

            def _impl(x):
                """Implementation of the tensorflow layer.

                Parameters
                ----------
                x : `numpy.ndarray`
                    Point(s) in which the operator should be evaluated.
                    The axes are:
                        0 : batch id. This is a constant if ``fixed_size`` is
                            true, otherwise it is dynamic.
                        1, ..., n-2 : spatial dimensions of data.
                        n-1 : (currently) unused data channel.

                Returns
                -------
                result : `numpy.ndarray`
                    Result of the computation.
                    The axes are:
                        0 : batch id. Data is pairwise matched with ``x``.
                        1, ..., m-2 : spatial dimensions of data.
                        m-1 : (currently) unused data channel.
                """
                # Validate input shape
                if fixed_size:
                    x_out_shape = out_shape
                    assert x.shape == in_shape
                else:
                    x_out_shape = (x.shape[0],) + out_shape[1:]
                    assert x.shape[1:] == in_shape[1:]

                # Evaluate the operator on all inputs in the batch.
                out = np.empty(x_out_shape, out_dtype)
                out_element = odl_op.range.element()
                for i in range(x_out_shape[0]):
                    if odl_op.is_functional:
                        out[i] = odl_op(x[i, ..., 0])
                    else:
                        odl_op(x[i, ..., 0], out=out_element)
                        out[i, ..., 0] = np.asarray(out_element)

                return out

            if differentiable:
                def tensorflow_layer_grad(op, grad):
                    """Thin wrapper for the gradient."""
                    x = op.inputs[0]
                    return tensorflow_layer_grad_impl(x, grad,
                                                      name=name + '_grad')
            else:
                tensorflow_layer_grad = None

            with ops.name_scope(name + '_pyfunc', values=[x]) as name_call:
                result = py_func(_impl,
                                 [x],
                                 [out_dtype],
                                 name=name_call,
                                 stateful=False,
                                 grad=tensorflow_layer_grad)

                # We must manually set the output shape since tensorflow cannot
                # figure it out
                result = result[0]
                result.set_shape(out_shape)
                return result

    return tensorflow_layer


def space_shape(space):
    """Return ``space.shape``, including power space base shape.

    If ``space`` is a power space, return ``(len(space),) + space[0].shape``,
    otherwise return ``space.shape``.
    """
    if isinstance(space, odl.ProductSpace) and space.is_power_space:
        return (len(space),) + space[0].shape
    else:
        return space.shape


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
