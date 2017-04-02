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

"""Utilities for interfacing ODL with Tensorflow."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from builtins import range, str
from future import standard_library
standard_library.install_aliases()

import odl
import numpy as np
import uuid
try:
    # TODO: Lazy import to improve performance
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False


__all__ = ('as_tensorflow_layer', 'TensorflowSpace', 'TensorflowSpaceOperator')


def as_tensorflow_layer(odl_op, default_name='ODLOperator',
                        differentiable=True):
    """Convert ``Operator`` to tensorflow layer.

    Parameters
    ----------
    odl_op : `Operator`
        The operator that should be wrapped to a tensorflow layer.
    default_name : str
        Default name for tensorflow layers created.
    differentiable : boolean
        True if the layer should be differentiable, otherwise assumes that
        the derivative is everywhere zero.

    Returns
    -------
    tensorflow_layer : callable
        Callable that, when called with an `tensorflow.Tensor` of shape
        `(n, *odl_op.domain.shape, 1)` returns a tensor of shape
        `(n, *odl_op.range.shape, 1)` where ``n`` is the number of batches.
        Hence for each evaluation, ``odl_op`` is called a total of ``n`` times.
        The `dtype` of the tensor is the same as the respective ODL spaces.

        If ``odl_op`` implements `Operator.derivative` which in turn implements
        `Operator.adjoint`, it is properly wrapped in ``tensorflow_layer``, and
        gradients propagate as expected.
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError('Tensorflow not installed')

    from tensorflow.python.framework import ops

    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
        """Define custom py_func which takes also a grad op as argument.

        We need to overwrite this function since the default tensorflow py_func
        does not support custom gradients.

        Parameters
        ----------
        func : callable
            Python function that takes and returns numpy arrays.
        inp : sequence of `tensorflow.Tensor`
            Input tensors for the function
        Tout : sequence of `tensorflow.dtype`
            Datatype of the output(s).
        stateful : bool
            If the function has internal state, i.e. if calling the function
            with a given input always gives the same output.
        name : string
            Name of the python function
        grad : callbable
            Gradient of the function.
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

    def tensorflow_layer_grad_impl(x, dx, name):
        """Implementation of the tensorflow gradient.

        Gradient in tensorflow is equivalent to the adjoint of the derivative
        in ODL.

        Returns a `tensorflow.Tensor` that represents a lazy application of ::

            odl_op.derivative(x).adjoint(dx)

        Parameters
        ----------
        x : `numpy.ndarray`
            Point(s) in which the derivative should be taken.
            If ``odl_op`` is an `Operator` the axes are:
                0 : batch id. This is a constant if ``fixed_size`` is
                    true, otherwise it is dynamic.
                1, ..., N-2 : spatial dimensions of data.
                n-1 : (currently) unused data channel.
            If ``odl_op`` is a `Functional` the axes are:
                0 : batch id.
        dx : `tensorflow.Tensor`
            Point(s) in which the adjoint of the derivative of the
            operator should be evaluated.
            The axes are:
                0 : batch id. Should be pairwise matched with ``x``.
                1, ..., M-2 : spatial dimensions of data.
                n-1 : (currently) unused data channel.
        name : string
            Name of the tensor.

        Returns
        -------
        result : `tensorflow.Tensor`
            Lazy result of the computation.
            If ``odl_op`` is an `Operator` the axes are:
                0 : batch id.
                1, ..., N-2 : spatial dimensions of data.
                n-1 : (currently) unused data channel.
            If ``odl_op`` is a `Functional` the axes are:
                0 : batch id.
        """
        with tf.name_scope(name):
            # Validate the input/output shape
            x_shape = x.get_shape()
            dx_shape = dx.get_shape()
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
                in_shape = (n_x,) + odl_op.range.shape + (1,)
            out_shape = (n_x,) + odl_op.domain.shape + (1,)

            assert x_shape[1:] == odl_op.domain.shape + (1,)
            if odl_op.is_functional:
                assert dx_shape[1:] == ()
            else:
                assert dx_shape[1:] == odl_op.range.shape + (1,)

            def _impl(x, dx):
                """Implementation of the adjoint of the derivative.

                Returns ::

                    odl_op.derivative(x).adjoint(dx)

                Parameters
                ----------
                x : `numpy.ndarray`
                    Point(s) in which the derivative should be taken.
                    If ``odl_op`` is an `Operator` the axes are:
                        0 : batch id. This is a constant if ``fixed_size`` is
                            true, otherwise it is dynamic.
                        1, ..., N-2 : spatial dimensions of data.
                        n-1 : (currently) unused data channel.
                    If ``odl_op`` is a `Functional` the axes are:
                        0 : batch id.
                dx : `numpy.ndarray`
                    Point(s) in which the adjoint of the derivative of the
                    operator should be evaluated.
                    The axes are:
                        0 : batch id. Should be pairwise matched with ``x``.
                        1, ..., M-2 : spatial dimensions of data.
                        n-1 : (currently) unused data channel.

                Returns
                -------
                result : `numpy.ndarray`
                    Result of the computation.

                    If ``odl_op`` is an `Operator` the axes are:
                        0 : batch id.
                        1, ..., N-2 : spatial dimensions of data.
                        n-1 : (currently) unused data channel.
                    If ``odl_op`` is a `Functional` the axes are:
                        0 : batch id.
                """
                # Validate the shape of the given input
                if fixed_size:
                    x_out_shape = out_shape
                    assert x.shape == out_shape
                    assert dx.shape == in_shape
                else:
                    x_out_shape = (x.shape[0],) + out_shape[1:]
                    assert x.shape[1:] == out_shape[1:]
                    assert dx.shape[1:] == in_shape[1:]

                # Evaluate the operator on all inputs in the batch.
                out = np.empty(x_out_shape, odl_op.domain.dtype)
                for i in range(x_out_shape[0]):
                    if odl_op.is_functional:
                        xi = x[i, ..., 0]
                        dxi = dx[i]
                        out[i, ..., 0] = np.asarray(odl_op.gradient(xi)) * dxi
                    else:
                        xi = x[i, ..., 0]
                        dxi = dx[i, ..., 0]
                        result = odl_op.derivative(xi).adjoint(dxi)
                        out[i, ..., 0] = np.asarray(result)

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

            with ops.name_scope(name + '_pyfunc', values=[x, dx]) as name_call:
                result = py_func(_impl,
                                 [x, dx],
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
                1, ..., M-2 : spatial dimensions of data.
                n-1 : (currently) unused data channel.
        name : string
            Name of the tensor. Default: Defaultname.

        Returns
        -------
        result : `tensorflow.Tensor`
            Lazy result of the computation.
            If ``odl_op`` is an `Operator` the axes are:
                0 : batch id.
                1, ..., N-2 : spatial dimensions of data.
                n-1 : (currently) unused data channel.
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

            in_shape = (n_x,) + odl_op.domain.shape + (1,)
            if odl_op.is_functional:
                out_shape = (n_x,)
            else:
                out_shape = (n_x,) + odl_op.range.shape + (1,)

            assert x_shape[1:] == odl_op.domain.shape + (1,)

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
                        1, ..., N-2 : spatial dimensions of data.
                        n-1 : (currently) unused data channel.

                Returns
                -------
                result : `numpy.ndarray`
                    Result of the computation.
                    The axes are:
                        0 : batch id. Data is pairwise matched with ``x``.
                        1, ..., M-2 : spatial dimensions of data.
                        n-1 : (currently) unused data channel.
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
                for i in range(x_out_shape[0]):
                    if odl_op.is_functional:
                        out[i] = odl_op(x[i, ..., 0])
                    else:
                        out[i, ..., 0] = np.asarray(odl_op(x[i, ..., 0]))

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


class TensorflowSpace(odl.LinearSpace):

    """A space of tensorflow Tensors."""

    def __init__(self, shape, name='ODLTensorflowSpace'):
        odl.LinearSpace.__init__(self, odl.RealNumbers())
        self.shape = tuple(tf.Dimension(si) if not isinstance(si, tf.Dimension)
                           else si
                           for si in shape)
        self.init_shape = tuple(si if si.value is not None
                                else tf.Dimension(1)
                                for si in self.shape)
        self.name = name

    def _lincomb(self, a, x1, b, x2, out):
        with tf.name_scope('{}_lincomb'.format(self.name)):
            if x1 is x2:
                # x1 is aligned with x2 -> out = (a+b)*x1
                out.data = (a + b) * x1.data
            elif out is x1 and out is x2:
                # All the vectors are aligned -> out = (a+b)*out
                if (a + b) != 1:
                    out.data *= (a + b)
            elif out is x1:
                # out is aligned with x1 -> out = a*out + b*x2
                out.data = a * out.data + b * x2.data
            elif out is x2:
                # out is aligned with x2 -> out = a*x1 + b*out
                out.data = a * x1.data + b * out.data
            else:
                # We have exhausted all alignment options, so x1 != x2 != out
                # We now optimize for various values of a and b
                if b == 0:
                    if a == 0:  # Zero assignment -> out = 0
                        out.data *= 0
                    else:  # Scaled copy -> out = a*x1
                        out.data = a * x1.data
                else:
                    if a == 0:  # Scaled copy -> out = b*x2
                        out.data = b * x2.data
                    elif a == 1:  # No scaling in x1 -> out = x1 + b*x2
                        out.data = x1.data + b * x2.data
                    else:  # Generic case -> out = a*x1 + b*x2
                        out.data = a * x1.data + b * x2.data

    def element(self, inp=None):
        if inp in self:
            return inp
        elif inp is None:
            return self.zero()
        else:
            return TensorflowSpaceElement(self, inp)

    def zero(self):
        with tf.name_scope('{}_zero'.format(self.name)):
            return self.element(tf.zeros(self.init_shape,
                                         dtype=tf.float32))

    def one(self):
        with tf.name_scope('{}_one'.format(self.name)):
            return self.element(tf.ones(self.init_shape,
                                        dtype=tf.float32))

    def __eq__(self, other):
        return isinstance(other, TensorflowSpace) and other.shape == self.shape

    def __repr__(self):
        return 'TensorflowSpace({})'.format(self.shape)


class TensorflowSpaceElement(odl.LinearSpaceElement):

    """Elements in TensorflowSpace."""

    def __init__(self, space, data):
        odl.LinearSpaceElement.__init__(self, space)
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, TensorflowSpaceElement):
            raise Exception(value.data)
        self._data = value

    def __repr__(self):
        return '{}.element({})'.format(self.space, self.data)


class TensorflowSpaceOperator(odl.Operator):

    """Wrap ODL operator so that it acts on TensorflowSpace elements."""

    def __init__(self, domain, range, func, adjoint=None, linear=False):
        odl.Operator.__init__(self, domain, range, linear)
        self.func = func
        self.adjoint_func = adjoint

    def _call(self, x):
        return self.func(x.data)

    @property
    def adjoint(self):
        return TensorflowSpaceOperator(self.range,
                                       self.domain,
                                       self.adjoint_func,
                                       self.func,
                                       self.is_linear)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
