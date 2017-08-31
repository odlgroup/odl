import odl
import tensorflow as tf
import numpy as np


__all__ = ('as_tensorflow_layer', 'TensorflowSpace', 'TensorflowSpaceOperator')


def as_tensorflow_layer(odl_op, name='ODLOperator', differentiable=True):
    """Convert ``Operator`` to tensorflow layer.

    Parameters
    ----------
    odl_op : `Operator`
        The operator that should be wrapped to a tensorflow layer.
    name : str
        Tensorflow name of the operator
    differentiable : boolean
        True if the operator should be differentiable, otherwise assumes that
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
    import tensorflow as tf
    from tensorflow.python.framework import ops

    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
        """Define custom py_func which takes also a grad op as argument."""
        if grad is None:
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        else:
            # Need to generate a unique name to avoid duplicates:
            rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

            tf.RegisterGradient(rnd_name)(grad)
            g = tf.get_default_graph()

            if stateful:
                override_name = 'PyFunc'
            else:
                override_name = 'PyFuncStateless'

            with g.gradient_override_map({override_name: rnd_name}):
                return tf.py_func(func, inp, Tout, stateful=stateful,
                                  name=name)

    def tensorflow_layer_grad_impl(x, dx):
        """Implementation of the tensorflow gradient.

        Gradient in tensorflow is equivalent to the adjoint of the derivative
        in ODL.
        """
        x_shape = x.get_shape()
        dx_shape = dx.get_shape()
        try:
            n_x = int(x_shape[0])
            fixed_size = True
        except TypeError:
            n_x = x_shape[0]
            fixed_size = False

        if odl_op.is_functional:
            in_shape = (n_x, 1)
        else:
            in_shape = (n_x,) + odl_op.range.shape + (1,)
        out_shape = (n_x,) + odl_op.domain.shape + (1,)

        # Validate input shape
        assert x_shape[1:] == odl_op.domain.shape + (1,)
        if odl_op.is_functional:
            assert dx_shape[1:] == (1,)
        else:
            assert dx_shape[1:] == odl_op.range.shape + (1,)

        def _impl(x, dx):
            if fixed_size:
                x_out_shape = out_shape
                assert x.shape == out_shape
                assert dx.shape == in_shape
            else:
                x_out_shape = (x.shape[0],) + out_shape[1:]
                assert x.shape[1:] == out_shape[1:]
                assert dx.shape[1:] == in_shape[1:]

            out = np.empty(x_out_shape, odl_op.domain.dtype)
            for i in range(x_out_shape[0]):
                if odl_op.is_functional:
                    xi = x[i, ..., 0]
                    dxi = dx[i, 0]
                    out[i, ..., 0] = np.asarray(odl_op.gradient(xi)) * dxi
                else:
                    xi = x[i, ..., 0]
                    dxi = dx[i, ..., 0]
                    result = odl_op.derivative(xi).adjoint(dxi)
                    out[i, ..., 0] = np.asarray(result)

            # TODO: Improve
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

        with ops.name_scope(name + 'Grad', "ODLTensorflowLayerGrad", [x, dx]):
            result = py_func(_impl,
                             [x, dx],
                             [tf.float32],
                             stateful=False)

            # We must manually set the output shape since tensorflow cannot
            # figure it out
            result = result[0]
            result.set_shape(out_shape)
            return result

    def tensorflow_layer_grad(op, grad):
        """Thin wrapper for the gradient."""
        x = op.inputs[0]
        return tensorflow_layer_grad_impl(x, grad)

    def tensorflow_layer(x):
        """Implementation of the tensorflow call.

        Gradient in tensorflow is equivalent to the adjoint of the derivative
        in ODL.
        """
        x_shape = x.get_shape()
        try:
            n_x = int(x_shape[0])
            fixed_size = True
        except TypeError:
            n_x = x_shape[0]
            fixed_size = False

        in_shape = (n_x,) + odl_op.domain.shape + (1,)
        if odl_op.is_functional:
            out_shape = (n_x, 1)
        else:
            out_shape = (n_x,) + odl_op.range.shape + (1,)

        # Validate input shape
        assert x_shape[1:] == odl_op.domain.shape + (1,)

        def _impl(x):
            if fixed_size:
                x_out_shape = out_shape
                assert x.shape == in_shape
            else:
                x_out_shape = (x.shape[0],) + out_shape[1:]
                assert x.shape[1:] == in_shape[1:]

            out = np.empty(x_out_shape, odl_op.domain.dtype)
            for i in range(x_out_shape[0]):
                if odl_op.is_functional:
                    out[i, 0] = odl_op(x[i, ..., 0])
                else:
                    out[i, ..., 0] = np.asarray(odl_op(x[i, ..., 0]))

            return out

        if differentiable:
            grad = tensorflow_layer_grad
        else:
            grad = None

        with ops.name_scope(name, "ODLTensorflowLayer", [x]) as name_call:
            result = py_func(_impl,
                             [x],
                             [tf.float32],
                             name=name_call,
                             stateful=False,
                             grad=grad)

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
        self.shape = tuple(tf.Dimension(si) if not isinstance(si, tf.Dimension) else si for si in shape)
        self.init_shape = tuple(si if si.value is not None else tf.Dimension(1) for si in self.shape)
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
