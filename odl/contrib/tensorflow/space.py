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

from odl.set import LinearSpace, RealNumbers
from odl.set.space import LinearSpaceElement
from odl.operator import Operator


__all__ = ('TensorflowSpace', 'TensorflowSpaceOperator')


class TensorflowSpace(LinearSpace):

    """A space of tensorflow Tensors."""

    def __init__(self, shape, name='ODLTensorflowSpace'):
        super(TensorflowSpace, self).__init__(RealNumbers())
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


class TensorflowSpaceElement(LinearSpaceElement):

    """Elements in TensorflowSpace."""

    def __init__(self, space, data):
        super(TensorflowSpaceElement, self).__init__(space)
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
        return '{!r}.element({!r})'.format(self.space, self.data)


class TensorflowSpaceOperator(Operator):

    """Wrap ODL operator so that it acts on TensorflowSpace elements."""

    def __init__(self, domain, range, func, adjoint=None, linear=False):
        super(TensorflowSpaceOperator, self).__init__(domain, range, linear)
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
    from odl.util.testutils import run_doctests
    run_doctests()
