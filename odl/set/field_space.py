# Copyright 2014-2016 The ODL development group
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

"""Fields as spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import numbers

from odl.set.sets import Field, ComplexNumbersSet, RealNumbersSet
from odl.set.space import LinearSpace, LinearSpaceElement, LinearSpaceTypeError


__all__ = ('FieldSpace', 'FieldSpaceElement', 'RealNumbers', 'ComplexNumbers')


class FieldSpaceElement(LinearSpaceElement, numbers.Number):
    def __init__(self, space, value):
        self.value = value
        LinearSpaceElement.__init__(self, space)

    @property
    def imag(self):
        return self.value.imag

    @property
    def real(self):
        return self.value.real

    def conjugate(self):
        return self.value.conjugate()

    def __float__(self):
        return float(self.value)

    def __complex__(self):
        return complex(self.value)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)

    def __array__(self, dtype=None):
        return np.array(self.value, dtype=dtype)

    def __array_wrap__(self, arr):
        return self.space.dtype.type(arr)


class FieldSpace(Field, LinearSpace):
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)

    def __eq__(self, other):
        return isinstance(other, FieldSpace) and other.dtype == self.dtype

    def element(self, inp=None):
        if inp is None:
            inp = 0

        try:
            val = self.dtype.type(inp)
        except TypeError:
            raise LinearSpaceTypeError()

        if val.size != 1:
            raise LinearSpaceTypeError()

        return self.element_type(self, val)

    def _lincomb(self, a, x1, b, x2, out):
        out.value = a * x1 + b * x2

    def _inner(self, x, y):
        return x * y

    element_type = FieldSpaceElement


# -- add arithmetic operators

def unary_op_factory(name):
    def op_impl(self):
        return getattr(self.value, name)()
    return op_impl


def binary_op_factory(name):
    def op_impl(self, other):
        other_val = self.space.dtype.type(other)
        return getattr(self.value, name)(other_val)
    return op_impl


def inplace_binary_op_factory(name):
    def op_impl(self, other):
        other_val = self.space.dtype.type(other)
        result = getattr(self.value, name)(other_val)
        self.value = self.space.dtype.type(result)
        return self
    return op_impl


def comparsion_op_factory(name):
    def op_impl(self, other):
        return getattr(self.value, name)(other)
    return op_impl


arithmetic_op_names = ['add', 'sub', 'div', 'truediv', 'mul', 'pow']
modifiers = ['', 'r']
for op_name in arithmetic_op_names:
    for modifier in modifiers:
        name = '__' + modifier + op_name + '__'
        setattr(FieldSpaceElement, name, binary_op_factory(name))

    # in place
    name = '__i' + op_name + '__'
    impl_name = '__' + op_name + '__'
    setattr(FieldSpaceElement, name, inplace_binary_op_factory(impl_name))


comparsion_op_names = ['le', 'lt', 'eq', 'ne', 'mod', 'rmod', 'floordiv',
                       'rfloordiv']
for op_name in comparsion_op_names:
    name = '__' + op_name + '__'
    setattr(FieldSpaceElement, name, comparsion_op_factory(name))


unary_op_names = ['neg', 'pos', 'abs', 'trunc', 'ceil', 'floor', 'round']
for op_name in unary_op_names:
    name = '__' + op_name + '__'
    setattr(FieldSpaceElement, name, unary_op_factory(name))


# -- complex numbers


class ComplexNumbersElement(FieldSpaceElement, numbers.Complex):
    pass


class ComplexNumbers(FieldSpace, ComplexNumbersSet):
    def __init__(self):
        FieldSpace.__init__(self, 'complex128')

    element_type = ComplexNumbersElement


# -- real numbers


class RealNumbersElement(FieldSpaceElement, numbers.Real):
    pass


class RealNumbers(FieldSpace, RealNumbersSet):
    def __init__(self):
        FieldSpace.__init__(self, 'float64')

    element_type = RealNumbersElement


# Examples to show it works as expected
if __name__ == '__main__':
    field = RealNumbers()
    a = field.element(3.0)
    print(a + a)
    print(np.abs(a))

    field = ComplexNumbers()
    a = field.element(3.0)
    print(a + a)
    print(np.abs(a))

    assert not np.isinf(a)

    # In place operations are actually in place
    ida = id(a)
    a += 3
    assert ida == id(a)
    assert a == 6

    # Any non-inplace operation simply

    import odl
    rn = odl.rn(3)
    x = rn.element(np.array([1.0, 2.0, 3.0]))
