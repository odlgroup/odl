# Copyright 2014, 2015 The ODL development group
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

"""Utilities for internal use."""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

# External module imports
import numpy as np

__all__ = ('array1d_repr', 'array1d_str', 'arraynd_repr', 'arraynd_str',
           'dtype_repr', 'UFUNCS')


def array1d_repr(array):
    """Stringification of a 1D array, keeping byte / unicode."""
    if len(array) < 7:
        return repr(list(array))
    else:
        return (repr(list(array[:3])).rstrip(']') + ', ..., ' +
                repr(list(array[-3:])).lstrip('['))


def array1d_str(array):
    """Stringification of a 1D array, regardless of byte or unicode."""
    if len(array) < 7:
        inner_str = ', '.join(str(a) for a in array)
        return '[{}]'.format(inner_str)
    else:
        left_str = ', '.join(str(a) for a in array[:3])
        right_str = ', '.join(str(a) for a in array[-3:])
        return '[{}, ..., {}]'.format(left_str, right_str)


def arraynd_repr(array):
    """Stringification of an nD array, keeping byte / unicode."""
    if array.ndim > 1:
        if len(array) < 7:
            inner_str = ',\n '.join(arraynd_repr(a) for a in array)
            return '[\n{}\n]'.format(inner_str)
        else:
            left_str = ',\n '.join(arraynd_repr(a) for a in array[:3])
            right_str = ',\n '.join(arraynd_repr(a) for a in array[-3:])
            return '[\n{},\n ...,\n{}\n]'.format(left_str, right_str)
    else:
        return array1d_repr(array)


def arraynd_str(array):
    """Stringification of a nD array, regardless of byte or unicode."""
    if array.ndim > 1:
        if len(array) < 7:
            inner_str = ',\n '.join(arraynd_str(a) for a in array)
            return '[{}]'.format(inner_str)
        else:
            left_str = ',\n '.join(arraynd_str(a) for a in array[:3])
            right_str = ',\n '.join(arraynd_str(a) for a in array[-3:])
            return '[{},\n ...,\n{}]'.format(left_str, right_str)
    else:
        return array1d_str(array)


def dtype_repr(dtype):
    """Stringification of data type with default for `int` and `float`."""
    if dtype == np.dtype(int):
        return "'int'"
    elif dtype == np.dtype(float):
        return "'float'"
    else:
        return "'{}'".format(dtype)

if __name__ == '__main__':
    import doctest
    doctest.testmod()


def is_scalar_dtype(dtype):
    """Whether a datatype is scalar or not."""
    return np.issubsctype(dtype, np.number)


def is_real_dtype(dtype):
    """Whether a datatype is real (including integer) or not."""
    return is_scalar_dtype(dtype) and not is_complex_floating_dtype(dtype)


def is_real_floating_dtype(dtype):
    """Whether a NumPy datatype is real complex-floating or not."""
    return np.issubsctype(dtype, np.floating)


def is_complex_floating_dtype(dtype):
    """Whether a NumPy datatype is complex floating-point or not."""
    return np.issubsctype(dtype, np.complexfloating)


def with_metaclass(meta, *bases):
    """
    Function from jinja2/_compat.py. License: BSD.

    Use it like this::

        class BaseForm(object):
            pass

        class FormType(type):
            pass

        class Form(with_metaclass(FormType, BaseForm)):
            pass

    This requires a bit of explanation: the basic idea is to make a
    dummy metaclass for one level of class instantiation that replaces
    itself with the actual metaclass.  Because of internal type checks
    we also need to make sure that we downgrade the custom metaclass
    for one level to something closer to type (that's why __call__ and
    __init__ comes back from type etc.).

    This has the advantage over six.with_metaclass of not introducing
    dummy classes into the final MRO.
    """
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)
    return metaclass('temporary_class', None, {})

#Ignore arithmetic since we already support that
# Name, args, optional args, docstring
UFUNCS = [#('add', 2, 1, 'Add arguments element-wise.'),
          #('subtract', 2, 1, 'Subtract arguments, element-wise.'),
          #('multiply', 2, 1, 'Multiply arguments element-wise.'),
          #('divide', 2, 1, 'Divide arguments element-wise.'),
          ('logaddexp', 2, 1, 'Logarithm of the sum of exponentiations of the inputs.'),
          ('logaddexp2', 2, 1, 'Logarithm of the sum of exponentiations of the inputs in base-2.'),
          ('true_divide', 2, 1, 'Returns a true division of the inputs, element-wise.'),
          ('floor_divide', 2, 1, 'Return the largest integer smaller or equal to the division of the inputs.'),
          ('negative', 1, 1, 'Numerical negative, element-wise.'),
          ('power', 2, 1, 'First array elements raised to powers from second array, element-wise.'),
          ('remainder', 2, 1, 'Return element-wise remainder of division.'),
          ('mod', 2, 1, 'Return element-wise remainder of division.'),
          ('fmod', 2, 1, 'Return the element-wise remainder of division.'),
          ('absolute', 1, 1, 'Calculate the absolute value element-wise.'),
          ('rint', 1, 1, 'Round elements of the array to the nearest integer.'),
          ('sign', 1, 1, 'Returns an element-wise indication of the sign of a number.'),
          ('conj', 1, 1, 'Return the complex conjugate, element-wise.'),
          ('exp', 1, 1, 'Calculate the exponential of all elements in the input array.'),
          ('exp2', 1, 1, 'Calculate 2**p for all p in the input array.'),
          ('log', 1, 1, 'Natural logarithm, element-wise.'),
          ('log2', 1, 1, 'Base-2 logarithm of x.'),
          ('log10', 1, 1, 'Return the base 10 logarithm of the input array, element-wise.'),
          ('expm1', 1, 1, 'Calculate exp(x) - 1 for all elements in the array.'),
          ('log1p', 1, 1, 'Return the natural logarithm of one plus the input array, element-wise.'),
          ('sqrt', 1, 1, 'Return the positive square-root of an array, element-wise.'),
          ('square', 1, 1, 'Return the element-wise square of the input.'),
          ('reciprocal', 1, 1, 'Return the reciprocal of the argument, element-wise.'),
          ('sin', 1, 1, 'Trigonometric sine, element-wise.'),
          ('cos', 1, 1, 'Cosine element-wise.'),
          ('tan', 1, 1, 'Compute tangent element-wise.'),
          ('arcsin', 1, 1, 'Inverse sine, element-wise.'),
          ('arccos', 1, 1, 'Trigonometric inverse cosine, element-wise.'),
          ('arctan', 1, 1, 'Trigonometric inverse tangent, element-wise.'),
          ('arctan2', 2, 1, 'Element-wise arc tangent of x1/x2 choosing the quadrant correctly.'),
          ('hypot', 2, 1, 'Given the "legs" of a right triangle, return its hypotenuse.'),
          ('sinh', 1, 1, 'Hyperbolic sine, element-wise.'),
          ('cosh', 1, 1, 'Hyperbolic cosine, element-wise.'),
          ('tanh', 1, 1, 'Compute hyperbolic tangent element-wise.'),
          ('arcsinh', 1, 1, 'Inverse hyperbolic sine element-wise.'),
          ('arccosh', 1, 1, 'Inverse hyperbolic cosine, element-wise.'),
          ('arctanh', 1, 1, 'Inverse hyperbolic tangent element-wise.'),
          ('deg2rad', 1, 1, 'Convert angles from degrees to radians.'),
          ('rad2deg', 1, 1, 'Convert angles from radians to degrees.'),
          ('bitwise_and', 2, 1, 'Compute the bit-wise AND of two arrays element-wise.'),
          ('bitwise_or', 2, 1, 'Compute the bit-wise OR of two arrays element-wise.'),
          ('bitwise_xor', 2, 1, 'Compute the bit-wise XOR of two arrays element-wise.'),
          ('invert', 1, 1, 'Compute bit-wise inversion, or bit-wise NOT, element-wise.'),
          ('left_shift', 2, 1, 'Shift the bits of an integer to the left.'),
          ('right_shift', 2, 1, 'Shift the bits of an integer to the right.'),
          ('greater', 2, 1, 'Return the truth value of (x1 > x2) element-wise.'),
          ('greater_equal', 2, 1, 'Return the truth value of (x1 >= x2) element-wise.'),
          ('less', 2, 1, 'Return the truth value of (x1 < x2) element-wise.'),
          ('less_equal', 2, 1, 'Return the truth value of (x1 =< x2) element-wise.'),
          ('not_equal', 2, 1, 'Return (x1 != x2) element-wise.'),
          ('equal', 2, 1, 'Return (x1 == x2) element-wise.'),
          ('logical_and', 2, 1, 'Compute the truth value of x1 AND x2 element-wise.'),
          ('logical_or', 2, 1, 'Compute the truth value of x1 OR x2 element-wise.'),
          ('logical_xor', 2, 1, 'Compute the truth value of x1 XOR x2, element-wise.'),
          ('logical_not', 1, 1, 'Compute the truth value of NOT x element-wise.'),
          ('maximum', 2, 1, 'Element-wise maximum of array elements.'),
          ('minimum', 2, 1, 'Element-wise minimum of array elements.'),
          ('fmax', 2, 1, 'Element-wise maximum of array elements.'),
          ('fmin', 2, 1, 'Element-wise minimum of array elements.'),
          ('isreal', 1, 0, 'Returns a bool array, where True if input element is real.'),
          ('iscomplex', 1, 0, 'Returns a bool array, where True if input element is complex.'),
          ('isfinite', 1, 1, 'Test element-wise for finiteness (not infinity or not Not a Number).'),
          ('isinf', 1, 1, 'Test element-wise for positive or negative infinity.'),
          ('isnan', 1, 1, 'Test element-wise for NaN and return result as a boolean array.'),
          ('signbit', 1, 1, 'Returns element-wise True where signbit is set (less than zero).'),
          ('copysign', 2, 1, 'Change the sign of x1 to that of x2, element-wise.'),
          ('nextafter', 2, 1, 'Return the next floating-point value after x1 towards x2, element-wise.'),
          ('modf', 1, 2, 'Return the fractional and integral parts of an array, element-wise.'),
          ('ldexp', 2, 1, 'Returns x1 * 2**x2, element-wise.'),
          ('frexp', 1, 2, 'Decompose the elements of x into mantissa and twos exponent.'),
          ('fmod', 2, 1, 'Return the element-wise remainder of division.'),
          ('floor', 1, 1, 'Return the floor of the input, element-wise.'),
          ('ceil', 1, 1, 'Return the ceiling of the input, element-wise.'),
          ('trunc', 1, 1, 'Return the truncated value of the input, element-wise.')]
