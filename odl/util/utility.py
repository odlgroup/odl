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
           'dtype_repr')


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
