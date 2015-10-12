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
from builtins import str

# External module imports
import numpy as np

__all__ = ('array1d_repr', 'array1d_str', 'arraynd_repr', 'arraynd_str',
           'dtype_repr')


def array1d_repr(array):
    """Stringification of a 1D array, keeping byte / unicode."""
    if len(array) < 7:
        return repr(list(array.asarray()))
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


def is_real_dtype(dtype):
    """Whether a datatype is real or not."""
    return np.isrealobj(np.empty(0, dtype=dtype))


def is_complex_dtype(dtype):
    """Whether a datatype is complex or not."""
    return np.iscomplexobj(np.empty(0, dtype=dtype))
