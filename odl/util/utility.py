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
from functools import wraps
import numpy as np


__all__ = ('array1d_repr', 'array1d_str', 'arraynd_repr', 'arraynd_str',
           'dtype_repr')

def _indent_rows(string, indent=4):
    out_string = '\n'.join((' '*indent) + row for row in string.split('\n'))
    return out_string

def array1d_repr(array, nprint=6):
    """Stringification of a 1D array, keeping byte / unicode.

    Parameters
    ----------
    array : array-like
        The array to print
    nprint : int
        Maximum number of elements to print
    """
    assert int(nprint) > 0

    if len(array) <= nprint:
        return repr(list(array))
    else:
        return (repr(list(array[:nprint//2])).rstrip(']') + ', ..., ' +
                repr(list(array[-nprint//2:])).lstrip('['))


def array1d_str(array, nprint=6):
    """Stringification of a 1D array, regardless of byte or unicode.

    Parameters
    ----------
    array : array-like
        The array to print
    nprint : int
        Maximum number of elements to print
    """
    assert int(nprint) > 0

    if len(array) <= nprint:
        inner_str = ', '.join(str(a) for a in array)
        return '[{}]'.format(inner_str)
    else:
        left_str = ', '.join(str(a) for a in array[:nprint//2])
        right_str = ', '.join(str(a) for a in array[-nprint//2:])
        return '[{}, ..., {}]'.format(left_str, right_str)


def arraynd_repr(array, nprint=None):
    """Stringification of an nD array, keeping byte / unicode.

    Parameters
    ----------
    array : array-like
        The array to print
    nprint : int
        Maximum number of elements to print.
        Default: 6 if array.ndim <= 2, else 2

    Examples
    --------
    >>> print(arraynd_repr([[1, 2, 3], [4, 5, 6]]))
    [[1, 2, 3],
     [4, 5, 6]]
    >>> print(arraynd_repr([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
    """
    array = np.asarray(array)
    if nprint is None:
        nprint = 6 if array.ndim <= 2 else 2
    else:
        assert nprint > 0

    if array.ndim > 1:
        if len(array) <= nprint:
            inner_str = ',\n '.join(arraynd_repr(a) for a in array)
            return '[{}]'.format(inner_str)
        else:
            left_str = ',\n '.join(arraynd_repr(a) for a in
                                   array[:nprint//2])
            right_str = ',\n '.join(arraynd_repr(a) for a in
                                    array[-nprint//2:])
            return '[{},\n ...,\n {}]'.format(left_str, right_str)
    else:
        return array1d_repr(array)


def arraynd_str(array, nprint=None):
    """Stringification of a nD array, regardless of byte or unicode.

    Parameters
    ----------
    array : array-like
        The array to print
    nprint : int
        Maximum number of elements to print.
        Default: 6 if array.ndim <= 2, else 2

    Examples
    --------
    >>> print(arraynd_str([[1, 2, 3], [4, 5, 6]]))
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
    >>> print(arraynd_str([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    """
    array = np.asarray(array)
    if nprint is None:
        nprint = 6 if array.ndim <= 2 else 2
    else:
        assert nprint > 0

    if array.ndim > 1:
        if len(array) <= nprint:
            inner_str = ',\n'.join(arraynd_str(a) for a in array)
            return '[\n{}\n]'.format(_indent_rows(inner_str))
        else:
            left_str = ',\n'.join(arraynd_str(a) for a in array[:nprint//2])
            right_str = ',\n'.join(arraynd_str(a) for a in array[-nprint//2:])
            return '[\n{},\n    ...,\n{}\n]'.format(_indent_rows(left_str),
                                                    _indent_rows(right_str))
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


def is_scalar_dtype(dtype):
    """`True` if ``dtype`` is scalar, else `False`."""
    return np.issubsctype(dtype, np.number)


def is_int_dtype(dtype):
    """`True` if ``dtype`` is integer, else `False`."""
    return np.issubsctype(dtype, np.integer)


def is_floating_dtype(dtype):
    """`True` if ``dtype`` is floating-point, else `False`."""
    return is_real_floating_dtype(dtype) or is_complex_floating_dtype(dtype)


def is_real_dtype(dtype):
    """`True` if ``dtype`` is real (including integer), else `False`."""
    return is_scalar_dtype(dtype) and not is_complex_floating_dtype(dtype)


def is_real_floating_dtype(dtype):
    """`True` if ``dtype`` is real floating-point, else `False`."""
    return np.issubsctype(dtype, np.floating)


def is_complex_floating_dtype(dtype):
    """`True` if ``dtype`` is complex floating-point, else `False`."""
    return np.issubsctype(dtype, np.complexfloating)


def preload_call_with(instance, mode):
    """Decorator to preload the first argument of a call method.

    Parameters
    ----------
    instance :
        Class instance to preload the call with
    mode : {'out-of-place', 'in-place'}

        'out-of-place': call is out-of-place -- ``f(x, **kwargs)``

        'in-place': call is in-place -- ``f(x, out, **kwargs)``

    Notes
    -----
    The decorated function has the signature according to ``mode``.

    Examples
    --------
    Define two functions which need some instance to act on and decorate
    them manually:

    >>> class A(object):
    ...     '''My name is A.'''
    >>> a = A()
    ...
    >>> def f_oop(inst, x):
    ...     print(inst.__doc__)
    ...
    >>> def f_ip(inst, out, x):
    ...     print(inst.__doc__)
    ...
    >>> f_oop_new = preload_call_with(a, 'out-of-place')(f_oop)
    >>> f_ip_new = preload_call_with(a, 'in-place')(f_ip)
    ...
    >>> f_oop_new(0)
    My name is A.
    >>> f_ip_new(0, out=1)
    My name is A.

    Decorate upon definition:

    >>> @preload_call_with(a, 'out-of-place')
    ... def set_x(obj, x):
    ...     '''Function to set x in ``obj`` to a given value.'''
    ...     obj.x = x
    >>> set_x(0)
    >>> a.x
    0

    The function's name and docstring are preserved:

    >>> set_x.__name__
    'set_x'
    >>> set_x.__doc__
    'Function to set x in ``obj`` to a given value.'
    """

    def decorator(call):

        @wraps(call)
        def oop_wrapper(x, **kwargs):
            return call(instance, x, **kwargs)

        @wraps(call)
        def ip_wrapper(x, out, **kwargs):
            return call(instance, x, out, **kwargs)

        if mode == 'out-of-place':
            return oop_wrapper
        elif mode == 'in-place':
            return ip_wrapper
        else:
            raise ValueError('bad mode {!r}.'.format(mode))

    return decorator


def preload_default_oop_call_with(vector):
    """Decorator to bind the default out-of-place call to an instance.

    Parameters
    ----------
    vector : `FunctionSetVector`
        Vector with which the default call is preloaded. Its
        `FunctionSetVector.space` determines the type of
        implementation chosen for the vectorized evaluation. If
        ``vector.space`` has a `LinearSpace.field` attribute, the
        required output data type is inferred from it. Otherwise,
        a "lazy" vectorization is performed (not implemented).

    Notes
    -----
    Usually this decorator is used as as a function factory::

        preload_default_oop_call_with(<vec>)(<call>)
    """

    def decorator(call):

        from odl.set.sets import RealNumbers, ComplexNumbers

        field = getattr(vector.space, 'field', None)
        if field is None:
            dtype = None
        elif field == RealNumbers():
            dtype = 'float64'
        elif field == ComplexNumbers():
            dtype = 'complex128'
        else:
            raise TypeError('cannot handle field {!r}.'.format(field))

        @wraps(call)
        def oop_wrapper(x, **kwargs):
            return call(vector, dtype, x, **kwargs)

        return oop_wrapper

    return decorator


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
