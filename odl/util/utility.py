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
    out_string = '\n'.join((' ' * indent) + row for row in string.split('\n'))
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
        return (repr(list(array[:nprint // 2])).rstrip(']') + ', ..., ' +
                repr(list(array[-(nprint // 2):])).lstrip('['))


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
        left_str = ', '.join(str(a) for a in array[:nprint // 2])
        right_str = ', '.join(str(a) for a in array[-(nprint // 2):])
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
                                   array[:nprint // 2])
            right_str = ',\n '.join(arraynd_repr(a) for a in
                                    array[-(nprint // 2):])
            return '[{},\n ...,\n {}]'.format(left_str, right_str)
    else:
        return array1d_repr(array)


def arraynd_str(array, nprint=None):
    """Stringification of an nD array, regardless of byte or unicode.

    Parameters
    ----------
    array : `array-like`
        The array to print
    nprint : int
        Maximum number of elements to print.
        Default: 6 if array.ndim <= 2, else 2

    Examples
    --------
    >>> print(arraynd_str([[1, 2, 3], [4, 5, 6]]))
    [[1, 2, 3],
     [4, 5, 6]]
    >>> print(arraynd_str([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
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
            inner_str = ',\n '.join(arraynd_str(a) for a in array)
            return '[{}]'.format(inner_str)
        else:
            left_str = ',\n'.join(arraynd_str(a) for a in
                                  array[:nprint // 2])
            right_str = ',\n'.join(arraynd_str(a) for a in
                                   array[- (nprint // 2):])
            return '[{},\n    ...,\n{}]'.format(left_str, right_str)
    else:
        return array1d_str(array)


def dtype_repr(dtype):
    """Stringification of data type with default for `int` and `float`."""
    if dtype == np.dtype(int):
        return "'int'"
    elif dtype == np.dtype(float):
        return "'float'"
    elif dtype == np.dtype(complex):
        return "'complex'"
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


def complex_space(space):
    """Return the complex counterpart of a given space.

    Parameters
    ----------
    space : `LinearSpace`
        Template space to be converted into its complex counterpart.
        If ``space`` is already a complex space, a copy of it is
        returned. Supported types of spaces: `FunctionSpace`, `FnBase`,
        `DiscreteLp`.

    Returns
    -------
    cspace : `LinearSpace`
        Space of the same type and all parameters equal except field
        and data type (if applicable), which are mapped to their complex
        counterparts.

    Examples
    --------
    >>> from odl import uniform_discr
    >>> rspace = uniform_discr(0, 1, 10, dtype='float32')
    >>> cspace = complex_space(rspace)
    >>> cspace
    uniform_discr(0.0, 1.0, 10, dtype='complex64')
    >>> rspace.one().norm() == cspace.one().norm()
    True
    >>> complex_space(cspace) == cspace
    True
    """
    from odl.discr.lp_discr import DiscreteLp
    from odl.set.sets import ComplexNumbers
    from odl.space.base_ntuples import FnBase, _TYPE_MAP_R2C
    from odl.space.fspace import FunctionSpace

    if isinstance(space, FunctionSpace):
        return FunctionSpace(space.domain, field=ComplexNumbers())
    else:
        if is_real_floating_dtype(space.dtype):
            complex_dtype = _TYPE_MAP_R2C[space.dtype]
        elif is_complex_floating_dtype(space.dtype):
            complex_dtype = space.dtype
        else:
            raise ValueError('data type {} is not a floating point type.'
                             ''.format(space.dtype))

    # DiscreteLp needs to come first since it inherits from FnBase
    if isinstance(space, DiscreteLp):
        return type(space)(complex_space(space.uspace), space.partition,
                           dspace=complex_space(space.dspace),
                           exponent=space.exponent, interp=space.interp,
                           order=space.order)
    elif isinstance(space, FnBase):
        return type(space)(space.size, dtype=complex_dtype,
                           weight=space._space_funcs)
    else:
        raise TypeError('space type {} not supported.'.format(type(space)))


def real_space(space):
    """Return the real counterpart of a given space.

    Parameters
    ----------
    space : `LinearSpace`
        Template space to be converted into its real counterpart.
        If ``space`` is already a real space, a copy of it is returned.
        Supported types of spaces: `FunctionSpace`, `FnBase`,
        `DiscreteLp`.

    Returns
    -------
    rspace : `LinearSpace`
        Space of the same type and all parameters equal except field
        and data type (if applicable), which are mapped to their real
        counterparts.

    Examples
    --------
    >>> from odl import uniform_discr
    >>> cspace = uniform_discr(0, 1, 10, dtype='complex64')
    >>> rspace = real_space(cspace)
    >>> rspace
    uniform_discr(0.0, 1.0, 10, dtype='float32')
    >>> cspace.one().norm() == rspace.one().norm()
    True
    >>> real_space(rspace) == rspace
    True
    """
    from odl.discr.lp_discr import DiscreteLp
    from odl.set.sets import RealNumbers
    from odl.space.base_ntuples import FnBase, _TYPE_MAP_C2R
    from odl.space.fspace import FunctionSpace

    if isinstance(space, FunctionSpace):
        return FunctionSpace(space.domain, field=RealNumbers())
    else:
        if is_real_floating_dtype(space.dtype):
            real_dtype = space.dtype
        elif is_complex_floating_dtype(space.dtype):
            real_dtype = _TYPE_MAP_C2R[space.dtype]
        else:
            raise ValueError('data type {} is not a floating point type.'
                             ''.format(space.dtype))

    # DiscreteLp needs to come first since it inherits from FnBase
    if isinstance(space, DiscreteLp):
        return type(space)(real_space(space.uspace), space.partition,
                           dspace=real_space(space.dspace),
                           exponent=space.exponent, interp=space.interp,
                           order=space.order)
    elif isinstance(space, FnBase):
        return type(space)(space.size, dtype=real_dtype,
                           weight=space._space_funcs)
    else:
        raise TypeError('space type {} not supported.'.format(type(space)))


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


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
