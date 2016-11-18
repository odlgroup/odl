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

from functools import wraps
import numpy as np


__all__ = ('array1d_repr', 'array1d_str', 'arraynd_repr', 'arraynd_str',
           'dtype_repr', 'conj_exponent',
           'is_scalar_dtype', 'is_int_dtype', 'is_floating_dtype',
           'is_real_dtype', 'is_real_floating_dtype',
           'is_complex_floating_dtype',
           'real_dtype', 'complex_dtype',
           'writable_array')

TYPE_MAP_R2C = {np.dtype(dtype): np.result_type(dtype, 1j)
                for dtype in np.sctypes['float']}

TYPE_MAP_C2R = {cdt: np.empty(0, dtype=cdt).real.dtype
                for rdt, cdt in TYPE_MAP_R2C.items()}
TYPE_MAP_C2R.update({k: k for k in TYPE_MAP_R2C.keys()})


def _indent_rows(string, indent=4):
    out_string = '\n'.join((' ' * indent) + row for row in string.split('\n'))
    return out_string


def array1d_repr(array, nprint=6):
    """Stringification of a 1D array, keeping byte / unicode.

    Parameters
    ----------
    array : `array-like`
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
    array : `array-like`
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
    array : `array-like`
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
    """Stringification of data type with default for int and float."""
    dtype = np.dtype(dtype)
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
    """Return ``True`` if ``dtype`` is a scalar type."""
    return np.issubsctype(dtype, np.number)


def is_int_dtype(dtype):
    """Return ``True`` if ``dtype`` is an integer type."""
    return np.issubsctype(dtype, np.integer)


def is_floating_dtype(dtype):
    """Return ``True`` if ``dtype`` is a floating point type."""
    return is_real_floating_dtype(dtype) or is_complex_floating_dtype(dtype)


def is_real_dtype(dtype):
    """Return ``True`` if ``dtype`` is a real (including integer) type."""
    return is_scalar_dtype(dtype) and not is_complex_floating_dtype(dtype)


def is_real_floating_dtype(dtype):
    """Return ``True`` if ``dtype`` is a real floating point type."""
    return np.issubsctype(dtype, np.floating)


def is_complex_floating_dtype(dtype):
    """Return ``True`` if ``dtype`` is a complex floating point type."""
    return np.issubsctype(dtype, np.complexfloating)


def real_dtype(dtype, default=None):
    """Return the real counterpart of ``dtype`` if existing.

    Parameters
    ----------
    dtype :
        Real or complex floating point data type. It can be given in any
        way the `numpy.dtype` constructor understands.
    default :
        Object to be returned if no real counterpart is found for
        ``dtype``, except for ``None``, in which case an error is raised.

    Returns
    -------
    real_dtype : `numpy.dtype`
        The real counterpart of ``dtype``.

    Raises
    ------
    ValueError
        if there is no real counterpart to the given data type and
        ``default == None``.
    """
    dtype, dtype_in = np.dtype(dtype), dtype

    if is_real_floating_dtype(dtype):
        return dtype

    try:
        real_dtype = TYPE_MAP_C2R[dtype]
    except KeyError:
        if default is not None:
            return default
        else:
            raise ValueError('no real counterpart exists for `dtype` {}'
                             ''.format(dtype_repr(dtype_in)))
    else:
        return real_dtype


def complex_dtype(dtype, default=None):
    """Return complex counterpart of ``dtype`` if existing, else ``default``.

    Parameters
    ----------
    dtype :
        Real or complex floating point data type. It can be given in any
        way the `numpy.dtype` constructor understands.
    default :
        Object to be returned if no complex counterpart is found for
        ``dtype``, except for ``None``, in which case an error is raised.

    Returns
    -------
    complex_dtype : `numpy.dtype`
        The complex counterpart of ``dtype``.

    Raises
    ------
    ValueError
        if there is no complex counterpart to the given data type and
        ``default == None``.
    """
    dtype, dtype_in = np.dtype(dtype), dtype

    if is_complex_floating_dtype(dtype):
        return dtype

    try:
        complex_dtype = TYPE_MAP_R2C[dtype]
    except KeyError:
        if default is not None:
            return default
        else:
            raise ValueError('no complex counterpart exists for `dtype` {}'
                             ''.format(dtype_repr(dtype_in)))
    else:
        return complex_dtype


def conj_exponent(exp):
    """Conjugate exponent ``exp / (exp - 1)``.

    Parameters
    ----------
    exp : positive float or inf
        Exponent for which to calculate the conjugate. Must be
        at least 1.0.

    Returns
    -------
    conj : positive float or inf
        Conjugate exponent. For ``exp=1``, return ``float('inf')``,
        for ``exp=float('inf')`` return 1. In all other cases, return
        ``exp / (exp - 1)``.
    """
    if exp == 1.0:
        return float('inf')
    elif exp == float('inf'):
        return 1.0  # This is not strictly correct in math, but anyway
    else:
        return exp / (exp - 1.0)


def preload_first_arg(instance, mode):
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
    >>> f_oop_new = preload_first_arg(a, 'out-of-place')(f_oop)
    >>> f_ip_new = preload_first_arg(a, 'in-place')(f_ip)
    ...
    >>> f_oop_new(0)
    My name is A.
    >>> f_ip_new(0, out=1)
    My name is A.

    Decorate upon definition:

    >>> @preload_first_arg(a, 'out-of-place')
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
            raise ValueError('bad mode {!r}'.format(mode))

    return decorator


class writable_array(object):
    """Context manager that casts obj to a `numpy.array` and saves changes."""

    def __init__(self, obj, *args, **kwargs):
        """initialize a new instance.

        Parameters
        ----------
        obj : `array-like`
            Object that should be cast to an array, must be usable with
            `numpy.asarray` and be set-able with ``obj[:] = arr``.
        args, kwargs :
            Arguments that should be passed to `numpy.asarray`.

        Examples
        --------
        Convert list to array and use with numpy

        >>> lst = [1, 2, 3]
        >>> with writable_array(lst) as arr:
        ...    arr *= 2
        >>> lst
        [2, 4, 6]

        Also usable with ODL vectors

        >>> space = odl.uniform_discr(0, 1, 3)
        >>> x = space.element([1, 2, 3])
        >>> with writable_array(x) as arr:
        ...    arr += [1, 1, 1]
        >>> x
        uniform_discr(0.0, 1.0, 3).element([2.0, 3.0, 4.0])

        Can also be called with arguments to `numpy.asarray`

        >>> lst = [1, 2, 3]
        >>> with writable_array(lst, dtype='complex') as arr:
        ...    arr  # print array
        array([ 1.+0.j,  2.+0.j,  3.+0.j])

        Note that the changes are only saved once the context manger exits,
        before, the input vector is in general unchanged

        >>> lst = [1, 2, 3]
        >>> with writable_array(lst) as arr:
        ...    arr *= 2
        ...    lst  # print content of lst before exiting
        [1, 2, 3]
        >>> lst  # print content of lst after exit
        [2, 4, 6]
        """
        self.obj = obj
        self.args = args
        self.kwargs = kwargs
        self.arr = None

    def __enter__(self):
        """called by ``with writable_array(obj):``.

        Returns
        -------
        arr : `numpy.ndarray`
            Array representing ``self.obj``, created by calling
            ``numpy.asarray``. Any changes to ``arr`` will be passed through
            to ``self.obj`` after the context manager exits.
        """
        self.arr = np.asarray(self.obj, *self.args, **self.kwargs)
        return self.arr

    def __exit__(self, type, value, traceback):
        """called when ``with writable_array(obj):`` ends.

        Saves any changes to ``self.arr`` to ``self.obj``, also "frees"
        self.arr in case the manager is used multiple times.

        Extra arguments are ignored, any exceptions are passed through.
        """
        self.obj[:] = self.arr
        self.arr = None


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
