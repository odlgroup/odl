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
           'dtype_repr', 'snr')


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


def conj_exponent(exp):
    """The conjugate exponent p / (p-1).

    Parameters
    ----------
    exp : positive `float` or inf
        Exponent for which to calculate the conjugate. Must be
        at least 1.0.

    Returns
    -------
    conj : positive `float` or inf
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


def normalized_index_expression(indices, shape, int_to_slice=False):
    """Enable indexing with almost Numpy-like capabilities.

    Implements the following features:

    - Usage of general slices and sequences of slices
    - Conversion of `Ellipsis` into an adequate number of ``slice(None)``
      objects
    - Fewer indices than axes by filling up with an `Ellipsis`
    - Error checking with respect to a given shape
    - Conversion of integer indices into corresponding slices

    Parameters
    ----------
    indices : `int`, `slice` or `sequence` of those
        Index expression to be normalized
    shape : `sequence` of `int`
        Target shape for error checking of out-of-bounds indices.
        Also needed to determine the number of axes.
    int_to_slice : `bool`, optional
        If `True`, turn integers into corresponding slice objects.

    Returns
    -------
    normalized : `tuple` of `slice`
        Normalized index expression
    """
    ndim = len(shape)
    # Support indexing with fewer indices as indexing along the first
    # corresponding axes. In the other cases, normalize the input.
    if np.isscalar(indices):
        indices = [indices, Ellipsis]
    elif (isinstance(indices, slice) or indices is Ellipsis):
        indices = [indices]

    indices = list(indices)
    if len(indices) < ndim and Ellipsis not in indices:
        indices.append(Ellipsis)

    # Turn Ellipsis into the correct number of slice(None)
    if Ellipsis in indices:
        if indices.count(Ellipsis) > 1:
            raise ValueError('cannot use more than one `Ellipsis`')

        eidx = indices.index(Ellipsis)
        extra_dims = ndim - len(indices) + 1
        indices = (indices[:eidx] + [slice(None)] * extra_dims +
                   indices[eidx + 1:])

    # Turn single indices into length-1 slices if desired
    for (i, idx), n in zip(enumerate(indices), shape):
        if np.isscalar(idx):
            if idx < 0:
                idx += n

            if idx >= n:
                raise IndexError('index {} is out of bounds for axis '
                                 '{} with size {}'
                                 ''.format(idx, i, n))
            if int_to_slice:
                indices[i] = slice(idx, idx + 1)

    # Catch most common errors
    if any(s.start == s.stop and s.start is not None or
           s.start == n
           for s, n in zip(indices, shape) if isinstance(s, slice)):
        raise ValueError('slices with empty axes not allowed')
    if None in indices:
        raise ValueError('creating new axes is not supported')
    if len(indices) > ndim:
        raise IndexError('too may indices: {} > {}'
                         ''.format(len(indices), ndim))

    return tuple(indices)


def snr(signal, noise, impl):
    """Compute the signal-to-noise ratio.

    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = s_power / n_power,
        'dB' means SNR = 10 * log10 (s_power / n_power).

    Returns
    -------
    snr : `float`
    Value of signal-to-noise ratio.
    If the power of noise is zero, then the return is 'inf',
    otherwise, the computed value.
    """
    if np.any(noise):
        s_power = np.var(signal)
        n_power = np.var(noise)
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
