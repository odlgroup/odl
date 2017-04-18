# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for internal use."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from functools import wraps
import numpy as np


__all__ = ('array1d_repr', 'array1d_str', 'arraynd_repr', 'arraynd_str',
           'dtype_repr', 'dtype_str', 'signature_string', 'indent_rows',
           'is_scalar_dtype', 'is_int_dtype', 'is_floating_dtype',
           'is_real_dtype', 'is_real_floating_dtype',
           'is_complex_floating_dtype', 'real_dtype', 'complex_dtype',
           'conj_exponent', 'as_flat_array', 'writable_array')

TYPE_MAP_R2C = {np.dtype(dtype): np.result_type(dtype, 1j)
                for dtype in np.sctypes['float']}

TYPE_MAP_C2R = {cdt: np.empty(0, dtype=cdt).real.dtype
                for rdt, cdt in TYPE_MAP_R2C.items()}
TYPE_MAP_C2R.update({k: k for k in TYPE_MAP_R2C.keys()})


def indent_rows(string, indent=4):
    """Return ``string`` indented by ``indent`` spaces."""
    return '\n'.join((' ' * indent) + row for row in string.split('\n'))


def array1d_repr(array, nprint=6):
    """Stringification of a 1D array, keeping byte / unicode.

    Parameters
    ----------
    array : `array-like`
        The array to print
    nprint : int, optional
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
    nprint : int, optional
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
    nprint : int, optional
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
    nprint : int, optional
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
    """Stringify ``dtype`` for ``repr`` with default for int and float."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype(int):
        return "'int'"
    elif dtype == np.dtype(float):
        return "'float'"
    elif dtype == np.dtype(complex):
        return "'complex'"
    else:
        return "'{}'".format(dtype)


def dtype_str(dtype):
    """Stringify ``dtype`` for ``str`` with default for int and float."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype(int):
        return 'int'
    elif dtype == np.dtype(float):
        return 'float'
    elif dtype == np.dtype(complex):
        return 'complex'
    else:
        return '{}'.format(dtype)


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


def as_flat_array(vec):
    """Return ``vec`` as a flat array according to the order of ``vec``."""
    if hasattr(vec, 'order'):
        return vec.asarray().ravel(vec.order)
    else:
        return vec.asarray().ravel()


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


def signature_string(posargs, optargs, sep=', ', mod=''):
    """Return a stringified signature from given arguments.

    Parameters
    ----------
    posargs : sequence
        Positional argument values, always included in the returned string.
        They appear in the string as (roughly)::

            sep.join(str(arg) for arg in posargs)

    optargs : sequence of 3-tuples
        Optional arguments with names and defaults, given in the form::

            [(name1, value1, default1), (name2, value2, default2), ...]

        Only those parameters that are different from the given default
        are included as ``name=value`` keyword pairs.

    sep : string or sequence of strings, optional
        Separator(s) for the argument strings. A provided single string is
        used for all joining operations.
        A given sequence must have 3 entries ``pos_sep, opt_sep, part_sep``.
        The ``pos_sep`` and ``opt_sep`` strings are used for joining the
        respective sequences of argument strings, and ``part_sep`` joins
        these two joined strings.
    mod : string or sequence, optional
        Format modifier(s) for the argument strings. A provided single
        string is used for all format strings.
        A given sequence can must have 2 entries ``pos_mod, opt_mod``
        that are either strings or sequences of strings.
        If they are strings, they are used as modifiers for the respective
        argument string sequences.
        If they are sequences of strings, their lengths must match those
        of ``posargs`` and ``optargs``, respectively, and they modify
        the format strings in a one-to-one fashion.

    Returns
    -------
    signature : string
        Stringification of a signature, typically used in the form::

            '{}({})'.format(self.__class__.__name__, signature)

    Examples
    --------
    Usage with non-trivial entries in both sequences, with a typical
    use case:

    >>> posargs = [1, 'hello', None]
    >>> optargs = [('dtype', 'float32', 'float64')]
    >>> signature_string(posargs, optargs)
    "1, 'hello', None, dtype='float32'"
    >>> '{}({})'.format('MyClass', signature_string(posargs, optargs))
    "MyClass(1, 'hello', None, dtype='float32')"

    Empty sequences and optargs values equal to default are omitted:

    >>> posargs = ['hello']
    >>> optargs = [('size', 1, 1)]
    >>> signature_string(posargs, optargs)
    "'hello'"
    >>> posargs = []
    >>> optargs = [('size', 2, 1)]
    >>> signature_string(posargs, optargs)
    'size=2'
    >>> posargs = []
    >>> optargs = [('size', 1, 1)]
    >>> signature_string(posargs, optargs)
    ''

    Using a different separator, globally or per argument "category":

    >>> posargs = [1, 'hello', None]
    >>> optargs = [('dtype', 'float32', 'float64'),
    ...            ('order', 'F', 'C')]
    >>> signature_string(posargs, optargs)
    "1, 'hello', None, dtype='float32', order='F'"
    >>> signature_string(posargs, optargs, sep=(',', ',', ', '))
    "1,'hello',None, dtype='float32',order='F'"

    Using format modifiers:

    >>> posargs = ['hello', 2.345]
    >>> optargs = [('extent', 1.442, 1.0), ('spacing', 0.0151, 1.0)]
    >>> signature_string(posargs, optargs)
    "'hello', 2.345, extent=1.442, spacing=0.0151"
    >>> # Print only two significant digits for all arguments.
    >>> # NOTE: this also affects the string!
    >>> mod = ':.2'
    >>> signature_string(posargs, optargs, mod=mod)
    'he, 2.3, extent=1.4, spacing=0.015'
    >>> mod = [['', ''], [':.3', ':.2']]  # one modifier per argument
    >>> signature_string(posargs, optargs, mod=mod)
    "'hello', 2.345, extent=1.44, spacing=0.015"
    """
    # Define the separators for the two possible cases
    try:
        sep + ''
    except TypeError:
        pos_sep, opt_sep, part_sep = sep
    else:
        pos_sep = opt_sep = part_sep = sep

    # Convert modifiers to 2-sequence of sequence of strings
    try:
        mod + ''
    except TypeError:
        pos_mod, opt_mod = mod
    else:
        pos_mod = opt_mod = mod

    mods = []
    for m, args in zip((pos_mod, opt_mod), (posargs, optargs)):
        try:
            m + ''
        except TypeError:
            if len(m) != len(args):
                raise ValueError('sequence length mismatch: '
                                 'len({}) != len({})'.format(m, args))
            mods.append(m)
        else:
            mods.append([m] * len(args))

    pos_mod, opt_mod = mods

    # Convert the arguments to strings
    parts = []

    # Stringify values, treating strings specially
    posargs_conv = []
    for arg, modifier in zip(posargs, pos_mod):
        try:
            arg + ''
        except TypeError:
            # All non-string types are passed a format conversion
            fmt = '{{{}}}'.format(modifier)
        else:
            # Preserve single quotes for strings by default
            if modifier:
                fmt = '{{{}}}'.format(modifier)
            else:
                fmt = "'{}'"

        posargs_conv.append(fmt.format(arg))

    if posargs_conv:
        parts.append(pos_sep.join(argstr for argstr in posargs_conv))

    # Build 'key=value' strings for values that are not equal to default
    optargs_conv = []
    for (name, value, default), modifier in zip(optargs, opt_mod):
        if value == default:
            # Don't include
            continue

        # See above on str and repr
        try:
            value + ''
        except TypeError:
            fmt = '{{{}}}'.format(modifier)
        else:
            if modifier:
                fmt = '{{{}}}'.format(modifier)
            else:
                fmt = "'{}'"

        value_str = fmt.format(value)

        optargs_conv.append('{}={}'.format(name, value_str))
    if optargs_conv:
        parts.append(opt_sep.join(optargs_conv))

    return part_sep.join(parts)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
