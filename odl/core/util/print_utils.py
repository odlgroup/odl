# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities mainly for internal use."""

# Python imports
from itertools import zip_longest
from contextlib import contextmanager

# Third-party import
import numpy as np

# ODL import
from odl.core.array_API_support.array_creation import asarray
from odl.core.array_API_support.utils import get_array_and_backend
from odl.core.util.dtype_utils import _universal_dtype_identifier

__all__ = (
    "is_string",
    "dtype_repr",
    "dtype_str",
    "REPR_PRECISION",
    "indent",
    "dedent",
    "npy_printoptions",
    "array_str",
    "signature_string",
    "signature_string_parts",
    "repr_string",
    "attribute_repr_string",
    "method_repr_string",
)


def is_string(obj):
    """Return ``True`` if ``obj`` behaves like a string, ``False`` else."""
    return isinstance(obj, str)


def dtype_repr(dtype):
    """gets the dtype representation of provided dtype"""
    return f"'{dtype_str(dtype)}'"


def dtype_str(dtype):
    """gets the universal dtype identifier of the provided dtype"""
    return f"{_universal_dtype_identifier(dtype)}"


REPR_PRECISION = 4


def indent(string, indent_str="    "):
    """Return a copy of ``string`` indented by ``indent_str``.

    Parameters
    ----------
    string : str
        Text that should be indented.
    indent_str : str, optional
        String to be inserted before each new line. The default is to
        indent by 4 spaces.

    Returns
    -------
    indented : str
        The indented text.

    Examples
    --------
    >>> text = '''This is line 1.
    ... Next line.
    ... And another one.'''
    >>> print(text)
    This is line 1.
    Next line.
    And another one.
    >>> print(indent(text))
        This is line 1.
        Next line.
        And another one.

    Indenting by random stuff:

    >>> print(indent(text, indent_str='<->'))
    <->This is line 1.
    <->Next line.
    <->And another one.
    """
    return "\n".join(indent_str + row for row in string.splitlines())


def dedent(string, indent_str="   ", max_levels=None):
    """Revert the effect of indentation.

    Examples
    --------
    Remove a simple one-level indentation:

    >>> text = '''<->This is line 1.
    ... <->Next line.
    ... <->And another one.'''
    >>> print(text)
    <->This is line 1.
    <->Next line.
    <->And another one.
    >>> print(dedent(text, '<->'))
    This is line 1.
    Next line.
    And another one.

    Multiple levels of indentation:

    >>> text = '''<->Level 1.
    ... <-><->Level 2.
    ... <-><-><->Level 3.'''
    >>> print(text)
    <->Level 1.
    <-><->Level 2.
    <-><-><->Level 3.
    >>> print(dedent(text, '<->'))
    Level 1.
    <->Level 2.
    <-><->Level 3.

    >>> text = '''<-><->Level 2.
    ... <-><-><->Level 3.'''
    >>> print(text)
    <-><->Level 2.
    <-><-><->Level 3.
    >>> print(dedent(text, '<->'))
    Level 2.
    <->Level 3.
    >>> print(dedent(text, '<->', max_levels=1))
    <->Level 2.
    <-><->Level 3.
    """
    if len(indent_str) == 0:
        return string

    lines = string.splitlines()

    # Determine common (minimum) number of indentation levels, capped at
    # `max_levels` if given
    def num_indents(line):
        max_num = int(np.ceil(len(line) / len(indent_str)))

        i = 0  # set for the case the loop is not run (`max_num == 0`)
        for i in range(max_num):
            if line.startswith(indent_str):
                line = line[len(indent_str) :]
            else:
                break

        return i

    num_levels = num_indents(min(lines, key=num_indents))
    if max_levels is not None:
        num_levels = min(num_levels, max_levels)

    # Dedent
    dedent_len = num_levels * len(indent_str)
    return "\n".join(line[dedent_len:] for line in lines)


@contextmanager
def npy_printoptions(**extra_opts):
    """Context manager to temporarily set NumPy print options.

    See Also
    --------
    numpy.get_printoptions
    numpy.set_printoptions

    Examples
    --------
    >>> print(np.array([np.nan, 1.00001]))
    [     nan  1.00001]
    >>> with npy_printoptions(precision=3):
    ...     print(np.array([np.nan, 1.00001]))
    [ nan   1.]
    >>> with npy_printoptions(nanstr='whoah!'):
    ...     print(np.array([np.nan, 1.00001]))
    [  whoah!  1.00001]
    """
    orig_opts = np.get_printoptions()

    try:
        new_opts = orig_opts.copy()
        new_opts.update(extra_opts)
        np.set_printoptions(**new_opts)
        yield

    finally:
        np.set_printoptions(**orig_opts)


def array_str(a, nprint=6):
    """Stringification of an array.

    Parameters
    ----------
    a : `array-like`
        The array to print.
    nprint : int, optional
        Maximum number of elements to print per axis in ``a``. For larger
        arrays, a summary is printed, with ``nprint // 2`` elements on
        each side and ``...`` in the middle (per axis).

    Examples
    --------
    Printing 1D arrays:

    >>> print(array_str(np.arange(4)))
    [0, 1, 2, 3]
    >>> print(array_str(np.arange(10)))
    [0, 1, 2, ..., 7, 8, 9]
    >>> print(array_str(np.arange(10), nprint=10))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    For 2D and higher, the ``nprint`` limitation applies per axis:

    >>> print(array_str(np.arange(24).reshape(4, 6)))
    [[ 0,  1,  2,  3,  4,  5],
     [ 6,  7,  8,  9, 10, 11],
     [12, 13, 14, 15, 16, 17],
     [18, 19, 20, 21, 22, 23]]
    >>> print(array_str(np.arange(32).reshape(4, 8)))
    [[ 0,  1,  2, ...,  5,  6,  7],
     [ 8,  9, 10, ..., 13, 14, 15],
     [16, 17, 18, ..., 21, 22, 23],
     [24, 25, 26, ..., 29, 30, 31]]
    >>> print(array_str(np.arange(32).reshape(8, 4)))
    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11],
     ...,
     [20, 21, 22, 23],
     [24, 25, 26, 27],
     [28, 29, 30, 31]]
    >>> print(array_str(np.arange(64).reshape(8, 8)))
    [[ 0,  1,  2, ...,  5,  6,  7],
     [ 8,  9, 10, ..., 13, 14, 15],
     [16, 17, 18, ..., 21, 22, 23],
     ...,
     [40, 41, 42, ..., 45, 46, 47],
     [48, 49, 50, ..., 53, 54, 55],
     [56, 57, 58, ..., 61, 62, 63]]

    Printing of empty arrays and 0D arrays:

    >>> print(array_str(np.array([])))  # 1D, size=0
    []
    >>> print(array_str(np.array(1.0)))  # 0D, size=1
    1.0

    Small deviations from round numbers will be suppressed:

    >>> # 2.0000000000000004 in double precision
    >>> print(array_str((np.array([2.0]) ** 0.5) ** 2))
    [ 2.]
    """
    a = asarray(a)
    a, backend = get_array_and_backend(a)
    a = backend.to_numpy(a)
    max_shape = tuple(n if n < nprint else nprint for n in a.shape)
    with npy_printoptions(
        threshold=int(np.prod(max_shape)), edgeitems=nprint // 2, suppress=True
    ):
        a_str = np.array2string(a, separator=", ")
    return a_str


def signature_string(posargs, optargs, sep=", ", mod="!r"):
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

        **Note:** The comparison is done by using ``if value == default:``,
        which is not valid for, e.g., NumPy arrays.

    sep : string or sequence of strings, optional
        Separator(s) for the argument strings. A provided single string is
        used for all joining operations.
        A given sequence must have 3 entries ``pos_sep, opt_sep, part_sep``.
        The ``pos_sep`` and ``opt_sep`` strings are used for joining the
        respective sequences of argument strings, and ``part_sep`` joins
        these two joined strings.
    mod : string or callable or sequence, optional
        Format modifier(s) for the argument strings.
        In its most general form, ``mod`` is a sequence of 2 sequences
        ``pos_mod, opt_mod`` with ``len(pos_mod) == len(posargs)`` and
        ``len(opt_mod) == len(optargs)``. Each entry ``m`` in those sequences
        can be eiter a string, resulting in the following stringification
        of ``arg``::

            arg_fmt = {{{}}}.format(m)
            arg_str = arg_fmt.format(arg)

        For a callable ``to_str``, the stringification is simply
        ``arg_str = to_str(arg)``.

        The entries ``pos_mod, opt_mod`` of ``mod`` can also be strings
        or callables instead of sequences, in which case the modifier
        applies to all corresponding arguments.

        Finally, if ``mod`` is a string or callable, it is applied to
        all arguments.

        The default behavior is to apply the "{!r}" (``repr``) conversion.
        For floating point scalars, the number of digits printed is
        determined by the ``precision`` value in NumPy's printing options,
        which can be temporarily modified with `npy_printoptions`.

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

    Using callables for stringification:

    >>> posargs = ['arg1', np.ones(3)]
    >>> optargs = []
    >>> signature_string(posargs, optargs, mod=[['', array_str], []])
    "'arg1', [ 1., 1., 1.]"

    The number of printed digits in floating point numbers can be changed
    with `npy_printoptions`:

    >>> posargs = ['hello', 0.123456789012345]
    >>> optargs = [('extent', 1.234567890123456, 1.0)]
    >>> signature_string(posargs, optargs)  # default is 8 digits
    "'hello', 0.12345679, extent=1.2345679"
    >>> with npy_printoptions(precision=2):
    ...     sig_str = signature_string(posargs, optargs)
    >>> sig_str
    "'hello', 0.12, extent=1.2"
    """
    # Define the separators for the two possible cases
    if is_string(sep):
        pos_sep = opt_sep = part_sep = sep
    else:
        pos_sep, opt_sep, part_sep = sep

    # Get the stringified parts
    posargs_conv, optargs_conv = signature_string_parts(posargs, optargs, mod)

    # Join the arguments using the separators
    parts = []
    if posargs_conv:
        parts.append(pos_sep.join(argstr for argstr in posargs_conv))
    if optargs_conv:
        parts.append(opt_sep.join(optargs_conv))

    return part_sep.join(parts)


def signature_string_parts(posargs, optargs, mod="!r"):
    """Return stringified arguments as tuples.

    Parameters
    ----------
    posargs : sequence
        Positional argument values, always included in the returned string
        tuple.
    optargs : sequence of 3-tuples
        Optional arguments with names and defaults, given in the form::

            [(name1, value1, default1), (name2, value2, default2), ...]

        Only those parameters that are different from the given default
        are included as ``name=value`` keyword pairs.

        **Note:** The comparison is done by using ``if value == default:``,
        which is not valid for, e.g., NumPy arrays.

    mod : string or callable or sequence, optional
        Format modifier(s) for the argument strings.
        In its most general form, ``mod`` is a sequence of 2 sequences
        ``pos_mod, opt_mod`` with ``len(pos_mod) == len(posargs)`` and
        ``len(opt_mod) == len(optargs)``. Each entry ``m`` in those sequences
        can be a string, resulting in the following stringification
        of ``arg``::

            arg_fmt = {{{}}}.format(m)
            arg_str = arg_fmt.format(arg)

        For a callable ``to_str``, the stringification is simply
        ``arg_str = to_str(arg)``.

        The entries ``pos_mod, opt_mod`` of ``mod`` can also be strings
        or callables instead of sequences, in which case the modifier
        applies to all corresponding arguments.

        Finally, if ``mod`` is a string or callable, it is applied to
        all arguments.

        The default behavior is to apply the "{!r}" (``repr``) conversion.
        For floating point scalars, the number of digits printed is
        determined by the ``precision`` value in NumPy's printing options,
        which can be temporarily modified with `npy_printoptions`.

    Returns
    -------
    pos_strings : tuple of str
        The stringified positional arguments.
    opt_strings : tuple of str
        The stringified optional arguments, not including the ones
        equal to their respective defaults.
    """
    # Convert modifiers to 2-sequence of sequence of strings
    if is_string(mod) or callable(mod):
        pos_mod = opt_mod = mod
    else:
        pos_mod, opt_mod = mod

    mods = []
    for m, args in zip((pos_mod, opt_mod), (posargs, optargs)):
        if is_string(m) or callable(m):
            mods.append([m] * len(args))
        else:
            if len(m) == 1:
                mods.append(m * len(args))
            elif len(m) == len(args):
                mods.append(m)
            else:
                raise ValueError(f"sequence length mismatch: len({m}) != len({args})")

    pos_mod, opt_mod = mods
    precision = np.get_printoptions()["precision"]

    # Stringify values, treating strings specially
    posargs_conv = []
    for arg, modifier in zip(posargs, pos_mod):
        if callable(modifier):
            posargs_conv.append(modifier(arg))
        elif is_string(arg):
            # Preserve single quotes for strings by default
            if modifier:
                fmt = "{{{}}}".format(modifier)
            else:
                fmt = "'{}'"
            posargs_conv.append(fmt.format(arg))
        elif np.isscalar(arg) and str(arg) in ("inf", "nan"):
            # Make sure the string quotes are added
            posargs_conv.append(f"'{arg}'")
        elif (
            np.isscalar(arg)
            and np.array(arg).real.astype("int64") != arg
            and modifier in ("", "!s", "!r")
        ):
            # Floating point value, use numpy print option 'precision'
            fmt = "{{:.{}}}".format(precision)
            posargs_conv.append(fmt.format(arg))
        else:
            # All non-string types are passed through a format conversion
            fmt = "{{{}}}".format(modifier)
            posargs_conv.append(fmt.format(arg))

    # Build 'key=value' strings for values that are not equal to default
    optargs_conv = []
    for (name, value, default), modifier in zip(optargs, opt_mod):
        if value == default:
            # Don't include
            continue

        # See above on str and repr
        if callable(modifier):
            optargs_conv.append(f"{name}={modifier(value)}")
        elif is_string(value):
            if modifier:
                fmt = "{{{}}}".format(modifier)
            else:
                fmt = "'{}'"
            value_str = fmt.format(value)
            optargs_conv.append(f"{name}={value_str}")
        elif np.isscalar(value) and str(value) in ("inf", "nan"):
            # Make sure the string quotes are added
            optargs_conv.append(f"{name}='{value}'")
        elif (
            np.isscalar(value)
            and np.array(value).real.astype("int64") != value
            and modifier in ("", "!s", "!r")
        ):
            fmt = "{{:.{}}}".format(precision)
            value_str = fmt.format(value)
            optargs_conv.append(f"{name}={value_str}")
        else:
            fmt = "{{{}}}".format(modifier)
            value_str = fmt.format(value)
            optargs_conv.append(f"{name}={value_str}")

    return tuple(posargs_conv), tuple(optargs_conv)


def _separators(strings, linewidth):
    """Return separators that keep joined strings within the line width."""
    if len(strings) <= 1:
        return ()

    indent_len = 4
    separators = []
    cur_line_len = indent_len + len(strings[0]) + 1
    if cur_line_len + 2 <= linewidth and "\n" not in strings[0]:
        # Next string might fit on same line
        separators.append(", ")
        cur_line_len += 1  # for the extra space
    else:
        # Use linebreak if string contains newline or doesn't fit
        separators.append(",\n")
        cur_line_len = indent_len

    for i, s in enumerate(strings[1:-1]):
        cur_line_len += len(s) + 1

        if "\n" in s:
            # Use linebreak before and after if string contains newline
            separators[i] = ",\n"
            cur_line_len = indent_len
            separators.append(",\n")

        elif cur_line_len + 2 <= linewidth:
            # This string fits, next one might also fit on same line
            separators.append(", ")
            cur_line_len += 1  # for the extra space

        elif cur_line_len <= linewidth:
            # This string fits, but next one won't
            separators.append(",\n")
            cur_line_len = indent_len

        else:
            # This string doesn't fit but has no newlines in it
            separators[i] = ",\n"
            cur_line_len = indent_len + len(s) + 1

            # Need to determine again what should come next
            if cur_line_len + 2 <= linewidth:
                # Next string might fit on same line
                separators.append(", ")
            else:
                separators.append(",\n")

    cur_line_len += len(strings[-1])
    if cur_line_len + 1 > linewidth or "\n" in strings[-1]:
        # This string and a comma don't fit on this line
        separators[-1] = ",\n"

    return tuple(separators)


def repr_string(outer_string, inner_strings, allow_mixed_seps=True):
    r"""Return a pretty string for ``repr``.

    The returned string is formatted such that it does not extend
    beyond the line boundary if avoidable. The line width is taken from
    NumPy's printing options that can be retrieved with
    `numpy.get_printoptions`. They can be temporarily overridden
    using the `npy_printoptions` context manager. See Examples for details.

    Parameters
    ----------
    outer_string : str
        Name of the class or function that should be printed outside
        the parentheses.
    inner_strings : sequence of sequence of str
        Stringifications of the positional and optional arguments.
        This is usually the return value of `signature_string_parts`.
    allow_mixed_seps : bool, optional
        If ``False`` and the string does not fit on one line, use
        ``',\n'`` to separate all strings.
        By default, a mixture of ``', '`` and ``',\n'`` is used to fit
        as much on one line as possible.

        In case some of the ``inner_strings`` span multiple lines, it is
        usually advisable to set ``allow_mixed_seps`` to ``False`` since
        the result tends to be more readable that way.

    Returns
    -------
    repr_string : str
        Full string that can be returned by a class' ``__repr__`` method.

    Examples
    --------
    Things that fit into one line are printed on one line:

    >>> outer_string = 'MyClass'
    >>> inner_strings = [('1', "'hello'", 'None'),
    ...                  ("dtype='float32'",)]
    >>> print(repr_string(outer_string, inner_strings))
    MyClass(1, 'hello', None, dtype='float32')

    Otherwise, if a part of ``inner_strings`` fits on a line of its own,
    it is printed on one line, but separated from the other part with
    a line break:

    >>> outer_string = 'MyClass'
    >>> inner_strings = [('2.0', "'this_is_a_very_long_argument_string'"),
    ...                  ("long_opt_arg='another_quite_long_string'",)]
    >>> print(repr_string(outer_string, inner_strings))
    MyClass(
        2.0, 'this_is_a_very_long_argument_string',
        long_opt_arg='another_quite_long_string'
    )

    If those parts are themselves too long, they are broken down into
    several lines:

    >>> outer_string = 'MyClass'
    >>> inner_strings = [("'this_is_a_very_long_argument_string'",
    ...                   "'another_very_long_argument_string'"),
    ...                  ("long_opt_arg='another_quite_long_string'",
    ...                   "long_opt2_arg='this_wont_fit_on_one_line_either'")]
    >>> print(repr_string(outer_string, inner_strings))
    MyClass(
        'this_is_a_very_long_argument_string',
        'another_very_long_argument_string',
        long_opt_arg='another_quite_long_string',
        long_opt2_arg='this_wont_fit_on_one_line_either'
    )

    The usage of mixed separators to optimally use horizontal space can
    be disabled by setting ``allow_mixed_seps=False``:

    >>> outer_string = 'MyClass'
    >>> inner_strings = [('2.0', "'this_is_a_very_long_argument_string'"),
    ...                  ("long_opt_arg='another_quite_long_string'",)]
    >>> print(repr_string(outer_string, inner_strings, allow_mixed_seps=False))
    MyClass(
        2.0,
        'this_is_a_very_long_argument_string',
        long_opt_arg='another_quite_long_string'
    )

    With the ``npy_printoptions`` context manager, the available line
    width can be changed:

    >>> outer_string = 'MyClass'
    >>> inner_strings = [('1', "'hello'", 'None'),
    ...                  ("dtype='float32'",)]
    >>> with npy_printoptions(linewidth=20):
    ...     print(repr_string(outer_string, inner_strings))
    MyClass(
        1, 'hello',
        None,
        dtype='float32'
    )
    """
    linewidth = np.get_printoptions()["linewidth"]
    pos_strings, opt_strings = inner_strings
    # Length of the positional and optional argument parts of the signature,
    # including separators `', '`
    pos_sig_len = sum(len(pstr) for pstr in pos_strings) + 2 * max(
        (len(pos_strings) - 1), 0
    )
    opt_sig_len = sum(len(pstr) for pstr in opt_strings) + 2 * max(
        (len(opt_strings) - 1), 0
    )

    # Length of the one-line string, including 2 for the parentheses and
    # 2 for the joining ', '
    repr_len = len(outer_string) + 2 + pos_sig_len + 2 + opt_sig_len

    if repr_len <= linewidth and not any("\n" in s for s in pos_strings + opt_strings):
        # Everything fits on one line
        fmt = "{}({})"
        pos_str = ", ".join(pos_strings)
        opt_str = ", ".join(opt_strings)
        parts_sep = ", "
    else:
        # Need to split lines in some way
        fmt = "{}(\n{}\n)"

        if not allow_mixed_seps:
            pos_separators = [",\n"] * (len(pos_strings) - 1)
        else:
            pos_separators = _separators(pos_strings, linewidth)
        if len(pos_strings) == 0:
            pos_str = ""
        else:
            pos_str = pos_strings[0]
            for s, sep in zip(pos_strings[1:], pos_separators):
                pos_str = sep.join([pos_str, s])

        if not allow_mixed_seps:
            opt_separators = [",\n"] * (len(opt_strings) - 1)
        else:
            opt_separators = _separators(opt_strings, linewidth)
        if len(opt_strings) == 0:
            opt_str = ""
        else:
            opt_str = opt_strings[0]
            for s, sep in zip(opt_strings[1:], opt_separators):
                opt_str = sep.join([opt_str, s])

        # Check if we can put both parts on one line. This requires their
        # concatenation including 4 for indentation and 2 for ', ' to
        # be less than the line width. And they should contain no newline.
        if pos_str and opt_str:
            inner_len = 4 + len(pos_str) + 2 + len(opt_str)
        elif (pos_str and not opt_str) or (opt_str and not pos_str):
            inner_len = 4 + len(pos_str) + len(opt_str)
        else:
            inner_len = 0

        if (
            not allow_mixed_seps
            or any("\n" in s for s in [pos_str, opt_str])
            or inner_len > linewidth
        ):
            parts_sep = ",\n"
            pos_str = indent(pos_str)
            opt_str = indent(opt_str)
        else:
            parts_sep = ", "
            pos_str = indent(pos_str)
            # Don't indent `opt_str`

    parts = [s for s in [pos_str, opt_str] if s.strip()]  # ignore empty
    inner_string = parts_sep.join(parts)
    return fmt.format(outer_string, inner_string)


def attribute_repr_string(inst_str, attr_str):
    """Return a repr string for an attribute that respects line width.

    Parameters
    ----------
    inst_str : str
        Stringification of a class instance.
    attr_str : str
        Name of the attribute (not including the ``'.'``).

    Returns
    -------
    attr_repr_str : str
        Concatenation of the two strings in a way that the line width
        is respected.

    Examples
    --------
    >>> inst_str = 'rn((2, 3))'
    >>> attr_str = 'byaxis'
    >>> print(attribute_repr_string(inst_str, attr_str))
    rn((2, 3)).byaxis
    >>> inst_str = 'MyClass()'
    >>> attr_str = 'attr_name'
    >>> print(attribute_repr_string(inst_str, attr_str))
    MyClass().attr_name
    >>> inst_str = 'MyClass'
    >>> attr_str = 'class_attr'
    >>> print(attribute_repr_string(inst_str, attr_str))
    MyClass.class_attr
    >>> long_inst_str = (
    ...     "MyClass('long string that will definitely trigger a line break')"
    ... )
    >>> long_attr_str = 'long_attribute_name'
    >>> print(attribute_repr_string(long_inst_str, long_attr_str))
    MyClass(
        'long string that will definitely trigger a line break'
    ).long_attribute_name
    """
    linewidth = np.get_printoptions()["linewidth"]
    if len(inst_str) + 1 + len(attr_str) <= linewidth or "(" not in inst_str:
        # Instance string + dot + attribute string fit in one line or
        # no parentheses -> keep instance string as-is and append attr string
        parts = [inst_str, attr_str]
    else:
        # TODO(kohr-h): use `maxsplit=1` kwarg, not supported in Py 2
        left, rest = inst_str.split("(", 1)
        right, middle = rest[::-1].split(")", 1)
        middle, right = middle[::-1], right[::-1]

        if middle.startswith("\n") and middle.endswith("\n"):
            # Already on multiple lines
            new_inst_str = inst_str
        else:
            init_parts = [left]
            if middle:
                init_parts.append(indent(middle))
            new_inst_str = "(\n".join(init_parts) + "\n)" + right
        parts = [new_inst_str, attr_str]

    return ".".join(parts)


def method_repr_string(inst_str, meth_str, arg_strs=None, allow_mixed_seps=True):
    r"""Return a repr string for a method that respects line width.

    This function is useful to generate a ``repr`` string for a derived
    class that is created through a method, for instance ::

        functional.translated(x)

    as a better way of representing ::

        FunctionalTranslation(functional, x)

    Parameters
    ----------
    inst_str : str
        Stringification of a class instance.
    meth_str : str
        Name of the method (not including the ``'.'``).
    arg_strs : sequence of str, optional
        Stringification of the arguments to the method.
    allow_mixed_seps : bool, optional
        If ``False`` and the argument strings do not fit on one line, use
        ``',\n'`` to separate all strings.
        By default, a mixture of ``', '`` and ``',\n'`` is used to fit
        as much on one line as possible.

        In case some of the ``arg_strs`` span multiple lines, it is
        usually advisable to set ``allow_mixed_seps`` to ``False`` since
        the result tends to be more readable that way.

    Returns
    -------
    meth_repr_str : str
        Concatenation of all strings in a way that the line width
        is respected.

    Examples
    --------
    >>> inst_str = 'MyClass'
    >>> meth_str = 'empty'
    >>> arg_strs = []
    >>> print(method_repr_string(inst_str, meth_str, arg_strs))
    MyClass.empty()
    >>> inst_str = 'MyClass'
    >>> meth_str = 'fromfile'
    >>> arg_strs = ["'tmpfile.txt'"]
    >>> print(method_repr_string(inst_str, meth_str, arg_strs))
    MyClass.fromfile('tmpfile.txt')
    >>> inst_str = "MyClass('init string')"
    >>> meth_str = 'method'
    >>> arg_strs = ['2.0']
    >>> print(method_repr_string(inst_str, meth_str, arg_strs))
    MyClass('init string').method(2.0)
    >>> long_inst_str = (
    ...     "MyClass('long string that will definitely trigger a line break')"
    ... )
    >>> meth_str = 'method'
    >>> long_arg1 = "'long argument string that should come on the next line'"
    >>> arg2 = 'param1=1'
    >>> arg3 = 'param2=2.0'
    >>> arg_strs = [long_arg1, arg2, arg3]
    >>> print(method_repr_string(long_inst_str, meth_str, arg_strs))
    MyClass(
        'long string that will definitely trigger a line break'
    ).method(
        'long argument string that should come on the next line',
        param1=1, param2=2.0
    )
    >>> print(method_repr_string(long_inst_str, meth_str, arg_strs,
    ...                          allow_mixed_seps=False))
    MyClass(
        'long string that will definitely trigger a line break'
    ).method(
        'long argument string that should come on the next line',
        param1=1,
        param2=2.0
    )
    """
    linewidth = np.get_printoptions()["linewidth"]

    # Part up to the method name
    if len(inst_str) + 1 + len(meth_str) + 1 <= linewidth or "(" not in inst_str:
        init_parts = [inst_str, meth_str]
        # Length of the line to the end of the method name
        meth_line_start_len = len(inst_str) + 1 + len(meth_str)
    else:
        # TODO(kohr-h): use `maxsplit=1` kwarg, not supported in Py 2
        left, rest = inst_str.split("(", 1)
        right, middle = rest[::-1].split(")", 1)
        middle, right = middle[::-1], right[::-1]
        if middle.startswith("\n") and middle.endswith("\n"):
            # Already on multiple lines
            new_inst_str = inst_str
        else:
            new_inst_str = "(\n".join([left, indent(middle)]) + "\n)" + right

        # Length of the line to the end of the method name, consisting of
        # ')' + '.' + <method name>
        meth_line_start_len = 1 + 1 + len(meth_str)
        init_parts = [new_inst_str, meth_str]

    # Method call part
    arg_str_oneline = ", ".join(arg_strs)
    if meth_line_start_len + 1 + len(arg_str_oneline) + 1 <= linewidth:
        meth_call_str = "(" + arg_str_oneline + ")"
    elif not arg_str_oneline:
        meth_call_str = "(\n)"
    else:
        if allow_mixed_seps:
            arg_seps = _separators(arg_strs, linewidth - 4)  # indented
        else:
            arg_seps = [",\n"] * (len(arg_strs) - 1)

        full_arg_str = ""
        for arg_str, sep in zip_longest(arg_strs, arg_seps, fillvalue=""):
            full_arg_str += arg_str + sep

        meth_call_str = "(\n" + indent(full_arg_str) + "\n)"

    return ".".join(init_parts) + meth_call_str
