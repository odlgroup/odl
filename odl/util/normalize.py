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

"""Utilities for normalization of user input."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('normalized_scalar_param_list', 'normalized_index_expression',
           'normalized_nodes_on_bdry', 'normalized_axes_tuple')


def normalized_scalar_param_list(param, length, param_conv=None,
                                 keep_none=True, return_nonconv=False):
    """Return a list of given length from a scalar parameter.

    The typical use case is when a single value or a sequence of
    values is accepted as input. This function makes a list from
    a given sequence or a list of identical elements from a single
    value, with cast to a given parameter type if desired.

    To distinguish a parameter sequence from a single parameter, the
    following rules are applied:

    * If ``param`` is not a sequence, it is treated as a single
      parameter (e.g. ``1``).
    * If ``len(param) == length == 1``, then ``param`` is interpreted
      as a single parameter (e.g. ``[1]`` or ``'1'``).
    * If ``len(param) == length != 1``, then ``param`` is interpreted as
      sequence of parameters.
    * Otherwise, ``param`` is interpreted as a single parameter.

    Note that this function is not applicable to parameters which are
    themselves iterable (e.g. ``'abc'`` with ``length=3`` will be
    interpreted as equivalent to ``['a', 'b', 'c']``).

    Parameters
    ----------
    param :
        Input parameter to turn into a list.
    length : positive int
        Desired length of the output list.
    param_conv : callable, optional
        Conversion applied to each list element. ``None`` means no conversion.
    keep_none : bool, optional
        If ``True``, ``None`` is not converted.
    return_nonconv : bool, optional
        If ``True``, return also the list where no conversion has been
        applied.

    Returns
    -------
    plist : list
        Input parameter turned into a list of length ``length``.
    nonconv : list
        The same as ``plist``, but without conversion. This is only
        returned if ``return_nonconv == True``.

    Examples
    --------
    Turn input into a list of given length, possibly by broadcasting.
    By default, no conversion is performed.

    >>> normalized_scalar_param_list((1, 2, 3), 3)
    [1, 2, 3]
    >>> normalized_scalar_param_list((1, None, 3.0), 3)
    [1, None, 3.0]

    Single parameters are broadcast to the given length.

    >>> normalized_scalar_param_list(1, 3)
    [1, 1, 1]
    >>> normalized_scalar_param_list('10', 3)
    ['10', '10', '10']
    >>> normalized_scalar_param_list(None, 3)
    [None, None, None]

    List entries can be explicitly converted using ``param_conv``. If
    ``None`` should be kept, set ``keep_none`` to ``True``:

    >>> normalized_scalar_param_list(1, 3, param_conv=float)
    [1.0, 1.0, 1.0]
    >>> normalized_scalar_param_list('10', 3, param_conv=int)
    [10, 10, 10]
    >>> normalized_scalar_param_list((1, None, 3.0), 3, param_conv=int,
    ...                              keep_none=True)  # default
    [1, None, 3]

    The conversion parameter can be any callable:

    >>> def myconv(x):
    ...     return False if x is None else bool(x)
    >>> normalized_scalar_param_list((0, None, 3.0), 3, param_conv=myconv,
    ...                              keep_none=False)
    [False, False, True]
    """
    length, length_in = int(length), length
    if length <= 0:
        raise ValueError('`length` must be positive, got {}'.format(length_in))

    try:
        # TODO: always use this when numpy >= 1.10 can be assumed
        param = np.array(param, dtype=object, copy=True, ndmin=1)
        nonconv_list = list(np.broadcast_to(param, (length,)))
    except AttributeError:
        # numpy.broadcast_to not available
        if np.isscalar(param):
            # Try this first, will work better with iterable input like '10'
            nonconv_list = [param] * length
        else:
            try:
                param_len = len(param)
            except TypeError:
                # Not a sequence -> single parameter
                nonconv_list = [param] * length
            else:
                if param_len == 1:
                    nonconv_list = list(param) * length
                elif param_len == length:
                    nonconv_list = list(param)
                else:
                    raise ValueError('sequence `param` has length {}, '
                                     'expected {}'.format(param_len, length))

    if len(nonconv_list) != length:
        raise ValueError('sequence `param` has length {}, expected {}'
                         ''.format(len(nonconv_list), length))

    if param_conv is None:
        out_list = list(nonconv_list)
    else:
        out_list = []
        for p in nonconv_list:
            if p is None and keep_none:
                out_list.append(p)
            else:
                out_list.append(param_conv(p))

    if return_nonconv:
        return out_list, nonconv_list
    else:
        return out_list


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
    indices : int, `slice`, `Ellipsis` or sequence of these
        Index expression to be normalized.
    shape : sequence of ints
        Target shape for error checking of out-of-bounds indices.
        Also needed to determine the number of axes.
    int_to_slice : bool, optional
        If ``True``, turn integers into corresponding slice objects.

    Returns
    -------
    normalized : tuple of ints or `slice`'s
        Normalized index expression

    Examples
    --------
    Sequences are turned into tuples. We can have at most as many entries
    as the length of ``shape``, but fewer are allowed - the remaining
    list places are filled up by ``slice(None)``:

    >>> normalized_index_expression([1, 2, 3], shape=(3, 4, 5))
    (1, 2, 3)
    >>> normalized_index_expression([1, 2], shape=(3, 4, 5))
    (1, 2, slice(None, None, None))
    >>> normalized_index_expression([slice(2), 2], shape=(3, 4, 5))
    (slice(None, 2, None), 2, slice(None, None, None))
    >>> normalized_index_expression([1, Ellipsis], shape=(3, 4, 5))
    (1, slice(None, None, None), slice(None, None, None))

    By default, integer indices are kept. If they should be converted
    to slices, use ``int_to_slice=True``. This can be useful to guarantee
    that the result of slicing with the returned object is of the same
    type as the array into which is sliced and has the same number of
    axes:

    >>> x = np.zeros(shape=(3, 4, 5))
    >>> idx1 = normalized_index_expression([1, 2, 3], shape=(3, 4, 5),
    ...                                   int_to_slice=True)
    >>> idx1
    (slice(1, 2, None), slice(2, 3, None), slice(3, 4, None))
    >>> x[idx1]
    array([[[ 0.]]])
    >>> idx2 = normalized_index_expression([1, 2, 3], shape=(3, 4, 5),
    ...                                   int_to_slice=False)
    >>> idx2
    (1, 2, 3)
    >>> x[idx2]
    0.0
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
            raise ValueError('cannot use more than one Ellipsis.')

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
                raise IndexError('Index {} is out of bounds for axis '
                                 '{} with size {}.'
                                 ''.format(idx, i, n))
            if int_to_slice:
                indices[i] = slice(idx, idx + 1)

    # Catch most common errors
    if any(s.start == s.stop and s.start is not None or
           s.start == n
           for s, n in zip(indices, shape) if isinstance(s, slice)):
        raise ValueError('Slices with empty axes not allowed.')
    if None in indices:
        raise ValueError('creating new axes is not supported.')
    if len(indices) > ndim:
        raise IndexError('too may indices: {} > {}.'
                         ''.format(len(indices), ndim))

    return tuple(indices)


def normalized_nodes_on_bdry(nodes_on_bdry, length):
    """Return a list of 2-tuples of bool from the input parameter.

    This function is intended to normalize a ``nodes_on_bdry`` parameter
    that can be given as a single boolean (global) or as a sequence
    (per axis). Each entry of the sequence can either be a single
    boolean (global for the axis) or a boolean sequence of length 2.

    Parameters
    ----------
    nodes_on_bdry : bool or sequence
        Input parameter to be normalized according to the above scheme.
    length : positive int
        Desired length of the returned list.

    Returns
    -------
    normalized : list of 2-tuples of bool
        Normalized list with ``length`` entries, each of which is a
        2-tuple of boolean values.

    Examples
    --------
    Global for all axes:

    >>> normalized_nodes_on_bdry(True, length=2)
    [(True, True), (True, True)]

    Global per axis:

    >>> normalized_nodes_on_bdry([True, False], length=2)
    [(True, True), (False, False)]

    Mixing global and explicit per axis:

    >>> normalized_nodes_on_bdry([[True, False], False, True], length=3)
    [(True, False), (False, False), (True, True)]
    """
    shape = np.shape(nodes_on_bdry)
    if shape == ():
        out_list = [(bool(nodes_on_bdry), bool(nodes_on_bdry))] * length
    elif length == 1 and shape == (2,):
        out_list = [(bool(nodes_on_bdry[0]), bool(nodes_on_bdry[1]))]
    elif len(nodes_on_bdry) == length:
        out_list = []

        for i, on_bdry in enumerate(nodes_on_bdry):
            shape_i = np.shape(on_bdry)
            if shape_i == ():
                out_list.append((bool(on_bdry), bool(on_bdry)))
            elif shape_i == (2,):
                out_list.append((bool(on_bdry[0]), bool(on_bdry[1])))
            else:
                raise ValueError('in axis {}: `nodes_on_bdry` has shape {}, '
                                 'expected (2,)'
                                 .format(i, shape_i))
    else:
        raise ValueError('`nodes_on_bdry` has shape {}, expected ({},)'
                         ''.format(shape, length))

    return out_list


def normalized_axes_tuple(axes, ndim):
    """Return a tuple of ``axes`` converted to positive integers.

    This function turns negative entries into equivalent positive
    ones according to standard Python indexing "from the right".

    Parameters
    ----------
    axes : int or sequence of ints
        Single integer or integer sequence of arbitrary length.
        Duplicate entries are not allowed. All entries must fulfill
        ``-ndim <= axis <= ndim - 1``.
    ndim : positive int
        Number of available axes determining the valid axis range.

    Returns
    -------
    axes_list : tuple of ints
        The converted tuple of axes.

    Examples
    --------
    Normalizing a sequence of axes:

    >>> normalized_axes_tuple([0, -1, 2], ndim=3)
    (0, 2, 2)

    Single integer works, too:

    >>> normalized_axes_tuple(-3, ndim=3)
    (0,)
    """
    try:
        axes, axes_in = (int(axes),), axes
    except TypeError:
        axes, axes_in = tuple(int(axis) for axis in axes), axes
        if any(axis != axis_in for axis, axis_in in zip(axes, axes_in)):
            raise ValueError('`axes` may only contain integers, got {}'
                             ''.format(axes_in))
    else:
        if axes[0] != axes_in:
            raise TypeError('`axes` must be integer or sequence, got {}'
                            ''.format(axes_in))

    if len(set(axes)) != len(axes):
        raise ValueError('`axes` may not contain duplicate entries')

    ndim, ndim_in = int(ndim), ndim
    if ndim <= 0:
        raise ValueError('`ndim` must be positive, got {}'.format(ndim_in))

    axes_arr = np.array(axes)
    axes_arr[axes_arr < 0] += ndim
    if np.any((axes_arr < 0) | (axes_arr >= ndim)):
        raise ValueError('all `axes` entries must satisfy -{0} <= axis < {0}, '
                         'got {1}'.format(ndim, axes_in))

    return tuple(axes_arr)


def safe_int_conv(number):
    """Safely convert a single number to integer."""
    try:
        return int(np.array(number).astype(int, casting='safe'))
    except TypeError:
        raise ValueError('cannot safely convert {} to integer'.format(number))


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
