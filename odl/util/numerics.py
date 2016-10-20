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

"""Numerical helper functions for convenience or speed."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.util.normalize import normalized_scalar_param_list, safe_int_conv


__all__ = ('apply_on_boundary', 'fast_1d_tensor_mult', 'resize_array')


_SUPPORTED_RESIZE_PAD_MODES = ('constant', 'symmetric', 'periodic',
                               'order0', 'order1')


def apply_on_boundary(array, func, only_once=True, which_boundaries=None,
                      axis_order=None, out=None):
    """Apply a function of the boundary of an n-dimensional array.

    All other values are preserved as is.

    Parameters
    ----------
    array : `array-like`
        Modify the boundary of this array
    func : callable or sequence of callables
        If a single function is given, assign
        ``array[slice] = func(array[slice])`` on the boundary slices,
        e.g. use ``lamda x: x / 2`` to divide values by 2.
        A sequence of functions is applied per axis separately. It
        must have length ``array.ndim`` and may consist of one function
        or a 2-tuple of functions per axis.
        ``None`` entries in a sequence cause the axis (side) to be
        skipped.
    only_once : bool, optional
        If ``True``, ensure that each boundary point appears in exactly
        one slice. If ``func`` is a list of functions, the
        ``axis_order`` determines which functions are applied to nodes
        which appear in multiple slices, according to the principle
        "first-come, first-served".
    which_boundaries : sequence, optional
        If provided, this sequence determines per axis whether to
        apply the function at the boundaries in each axis. The entry
        in each axis may consist in a single bool or a 2-tuple of
        bool. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``array.ndim``. ``None`` is interpreted as
        "all boundaries".
    axis_order : sequence of ints, optional
        Permutation of ``range(array.ndim)`` defining the order in which
        to process the axes. If combined with ``only_once`` and a
        function list, this determines which function is evaluated in
        the points that are potentially processed multiple times.
    out : `numpy.ndarray`, optional
        Location in which to store the result, can be the same as ``array``.
        Default: copy of ``array``

    Examples
    --------
    >>> arr = np.ones((3, 3))
    >>> apply_on_boundary(arr, lambda x: x / 2)
    array([[ 0.5,  0.5,  0.5],
           [ 0.5,  1. ,  0.5],
           [ 0.5,  0.5,  0.5]])

    If called with ``only_once=False``, the function is applied repeatedly:

    >>> apply_on_boundary(arr, lambda x: x / 2, only_once=False)
    array([[ 0.25,  0.5 ,  0.25],
           [ 0.5 ,  1.  ,  0.5 ],
           [ 0.25,  0.5 ,  0.25]])

    >>> apply_on_boundary(arr, lambda x: x / 2, only_once=True,
    ...                   which_boundaries=((True, False), True))
    array([[ 0.5,  0.5,  0.5],
           [ 0.5,  1. ,  0.5],
           [ 0.5,  1. ,  0.5]])

    Use the ``out`` parameter to store the result in an existing array:

    >>> out = np.empty_like(arr)
    >>> result = apply_on_boundary(arr, lambda x: x / 2, out=out)
    >>> result
    array([[ 0.5,  0.5,  0.5],
           [ 0.5,  1. ,  0.5],
           [ 0.5,  0.5,  0.5]])
    >>> result is out
    True
    """
    array = np.asarray(array)

    if callable(func):
        func = [func] * array.ndim
    elif len(func) != array.ndim:
        raise ValueError('sequence of functions has length {}, expected {}'
                         ''.format(len(func), array.ndim))

    if which_boundaries is None:
        which_boundaries = ([(True, True)] * array.ndim)
    elif len(which_boundaries) != array.ndim:
        raise ValueError('`which_boundaries` has length {}, expected {}'
                         ''.format(len(which_boundaries), array.ndim))

    if axis_order is None:
        axis_order = list(range(array.ndim))
    elif len(axis_order) != array.ndim:
        raise ValueError('`axis_order` has length {}, expected {}'
                         ''.format(len(axis_order), array.ndim))

    if out is None:
        out = array.copy()
    else:
        out[:] = array  # Self assignment is free, in case out is array

    # The 'only_once' functionality is implemented by storing for each axis
    # if the left and right boundaries have been processed. This information
    # is stored in a list of slices which is reused for the next axis in the
    # list.
    slices = [slice(None)] * array.ndim
    for ax, function, which in zip(axis_order, func, which_boundaries):
        if only_once:
            slc_l = list(slices)  # Make a copy; copy() exists in Py3 only
            slc_r = list(slices)
        else:
            slc_l = [slice(None)] * array.ndim
            slc_r = [slice(None)] * array.ndim

        # slc_l and slc_r select left and right boundary, resp, in this axis.
        slc_l[ax] = 0
        slc_r[ax] = -1

        try:
            # Tuple of functions in this axis
            func_l, func_r = function
        except TypeError:
            # Single function
            func_l = func_r = function

        try:
            # Tuple of bool
            mod_left, mod_right = which
        except TypeError:
            # Single bool
            mod_left = mod_right = which

        if mod_left and func_l is not None:
            out[slc_l] = func_l(out[slc_l])
            start = 1
        else:
            start = None

        if mod_right and func_r is not None:
            out[slc_r] = func_r(out[slc_r])
            end = -1
        else:
            end = None

        # Write the information for the processed axis into the slice list.
        # Start and end include the boundary if it was processed.
        slices[ax] = slice(start, end)

    return out


def fast_1d_tensor_mult(ndarr, onedim_arrs, axes=None, out=None):
    """Fast multiplication of an n-dim array with an outer product.

    This method implements the multiplication of an n-dimensional array
    with an outer product of one-dimensional arrays, e.g.::

        a = np.ones((10, 10, 10))
        x = np.random.rand(10)
        a *= x[:, None, None] * x[None, :, None] * x[None, None, :]

    Basically, there are two ways to do such an operation:

    1. First calculate the factor on the right-hand side and do one
       "big" multiplication; or
    2. Multiply by one factor at a time.

    The procedure of building up the large factor in the first method
    is relatively cheap if the number of 1d arrays is smaller than the
    number of dimensions. For exactly n vectors, the second method is
    faster, although it loops of the array ``a`` n times.

    This implementation combines the two ideas into a hybrid scheme:

    - If there are less 1d arrays than dimensions, choose 1.
    - Otherwise, calculate the factor array for n-1 arrays
      and multiply it to the large array. Finally, multiply with the
      last 1d array.

    The advantage of this approach is that it is memory-friendly and
    loops over the big array only twice.

    Parameters
    ----------
    ndarr : `array-like`
        Array to multiply to
    onedim_arrs : sequence of `array-like`'s
        One-dimensional arrays to be multiplied with ``ndarr``. The
        sequence may not be longer than ``ndarr.ndim``.
    axes : sequence of ints, optional
        Take the 1d transform along these axes. ``None`` corresponds to
        the last ``len(onedim_arrs)`` axes, in ascending order.
    out : `numpy.ndarray`, optional
        Array in which the result is stored

    Returns
    -------
    out : `numpy.ndarray`
        Result of the modification. If ``out`` was given, the returned
        object is a reference to it.
    """
    if out is None:
        out = np.array(ndarr, copy=True)
    else:
        out[:] = ndarr  # Self-assignment is free if out is ndarr

    if not onedim_arrs:
        raise ValueError('no 1d arrays given')

    if axes is None:
        axes = list(range(out.ndim - len(onedim_arrs), out.ndim))
        axes_in = None
    elif len(axes) != len(onedim_arrs):
        raise ValueError('there are {} 1d arrays, but {} axes entries'
                         ''.format(len(onedim_arrs), len(axes)))
    else:
        # Make axes positive
        axes, axes_in = np.array(axes, dtype=int), axes
        axes[axes < 0] += out.ndim
        axes = list(axes)

    if np.any(np.array(axes) >= out.ndim) or np.any(np.array(axes) < 0):
        raise ValueError('`axes` {} out of bounds for {} dimensions'
                         ''.format(axes_in, out.ndim))

    # Make scalars 1d arrays and squeezable arrays 1d
    alist = [np.atleast_1d(np.asarray(a).squeeze()) for a in onedim_arrs]
    if any(a.ndim != 1 for a in alist):
        raise ValueError('only 1d arrays allowed')

    if len(axes) < out.ndim:
        # Make big factor array (start with 0d)
        factor = np.array(1.0)
        for ax, arr in zip(axes, alist):
            # Meshgrid-style slice
            slc = [None] * out.ndim
            slc[ax] = slice(None)
            factor = factor * arr[slc]

        out *= factor

    else:
        # Hybrid approach

        # Get the axis to spare for the final multiplication, the one
        # with the largest stride.
        axis_order = np.argsort(out.strides)
        last_ax = axis_order[-1]
        last_arr = alist[axes.index(last_ax)]

        # Build the semi-big array and multiply
        factor = np.array(1.0)
        for ax, arr in zip(axes, alist):
            if ax == last_ax:
                continue

            slc = [None] * out.ndim
            slc[ax] = np.s_[:]
            factor = factor * arr[slc]

        out *= factor

        # Finally multiply by the remaining 1d array
        slc = [None] * out.ndim
        slc[last_ax] = np.s_[:]
        out *= last_arr[slc]

    return out


def resize_array(arr, newshp, offset=None, pad_mode='constant', pad_const=0,
                 direction='forward', out=None):
    """Return the resized version of ``arr`` with shape ``newshp``.

    In axes where ``newshp > arr.shape``, padding is applied according
    to the supplied options.
    Where ``newshp < arr.shape``, the array is cropped to the new
    size.

    See `the online documentation
    <https://odlgroup.github.io/odl/math/resizing_ops.html>`_
    on resizing operators for mathematical details.

    Parameters
    ----------
    arr : `array-like`
        Array to be resized.
    newshp : sequence of ints
        Desired shape of the output array.
    offset: sequence of ints, optional
        Specifies how many entries are added to/removed from the "left"
        side (corresponding to low indices) of ``arr``.
    pad_mode : string, optional
        Method to be used to fill in missing values in an enlarged array.

        ``'constant'``: Fill with ``pad_const``.

        ``'symmetric'``: Reflect at the boundaries, not doubling the
        outmost values. This requires left and right padding sizes
        to be strictly smaller than the original array shape.

        ``'periodic'``: Fill in values from the other side, keeping
        the order. This requires left and right padding sizes to be
        at most as large as the original array shape.

        ``'order0'``: Extend constantly with the outmost values
        (ensures continuity).

        ``'order1'``: Extend with constant slope (ensures continuity of
        the first derivative). This requires at least 2 values along
        each axis where padding is applied.

    pad_const : scalar, optional
        Value to be used in the ``'constant'`` padding mode.
    direction : {'forward', 'adjoint'}
        Determines which variant of the resizing is applied.

        'forward' : in axes where ``out`` is larger than ``arr``,
        apply padding. Otherwise, restrict to the smaller size.

        'adjoint' : in axes where ``out`` is larger than ``arr``,
        apply zero-padding. Otherwise, restrict to the smaller size
        and add the outside contributions according to ``pad_mode``.

    out : `numpy.ndarray`, optional
        Array to write the result to. Must have shape ``newshp`` and
        be able to hold the data type of the input array.

    Returns
    -------
    resized : `numpy.ndarray`
        Resized array created according to the above rules. If ``out``
        was given, the returned object is a reference to it.

    Examples
    --------
    The input can be shrunk by simply providing a smaller size.
    By default, values are removed from the right. When enlarging,
    zero-padding is applied by default, and the zeros are added to
    the right side. That behavior can be changed with the ``offset``
    parameter:

    >>> from odl.util.numerics import resize_array
    >>> resize_array([1, 2, 3], (1,))
    array([1])
    >>> resize_array([1, 2, 3], (1,), offset=2)
    array([3])
    >>> resize_array([1, 2, 3], (6,))
    array([1, 2, 3, 0, 0, 0])
    >>> resize_array([1, 2, 3], (7,), offset=2)
    array([0, 0, 1, 2, 3, 0, 0])

    The padding constant can be changed, as well as the padding
    mode:

    >>> resize_array([1, 2, 3], (7,), pad_const=-1, offset=2)
    array([-1, -1,  1,  2,  3, -1, -1])
    >>> resize_array([1, 2, 3], (7,), pad_mode='periodic', offset=2)
    array([2, 3, 1, 2, 3, 1, 2])
    >>> resize_array([1, 2, 3], (7,), pad_mode='symmetric', offset=2)
    array([3, 2, 1, 2, 3, 2, 1])
    >>> resize_array([1, 2, 3], (7,), pad_mode='order0', offset=2)
    array([1, 1, 1, 2, 3, 3, 3])
    >>> resize_array([1, 2, 3], (7,), pad_mode='order1', offset=2)
    array([-1,  0,  1,  2,  3,  4,  5])

    Everything works for arbitrary number of dimensions:

    >>> # Take the middle two columns and extend rows symmetrically
    >>> resize_array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12]],
    ...               (5, 2), pad_mode='symmetric', offset=[1, 1])
    array([[ 6,  7],
           [ 2,  3],
           [ 6,  7],
           [10, 11],
           [ 6,  7]])
    >>> # Take the rightmost two columns and extend rows symmetrically
    >>> # downwards
    >>> resize_array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12]], (5, 2), pad_mode='symmetric',
    ...              offset=[0, 2])
    array([[ 3,  4],
           [ 7,  8],
           [11, 12],
           [ 7,  8],
           [ 3,  4]])
    """
    # Handle arrays and shapes
    try:
        newshp = tuple(newshp)
    except TypeError:
        raise TypeError('`newshp` must be a sequence, got {!r}'.format(newshp))

    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError('`out` must be a `numpy.ndarray` instance, got '
                            '{!r}'.format(out))
        if out.shape != newshp:
            raise ValueError('`out` must have shape {}, got {}'
                             ''.format(newshp, out.shape))

        order = 'C' if out.flags.c_contiguous else 'F'
        arr = np.asarray(arr, dtype=out.dtype, order=order)
        if arr.ndim != out.ndim:
            raise ValueError('number of axes of `arr` and `out` do not match '
                             '({} != {})'.format(arr.ndim, out.ndim))
    else:
        arr = np.asarray(arr)
        order = 'C' if arr.flags.c_contiguous else 'F'
        out = np.empty(newshp, dtype=arr.dtype, order=order)
        if len(newshp) != arr.ndim:
            raise ValueError('number of axes of `arr` and `len(newshp)` do '
                             'not match ({} != {})'
                             ''.format(arr.ndim, len(newshp)))

    # Handle offset
    if offset is None:
        offset = [0] * out.ndim
    else:
        offset = normalized_scalar_param_list(
            offset, out.ndim, param_conv=safe_int_conv, keep_none=False)

    # Handle padding
    pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode
    if pad_mode not in _SUPPORTED_RESIZE_PAD_MODES:
        raise ValueError("`pad_mode` '{}' not understood".format(pad_mode_in))

    if (pad_mode == 'constant' and
        not np.can_cast(pad_const, out.dtype) and
        any(n_new > n_orig
            for n_orig, n_new in zip(arr.shape, out.shape))):
        raise ValueError('`pad_const` {} cannot be safely cast to the data '
                         'type {} of the output array'
                         ''.format(pad_const, out.dtype))

    # Handle direction
    direction, direction_in = str(direction).lower(), direction
    if direction not in ('forward', 'adjoint'):
        raise ValueError("`direction` '{}' not understood"
                         "".format(direction_in))

    if direction == 'adjoint' and pad_mode == 'constant' and pad_const != 0:
        raise ValueError("`pad_const` must be 0 for 'adjoint' direction, "
                         "got {}".format(pad_const))

    if direction == 'forward' and pad_mode == 'constant' and pad_const != 0:
        out.fill(pad_const)
    else:
        out.fill(0)

    # Perform the resizing
    if direction == 'forward':
        if pad_mode == 'constant':
            # Constant padding does not require the helper function
            _assign_intersection(out, arr, offset)
        else:
            # First copy the inner part and use it for padding
            _assign_intersection(out, arr, offset)
            _apply_padding(out, arr, offset, pad_mode, 'forward')
    else:
        if pad_mode == 'constant':
            # Skip the padding helper
            _assign_intersection(out, arr, offset)
        else:
            # Apply adjoint padding to a copy of the input and copy the inner
            # part when finished
            tmp = arr.copy()
            _apply_padding(tmp, out, offset, pad_mode, 'adjoint')
            _assign_intersection(out, tmp, offset)

    return out


def _intersection_slice_tuples(lhs_arr, rhs_arr, offset):
    """Return tuples to yield the intersecting part of both given arrays.

    The returned slices ``lhs_slc`` and ``rhs_slc`` are such that
    ``lhs_arr[lhs_slc]`` and ``rhs_arr[rhs_slc]`` have the same shape.
    The ``offset`` parameter determines how much is skipped/added on the
    "left" side (small indices).
    """
    lhs_slc, rhs_slc = [], []
    for istart, n_lhs, n_rhs in zip(offset, lhs_arr.shape, rhs_arr.shape):

        # Slice for the inner part in the larger array corresponding to the
        # small one, offset by the given amount
        istop = istart + min(n_lhs, n_rhs)
        inner_slc = slice(istart, istop)

        if n_lhs > n_rhs:
            # Extension
            lhs_slc.append(inner_slc)
            rhs_slc.append(slice(None))
        elif n_lhs < n_rhs:
            # Restriction
            lhs_slc.append(slice(None))
            rhs_slc.append(inner_slc)
        else:
            # Same size, so full slices for both
            lhs_slc.append(slice(None))
            rhs_slc.append(slice(None))

    return tuple(lhs_slc), tuple(rhs_slc)


def _assign_intersection(lhs_arr, rhs_arr, offset):
    """Assign the intersecting region from ``rhs_arr`` to ``lhs_arr``."""
    lhs_slc, rhs_slc = _intersection_slice_tuples(lhs_arr, rhs_arr, offset)
    lhs_arr[lhs_slc] = rhs_arr[rhs_slc]


def _padding_slices_outer(lhs_arr, rhs_arr, axis, offset):
    """Return slices into the outer array part where padding is applied.

    When padding is performed, these slices yield the outer (excess) part
    of the larger array that is to be filled with values. Slices for
    both sides of the arrays in a given ``axis`` are returned.

    The same slices are used also in the adjoint padding correction,
    however in a different way.

    See `the online documentation
    <https://odlgroup.github.io/odl/math/resizing_ops.html>`_
    on resizing operators for details.
    """
    istart_inner = offset[axis]
    istop_inner = istart_inner + min(lhs_arr.shape[axis], rhs_arr.shape[axis])
    return slice(istart_inner), slice(istop_inner, None)


def _padding_slices_inner(lhs_arr, rhs_arr, axis, offset, pad_mode):
    """Return slices into the inner array part for a given ``pad_mode``.

    When performing padding, these slices yield the values from the inner
    part of a larger array that are to be assigned to the excess part
    of the same array. Slices for both sides ("left", "right") of
    the arrays in a given ``axis`` are returned.
    """
    # Calculate the start and stop indices for the inner part
    istart_inner = offset[axis]
    n_large = max(lhs_arr.shape[axis], rhs_arr.shape[axis])
    n_small = min(lhs_arr.shape[axis], rhs_arr.shape[axis])
    istop_inner = istart_inner + n_small
    # Number of values padded to left and right
    n_pad_l = istart_inner
    n_pad_r = n_large - istop_inner

    if pad_mode == 'periodic':
        # left: n_pad_l forward, ending at istop_inner - 1
        pad_slc_l = slice(istop_inner - n_pad_l, istop_inner)
        # right: n_pad_r forward, starting at istart_inner
        pad_slc_r = slice(istart_inner, istart_inner + n_pad_r)

    elif pad_mode == 'symmetric':
        # left: n_pad_l backward, ending at istart_inner + 1
        pad_slc_l = slice(istart_inner + n_pad_l, istart_inner, -1)
        # right: n_pad_r backward, starting at istop_inner - 2
        # For the corner case that the stopping index is -1, we need to
        # replace it with None, since -1 as stopping index is equivalent
        # to the last index, which is not what we want (0 as last index).
        istop_r = istop_inner - 2 - n_pad_r
        if istop_r == -1:
            istop_r = None
        pad_slc_r = slice(istop_inner - 2, istop_r, -1)

    elif pad_mode in ('order0', 'order1'):
        # left: only the first entry, using a slice to avoid squeezing
        pad_slc_l = slice(istart_inner, istart_inner + 1)
        # right: only last entry
        pad_slc_r = slice(istop_inner - 1, istop_inner)

    else:
        # Slices are not used, returning trivial ones. The function should not
        # be used for other modes anyway.
        pad_slc_l, pad_slc_r = slice(0), slice(0)

    return pad_slc_l, pad_slc_r


def _apply_padding(lhs_arr, rhs_arr, offset, pad_mode, direction):
    """Apply padding to ``lhs_arr`` according to ``pad_mode``.

    This helper assigns the values in the excess parts (if existent)
    of ``lhs_arr`` according to the provided padding mode.

    This applies to the following values for ``pad_mode``:
    ``periodic``, ``symmetric``, ``order0``, ``order1``

    See `the online documentation
    <https://odlgroup.github.io/odl/math/resizing_ops.html>`_
    on resizing operators for details.
    """
    if pad_mode not in ('periodic', 'symmetric', 'order0', 'order1'):
        return

    full_slc = [slice(None)] * lhs_arr.ndim
    intersec_slc, _ = _intersection_slice_tuples(lhs_arr, rhs_arr, offset)

    if direction == 'forward':
        working_slc = list(intersec_slc)
    else:
        working_slc = list(full_slc)

    # TODO: order axes according to padding size for optimization (largest
    # last)? Axis strides could be important, too.
    for axis, (n_lhs, n_rhs) in enumerate(zip(lhs_arr.shape, rhs_arr.shape)):

        if n_lhs <= n_rhs:
            continue  # restriction, nothing to do

        n_pad_l = offset[axis]
        n_pad_r = n_lhs - n_rhs - n_pad_l

        # Error scenarios with illegal lengths
        if pad_mode == 'order0' and n_rhs == 0:
            raise ValueError('in axis {}: the smaller array must have size '
                             '>= 1 for order 0 padding, got 0'
                             ''.format(axis))

        if pad_mode == 'order1' and n_rhs < 2:
            raise ValueError('in axis {}: the smaller array must have size '
                             '>= 2 for order 1 padding, got {}'
                             ''.format(axis, n_rhs))

        for lr, pad_len in [('left', n_pad_l), ('right', n_pad_r)]:
            if pad_mode == 'periodic' and pad_len > n_rhs:
                raise ValueError('in axis {}: {} padding length {} exceeds '
                                 'the size {} of the smaller array; this is '
                                 'not allowed for periodic padding'
                                 ''.format(axis, lr, pad_len, n_rhs))

            elif pad_mode == 'symmetric' and pad_len >= n_rhs:
                raise ValueError('in axis {}: {} padding length {} is larger '
                                 'or equal to the size {} of the smaller '
                                 'array; this is not allowed for symmetric '
                                 'padding'
                                 ''.format(axis, lr, pad_len, n_rhs))

        # Slice tuples used to index LHS and RHS for left and right padding,
        # respectively. Since `lhs_arr` is used on both sides of the
        # assignments, full slices are used in all axes except `axis`.
        # TODO: change comment

        # TODO: use working_slc instead of full_slc
        lhs_slc_l, lhs_slc_r = list(working_slc), list(working_slc)
        rhs_slc_l, rhs_slc_r = list(working_slc), list(working_slc)

        # We're always using the outer (excess) parts involved in padding
        # on the LHS of the assignment, so we set them here.
        # TODO: change comment
        pad_slc_outer_l, pad_slc_outer_r = _padding_slices_outer(
            lhs_arr, rhs_arr, axis, offset)

        if direction == 'forward':
            lhs_slc_l[axis] = pad_slc_outer_l
            lhs_slc_r[axis] = pad_slc_outer_r
        else:
            rhs_slc_l[axis] = pad_slc_outer_l
            rhs_slc_r[axis] = pad_slc_outer_r

        if pad_mode in ('periodic', 'symmetric'):
            pad_slc_inner_l, pad_slc_inner_r = _padding_slices_inner(
                lhs_arr, rhs_arr, axis, offset, pad_mode)

            # Using `lhs_arr` on both sides of the assignment such that the
            # shapes match and the "corner" blocks are properly assigned
            # or used in the addition for the adjoint, respectively.
            if direction == 'forward':
                rhs_slc_l[axis] = pad_slc_inner_l
                rhs_slc_r[axis] = pad_slc_inner_r

                lhs_arr[tuple(lhs_slc_l)] = lhs_arr[tuple(rhs_slc_l)]
                lhs_arr[tuple(lhs_slc_r)] = lhs_arr[tuple(rhs_slc_r)]
            else:
                lhs_slc_l[axis] = pad_slc_inner_l
                lhs_slc_r[axis] = pad_slc_inner_r

                lhs_arr[tuple(lhs_slc_l)] += lhs_arr[tuple(rhs_slc_l)]
                lhs_arr[tuple(lhs_slc_r)] += lhs_arr[tuple(rhs_slc_r)]

        elif pad_mode == 'order0':
            # The `_padding_slices_inner` helper returns the slices for the
            # boundary values.
            left_slc, right_slc = _padding_slices_inner(
                lhs_arr, rhs_arr, axis, offset, pad_mode)

            if direction == 'forward':
                rhs_slc_l[axis] = left_slc
                rhs_slc_r[axis] = right_slc

                lhs_arr[tuple(lhs_slc_l)] = lhs_arr[tuple(rhs_slc_l)]
                lhs_arr[tuple(lhs_slc_r)] = lhs_arr[tuple(rhs_slc_r)]
            else:
                lhs_slc_l[axis] = left_slc
                lhs_slc_r[axis] = right_slc

                lhs_arr[tuple(lhs_slc_l)] += np.sum(
                    lhs_arr[tuple(rhs_slc_l)],
                    axis=axis, keepdims=True, dtype=lhs_arr.dtype)
                lhs_arr[tuple(lhs_slc_r)] += np.sum(
                    lhs_arr[tuple(rhs_slc_r)],
                    axis=axis, keepdims=True, dtype=lhs_arr.dtype)

        elif pad_mode == 'order1':
            # Some extra work necessary: need to compute the derivative at
            # the boundary and use that to continue with constant derivative.

            # Slice for broadcasting of a 1D array along `axis`
            bcast_slc = [None] * lhs_arr.ndim
            bcast_slc[axis] = slice(None)

            # Slices for the boundary in `axis`
            left_slc, right_slc = _padding_slices_inner(
                lhs_arr, rhs_arr, axis, offset, pad_mode)

            # Create slice tuples for indexing of the boundary values
            bdry_slc_l = list(working_slc)
            bdry_slc_l[axis] = left_slc
            bdry_slc_r = list(working_slc)
            bdry_slc_r[axis] = right_slc

            # For the slope at the boundary, we need two neighboring points.
            # We create the corresponding slices from the boundary slices.
            slope_slc_l = list(working_slc)
            slope_slc_l[axis] = slice(left_slc.start, left_slc.stop + 1)
            slope_slc_r = list(working_slc)
            slope_slc_r[axis] = slice(right_slc.start - 1, right_slc.stop)

            # The `np.arange`s, broadcast along `axis`, are used to create the
            # constant-slope continuation (forward) or to calculate the
            # first order moments (adjoint).
            arange_l = np.arange(-n_pad_l, 0,
                                 dtype=lhs_arr.dtype)[bcast_slc]
            arange_r = np.arange(1, n_pad_r + 1,
                                 dtype=lhs_arr.dtype)[bcast_slc]

            if direction == 'forward':
                # Take first order difference to get the derivative
                # along `axis`.
                slope_l = np.diff(lhs_arr[slope_slc_l], n=1, axis=axis)
                slope_r = np.diff(lhs_arr[slope_slc_r], n=1, axis=axis)

                # Finally assign the constant slope values
                lhs_arr[tuple(lhs_slc_l)] = (
                    lhs_arr[tuple(bdry_slc_l)] + arange_l * slope_l)
                lhs_arr[tuple(lhs_slc_r)] = (
                    lhs_arr[tuple(bdry_slc_r)] + arange_r * slope_r)
            else:
                # Same as in 'order0'
                lhs_arr[tuple(bdry_slc_l)] += np.sum(
                    lhs_arr[tuple(rhs_slc_l)],
                    axis=axis, keepdims=True, dtype=lhs_arr.dtype)
                lhs_arr[tuple(bdry_slc_r)] += np.sum(
                    lhs_arr[tuple(rhs_slc_r)],
                    axis=axis, keepdims=True, dtype=lhs_arr.dtype)

                # Calculate the order 1 moments
                moment1_l = np.sum(arange_l * lhs_arr[tuple(rhs_slc_l)],
                                   axis=axis, keepdims=True,
                                   dtype=lhs_arr.dtype)
                moment1_r = np.sum(arange_r * lhs_arr[tuple(rhs_slc_r)],
                                   axis=axis, keepdims=True,
                                   dtype=lhs_arr.dtype)

                # Add moment1 at the "width-2 boundary layers", with the sign
                # corresponding to the sign in the derivative calculation
                # of the forward padding.
                sign = np.array([-1, 1])[bcast_slc]
                lhs_arr[slope_slc_l] += moment1_l * sign
                lhs_arr[slope_slc_r] += moment1_r * sign

        if direction == 'forward':
            working_slc[axis] = full_slc[axis]
        else:
            working_slc[axis] = intersec_slc[axis]


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
