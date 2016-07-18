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


_SUPPORTED_PAD_MODES = ('constant', 'symmetric', 'periodic',
                        'order0', 'order1')


def apply_on_boundary(array, func, only_once=True, which_boundaries=None,
                      axis_order=None, out=None):
    """Apply a function of the boundary of an n-dimensional array.

    All other values are preserved as is.

    Parameters
    ----------
    array : array-like
        Modify the boundary of this array
    func : `callable` or `sequence`
        If a single function is given, assign
        ``array[slice] = func(array[slice])`` on the boundary slices,
        e.g. use ``lamda x: x / 2`` to divide values by 2.
        A sequence of functions is applied per axis separately. It
        must have length ``array.ndim`` and may consist of one function
        or a 2-tuple of functions per axis.
        `None` entries in a sequence cause the axis (side) to be
        skipped.
    only_once : `bool`, optional
        If `True`, ensure that each boundary point appears in exactly
        one slice. If ``func`` is a list of functions, the
        ``axis_order`` determines which functions are applied to nodes
        which appear in multiple slices, according to the principle
        "first-come, first-served".
    which_boundaries : `sequence`, optional
        If provided, this sequence determines per axis whether to
        apply the function at the boundaries in each axis. The entry
        in each axis may consist in a single `bool` or a 2-tuple of
        `bool`. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``array.ndim``. `None` is interpreted as
        'all boundaries'.
    axis_order : `sequence` of `int`, optional
        Permutation of ``range(array.ndim)`` defining the order in which
        to process the axes. If combined with ``only_once`` and a
        function list, this determines which function is evaluated in
        the points that are potentially processed multiple times.
    out : `numpy.ndarray`, optional
        Location in which to store the result, can be the same as ``array``.
        Default: copy of ``array``

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.ones((3, 3))
    >>> apply_on_boundary(arr, lambda x: x / 2)
    array([[ 0.5,  0.5,  0.5],
           [ 0.5,  1. ,  0.5],
           [ 0.5,  0.5,  0.5]])

    If called with ``only_once=False``, applies function repeatedly

    >>> apply_on_boundary(arr, lambda x: x / 2, only_once=False)
    array([[ 0.25,  0.5 ,  0.25],
           [ 0.5 ,  1.  ,  0.5 ],
           [ 0.25,  0.5 ,  0.25]])

    >>> apply_on_boundary(arr, lambda x: x / 2, only_once=True,
    ...                   which_boundaries=((True, False), True))
    array([[ 0.5,  0.5,  0.5],
           [ 0.5,  1. ,  0.5],
           [ 0.5,  1. ,  0.5]])

    Also accepts out parameter:

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
    onedim_arrs : sequence of array-like
        One-dimensional arrays to be multiplied with ``ndarr``. The
        sequence may not be longer than ``ndarr.ndim``.
    axes : sequence of `int`, optional
        Take the 1d transform along these axes. `None` corresponds to
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


def resize_array(arr, newshp, frac_left=None, num_left=None,
                 pad_mode='constant', pad_const=0, direction='forward',
                 out=None):
    """Return the resized version of ``arr`` with shape ``newshp``.

    In axes where ``newshp > arr.shape``, padding is applied according
    to the supplied options.
    Where ``newshp < arr.shape``, the array is cropped to the new
    size.

    Parameters
    ----------
    arr : array-like
        Array to be resized.
    newshp : sequence of int
        Desired shape of the output array.
    frac_left : float or sequence of float, optional
        Number between 0 and 1 (inclusive) that determines which fraction
        of the addition/removal is performed on the "left" side. If
        ``frac_left`` times the shape difference is not integer, the
        number of entries added to the left is gained by rounding to the
        nearest integer. The default ``None`` is equivalent to 0.5.
        This option can be specified per axis with a sequence.
        Cannot be combined with ``num_left``.
    num_left : sequence of int, optional
        Specifies how many entries are added to/removed from the left
        side of ``arr``.
        Cannot be combined with ``frac_left``.
    pad_mode : str, optional
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

        Note that for ``'symmetric'`` and ``'periodic'`` padding, the
        number of added values on each side of the array cannot exceed
        the original size.

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
    The left/right distribution of removed entries is ananogous to
    the case of array enlargement, see below.

    >>> from odl.util.numerics import resize_array
    >>> resize_array([1, 2, 3], (1,))
    array([2])
    >>> resize_array([1, 2, 3], (2,))
    array([1, 2])

    When enlarging, zero-padding is applied by default, and half of the
    zeros are added to each side (with preference for the left side in
    case of ambiguity):

    >>> resize_array([1, 2, 3], (6,))
    array([0, 0, 1, 2, 3, 0])
    >>> resize_array([1, 2, 3], (7,))
    array([0, 0, 1, 2, 3, 0, 0])

    One of the ``frac_left`` and ``num_left`` parameters can be
    supplied to change the default distribution of the extra values:

    >>> resize_array([1, 2, 3], (7,), frac_left=0.25)
    array([0, 1, 2, 3, 0, 0, 0])
    >>> resize_array([1, 2, 3], (7,), num_left=1)
    array([0, 1, 2, 3, 0, 0, 0])

    The padding constant can be changed, as well as the padding
    mode:

    >>> resize_array([1, 2, 3], (7,), pad_const=-1)
    array([-1, -1,  1,  2,  3, -1, -1])
    >>> resize_array([1, 2, 3], (7,), pad_mode='periodic')
    array([2, 3, 1, 2, 3, 1, 2])
    >>> resize_array([1, 2, 3], (7,), pad_mode='symmetric')
    array([3, 2, 1, 2, 3, 2, 1])
    >>> resize_array([1, 2, 3], (7,), pad_mode='order0')
    array([1, 1, 1, 2, 3, 3, 3])
    >>> resize_array([1, 2, 3], (7,), pad_mode='order1')
    array([-1,  0,  1,  2,  3,  4,  5])

    Everything works for arbitrary number of dimensions:

    >>> # Take the middle two columns and extend rows symmetrically
    >>> resize_array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12]], (5, 2), pad_mode='symmetric')
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
    ...              frac_left=[0, 1])
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

        arr = np.asarray(arr, dtype=out.dtype)
        if arr.ndim != out.ndim:
            raise ValueError('number of axes of `arr` and `out` do not match '
                             '({} != {})'.format(arr.ndim, out.ndim))
    else:
        arr = np.asarray(arr)
        out = np.empty(newshp, dtype=arr.dtype)
        if len(newshp) != arr.ndim:
            raise ValueError('number of axes of `arr` and `len(newshp)` do '
                             'not match ({} != {})'
                             ''.format(arr.ndim, len(newshp)))

    # Handle frac_left and num_left parameters
    if frac_left is not None and num_left is not None:
        raise ValueError('the options `frac_left` and `num_left` cannot be '
                         'combined')
    if num_left is None:
        # Compute num_left from frac_left
        if frac_left is None:
            frac_left = 0.5
        frac_left, frac_left_in = normalized_scalar_param_list(
            frac_left, out.ndim, param_conv=float, keep_none=False,
            return_nonconv=True)
        num_left = [round(frac_l * abs(n_orig - n_new))
                    for frac_l, n_orig, n_new in zip(frac_left, arr.shape,
                                                     out.shape)]
        for i, (fl, fl_in) in enumerate(zip(frac_left, frac_left_in)):
            if not 0.0 <= fl <= 1.0:
                raise ValueError('in axis {}: `frac_left` must lie between 0 '
                                 'and 1, got {}'.format(i, fl_in))
    else:
        num_left = normalized_scalar_param_list(
            num_left, out.ndim, param_conv=safe_int_conv, keep_none=False)

    # Handle padding
    pad_mode, pad_mode_in = str(pad_mode), pad_mode
    if pad_mode not in _SUPPORTED_PAD_MODES:
        raise ValueError("`pad_mode` '{}' not understood".format(pad_mode_in))

    if (pad_mode == 'constant' and
        not np.can_cast(pad_const, out.dtype) and
        any(n_new > n_orig
            for n_orig, n_new in zip(arr.shape, out.shape))):
        raise ValueError('`pad_const` {} cannot be safely cast to the data '
                         'type {} of the output array'
                         ''.format(pad_const, out.dtype))

    # Calculate the slices for the inner and outer parts
    arr_slc, out_slc, pad_l_slc, pad_r_slc = [], [], [], []
    for i, (n_orig, n_new, num_l) in enumerate(zip(arr.shape, out.shape,
                                                   num_left)):
        if n_new < n_orig:
            # Simple case: remove according to num_left
            n_remove = n_orig - n_new
            istart = num_l
            istop = n_orig - (n_remove - istart)
            arr_slc.append(slice(istart, istop))
            out_slc.append(slice(None))

            # Make trivial entries for padding slice lists
            pad_l_slc.append(slice(0))
            pad_r_slc.append(slice(0))

        elif n_new > n_orig:
            # Padding case: calculate start and stop indices for the
            # bigger new array, together with the padding slices
            n_add = n_new - n_orig
            istart = num_l
            istop = n_new - (n_add - istart)

            n_pad_l = len(range(istart))
            n_pad_r = len(range(istop, n_new))

            # Handle some error scenarios with illegal lengths
            if pad_mode == 'order0' and n_orig == 0:
                raise ValueError('in axis {}: need at least 1 value for '
                                 'order 0 padding, got 0'
                                 ''.format(i))

            if pad_mode == 'order1' and n_orig == 1:
                raise ValueError('in axis {}: need at least 2 values for '
                                 'order 1 padding, got 1'
                                 ''.format(i))

            for lr, pad_len in [('left', n_pad_l), ('right', n_pad_r)]:
                if pad_mode == 'periodic' and pad_len > n_orig:
                    raise ValueError('in axis {}: {} padding length {} '
                                     'exceeds original array size {}; this is '
                                     'not allowed for periodic padding'
                                     ''.format(i, lr, pad_len, n_orig))

                elif pad_mode == 'symmetric' and pad_len >= n_orig:
                    raise ValueError('in axis {}: {} padding length {} '
                                     'larger or equal to the original array '
                                     'size {}; this is not allowed for '
                                     'symmetric padding'
                                     ''.format(i, lr, pad_len, n_orig))

            arr_slc.append(slice(None))
            out_slc.append(slice(istart, istop))
            pad_l_slc.append(slice(istart))
            pad_r_slc.append(slice(istop, None))

        else:
            arr_slc.append(slice(None))
            out_slc.append(slice(None))
            pad_l_slc.append(slice(0))
            pad_r_slc.append(slice(0))

    # Set the "inner" part
    out[tuple(out_slc)] = arr[tuple(arr_slc)]

    # Perform the padding
    for i, (slc_l, slc_r) in enumerate(zip(pad_l_slc, pad_r_slc)):
        _apply_padding(out, slc_l, slc_r, i, pad_mode, pad_const)

    return out


def _apply_padding(arr, slc_l, slc_r, axis, pad_mode, const):
    """Apply padding of given mode to ``arr``.

    ``slc[l,r]`` must be slices with step size 1.

    ``const`` is only used for constant padding.

    The following assignment is performed, depending on ``par_mode``
    (semantically, the slices appear in the location indexed by
    ``axis``; ``slc`` is either ``slc_l`` or ``slc_r``):

    ``'constant'``: ``arr[..., slc, ...] = const``.

    ``'symmetric'``: ``arr[..., slc, ...] = arr[..., sym_slc, ...]``.
    Here, ``sym_slc`` is the symmetric counterpart to ``slc``.

    ``'periodic'``: ``arr[..., slc, ...] = arr[..., per_slc, ...]``.
    Here, ``per_slc`` is the periodic counterpart to ``slc``.

    ``'order0'``: ``arr[..., slc, ...] = arr[..., bdry_slc, ...]``.
    Here, ``bdry_idx`` is a slice for the boundary in ``axis``.

    ``'order1'``: ``arr[..., slc, ...] = val[..., bdry_slc, ...] +
    lin_arr``.
    Here, ``bdry_idx`` is as above, and ``lin_arr = h * arange``
    is the array with constant slope ``h`` along ``axis``.
    """
    ax_len = arr.shape[axis]
    ndim = arr.ndim

    def slice_tuple(length, idx, at_idx, at_other_idcs):
        """Return tuple for slicing.

        Create a tuple of given ``length`` with entries ``at_other_idcs``
        except at index ``idx``, where the entry is ``at_idx``.
        """
        return ((at_other_idcs,) * idx + (at_idx,) +
                (at_other_idcs,) * (length - 1 - idx))

    # For indexing on the left hand side of the assignment, use the slices
    # in `axis` and everything in the other axes.
    arr_indcs_l = slice_tuple(ndim, axis, slc_l, slice(None))
    arr_indcs_r = slice_tuple(ndim, axis, slc_r, slice(None))

    start_l, stop_l, _ = slc_l.indices(ax_len)
    n_l = stop_l - start_l
    start_r, stop_r, _ = slc_r.indices(ax_len)
    n_r = stop_r - start_r

    # Symmetric left: n_l backward, ending at stop_l + 1
    sym_slc_l = slice(stop_l + n_l, stop_l, -1)
    # Symmetric right: n_r backward, starting at start_r - 2
    sym_r_stop = start_r - 2 - n_r
    if sym_r_stop == -1:  # This will not yield the correct slice (0 last)
        sym_r_stop = None  # Fix
    sym_slc_r = slice(start_r - 2, sym_r_stop, -1)

    # Periodic left: n_l forward, ending at start_r - 1
    per_slc_l = slice(start_r - n_l, start_r)
    # Periodic right: n_r forward, starting at stop_l
    per_slc_r = slice(stop_l, stop_l + n_r)

    # Boundary indices
    bdry_idx_l = stop_l
    bdry_idx_r = start_r - 1

    if pad_mode == 'constant':
        # Just assign here
        arr[arr_indcs_l] = const
        arr[arr_indcs_r] = const

    elif pad_mode == 'symmetric':
        # Build indexing tuples with the sym_slc slices in position `axis`
        sym_indcs_l = slice_tuple(ndim, axis, sym_slc_l, slice(None))
        sym_indcs_r = slice_tuple(ndim, axis, sym_slc_r, slice(None))
        l_view = arr[sym_indcs_l]
        r_view = arr[sym_indcs_r]
        if l_view.shape[axis]:
            arr[arr_indcs_l] = l_view
        if r_view.shape[axis]:
            arr[arr_indcs_r] = r_view

    elif pad_mode == 'periodic':
        # Similar to symmetric
        per_indcs_l = slice_tuple(ndim, axis, per_slc_l, slice(None))
        per_indcs_r = slice_tuple(ndim, axis, per_slc_r, slice(None))
        arr[arr_indcs_l] = arr[per_indcs_l]
        arr[arr_indcs_r] = arr[per_indcs_r]

    elif pad_mode == 'order0':
        # Avoid squeezing by using a length-1 slice. This allows broadcasting
        # along `axis` in the final assignments.
        bdry_indcs_l = slice_tuple(ndim, axis,
                                   slice(bdry_idx_l, bdry_idx_l + 1),
                                   slice(None))
        bdry_indcs_r = slice_tuple(ndim, axis,
                                   slice(bdry_idx_r, bdry_idx_r + 1),
                                   slice(None))
        arr[arr_indcs_l] = arr[bdry_indcs_l]
        arr[arr_indcs_r] = arr[bdry_indcs_r]

    elif pad_mode == 'order1':
        # Create arrays of slopes along `axis` using two boundary indices.
        # This is done by slicing into `arr` with a length-2 slice in `axis`
        # and taking the difference in the same axis.
        bdry_indcs_l = slice_tuple(ndim, axis,
                                   slice(bdry_idx_l, bdry_idx_l + 1),
                                   slice(None))
        slope_indcs_l = slice_tuple(ndim, axis,
                                    slice(bdry_idx_l, bdry_idx_l + 2),
                                    slice(None))
        slope_arr_l = np.diff(arr[slope_indcs_l], axis=axis)

        bdry_indcs_r = slice_tuple(ndim, axis,
                                   slice(bdry_idx_r, bdry_idx_r + 1),
                                   slice(None))
        slope_indcs_r = slice_tuple(ndim, axis,
                                    slice(bdry_idx_r - 1, bdry_idx_r + 1),
                                    slice(None))
        slope_arr_r = np.diff(arr[slope_indcs_r], axis=axis)

        # Create linear index arrays, reversed for the left boundary since
        # we computed the "forward" slope. The arrays are used to create the
        # "sloped lines" along `axis`.
        arange_l = -np.arange(n_l, 0, -1)
        arange_r = np.arange(1, n_r + 1)

        # Broadcast the 1D arange arrays in all axes except `axis`
        arange_indcs = slice_tuple(ndim, axis, slice(None), None)

        arr[arr_indcs_l] = (arr[bdry_indcs_l] +
                            slope_arr_l * arange_l[arange_indcs])
        arr[arr_indcs_r] = (arr[bdry_indcs_r] +
                            slope_arr_r * arange_r[arange_indcs])


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
