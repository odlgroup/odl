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

# External module imports


__all__ = ('apply_on_boundary',)


def apply_on_boundary(array, func, only_once=True, which_boundaries=None,
                      axis_order=None):
    """Apply a function of the boundary of an n-dimensional array.

    Parameters
    ----------
    array : `numpy.ndarray`
        Modify the boundary of this array in place
    func : callable or sequence
        If a single function is given, assign
        ``array[slice] = func(array[slice])`` on the boundary slices,
        e.g. use ``lamda x: x / 2`` to divide values by 2.
        A sequence of functions is applied per axis separately. It
        must have length ``array.ndim`` and may consist of one function
        or a 2-tuple of functions per axis.
    only_once : `bool`, optional
        If `True`, ensure that each boundary point appears in exactly
        one slice.
    which_boundaries : sequence, optional
        If provided, this sequence determines per axis whether to
        apply the function at the boundaries in each axis. The entry
        in each axis may consist in a single `bool` or a 2-tuple of
        `bool`. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``array.ndim``. `None` is interpreted as
        'all boundaries'.
    axis_order : sequence of `int`
        Permutation of ``range(array.ndim)`` defining the order in which
        to process the axes. If combined with ``only_once`` and a
        function list, this determines which function is evaluated in
        the points that are potentially processed multiple times.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.ones((3, 3))
    >>> apply_on_boundary(arr, lambda x: x / 2)
    >>> arr
    array([[ 0.5,  0.5,  0.5],
           [ 0.5,  1. ,  0.5],
           [ 0.5,  0.5,  0.5]])
    >>>
    >>> arr = np.ones((3, 3))
    >>> apply_on_boundary(arr, lambda x: x / 2, only_once=False)
    >>> arr
    array([[ 0.25,  0.5 ,  0.25],
           [ 0.5 ,  1.  ,  0.5 ],
           [ 0.25,  0.5 ,  0.25]])
    >>> arr = np.ones((3, 3))
    >>> apply_on_boundary(arr, lambda x: x / 2, only_once=True,
    ...                   which_boundaries=((True, False), True))
    >>> arr
    array([[ 0.5,  0.5,  0.5],
           [ 0.5,  1. ,  0.5],
           [ 0.5,  1. ,  0.5]])
    """
    if callable(func):
        func = [func] * array.ndim
    elif len(func) != array.ndim:
        raise ValueError('sequence of functions has length {}, expected {}.'
                         ''.format(len(func), array.ndim))

    if which_boundaries is None:
        which_boundaries = ([(True, True)] * array.ndim)
    elif len(which_boundaries) != array.ndim:
        raise ValueError('which_boundaries has length {}, expected {}.'
                         ''.format(len(which_boundaries), array.ndim))

    if axis_order is None:
        axis_order = list(range(array.ndim))
    elif len(axis_order) != array.ndim:
        raise ValueError('axis_order has length {}, expected {}.'
                         ''.format(len(axis_order), array.ndim))

    slices = [slice(None)] * array.ndim
    for ax, function, which in zip(axis_order, func, which_boundaries):
        # Make slices selecting left and right
        if only_once:
            slc_l = list(slices)  # Make a copy; copy() exists in Py3 only
            slc_r = list(slices)
        else:
            slc_l = [slice(None)] * array.ndim
            slc_r = [slice(None)] * array.ndim

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

        if mod_left:
            array[slc_l] = func_l(array[slc_l])
            start = 1
        else:
            start = None

        if mod_right:
            array[slc_r] = func_r(array[slc_r])
            end = -1
        else:
            end = None

        slices[ax] = slice(start, end)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
