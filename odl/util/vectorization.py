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
from itertools import product
import numpy as np


__all__ = ('is_valid_input_array', 'is_valid_input_meshgrid',
           'meshgrid_input_order', 'vecs_from_meshgrid',
           'out_shape_from_meshgrid', 'out_shape_from_array',
           'vectorize')


def is_valid_input_array(x, d):
    """Test whether `x` is a correctly shaped array of points in R^d."""
    if not isinstance(x, np.ndarray):
        return False
    if d == 1:
        return x.ndim == 1 or x.ndim == 2 and x.shape[0] == 1
    else:
        return x.ndim == 2 and x.shape[0] == d


def is_valid_input_meshgrid(x, d):
    """Test whether `x` is a meshgrid sequence for points in R^d."""
    if isinstance(x, np.ndarray):
        return False
    if d > 1:
        try:
            np.broadcast(*x)
        except ValueError:  # cannot be broadcast
            return False

    return (len(x) == d and
            all(isinstance(xi, np.ndarray) for xi in x) and
            all(xi.ndim == d for xi in x))


def meshgrid_input_order(x):
    """Determine the ordering of a meshgrid argument."""
    # Case 1: all elements have the same shape -> non-sparse
    if all(xi.shape == x[0].shape for xi in x):
        # Contiguity check only works for meshgrid created with copy=True.
        # Otherwise, there is no way to find out the intended ordering.
        if all(xi.flags.f_contiguous for xi in x):
            return 'F'
        else:
            return 'C'
    # Case 2: sparse meshgrid, each member's shape has at most one non-one
    # entry (corner case of all ones is included)
    elif all(xi.shape.count(1) >= len(x) - 1 for xi in x):
        # Reversed ordering of dimensions in the meshgrid tuple indicates
        # 'F' ordering intention
        if all(xi.shape[-1 - i] != 1 for i, xi in enumerate(x)):
            return 'F'
        else:
            return 'C'
    else:
        return 'C'


def vecs_from_meshgrid(mg, order):
    """Get the coordinate vectors from a meshgrid (as a tuple)."""
    vecs = []
    for ax in range(len(mg)):
        select = [0] * len(mg)
        if str(order).upper() == 'F':
            select[-ax] = np.s_[:]
        else:
            select[ax] = np.s_[:]
        vecs.append(mg[ax][select])
    return tuple(vecs)


def out_shape_from_meshgrid(mg):
    """Get the broadcast output shape from a meshgrid."""
    if len(mg) == 1:
        return (len(mg[0]),)
    else:
        return np.broadcast(*mg).shape


def out_shape_from_array(arr):
    """Get the output shape from an array."""
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)


def vectorize(dtype=None, outarg='none'):
    """Vectorization decorator for our input parameter pattern.

    The wrapped function must be callable with one positional
    parameter. Keyword arguments are passed through, hence positional
    arguments with defaults can either be left out or passed by keyword,
    but not by position.

    Parameters
    ----------
    dtype : `type` or `str`, optional
        Data type of the output array. Needs to be understood by the
        `numpy.dtype` function. If not provided, a "lazy" vectorization
        is performed, meaning that the results are collected in a
        list instead of an array.
    outarg : {'none', 'positional', 'optional'}
        Type of the output argument of the decorated function for
        in-place evaluation

        'none': No output parameter. This is the default.
        Resulting argspec: ``func(x, **kwargs)``
        Returns: the new array

        'positional': Required argument ``out`` at second position.
        Resulting argspec: ``func(x, out=None, **kwargs)``
        Returns: ``out``

        'optional': optional argument ``out`` with default `None`.
        Resulting argspec: ``func(x, out=None, **kwargs)``
        Returns: ``out`` if it is not `None` otherwise a new array

    Note
    ----
    For ``outarg`` not equal to 'none', the decorated function returns
    the array given as ``out`` argument if it is not `None`.

    Examples
    --------
    Vectorize a step function in the first variable:

    >>> @vectorize(dtype=float)
    ... def step(x):
    ...     return 0 if x[0] <= 0 else 1

    This corresponds to (but is much slower than)

    >>> import numpy as np
    >>> def step_vec(x):
    ...     x0, x1 = x
    ...     # np.broadcast is your friend to determine the output shape
    ...     out = np.zeros(np.broadcast(x0, x1).shape, dtype=x0.dtype)
    ...     idcs = np.where(x0 > 0)
    ...     # Need to throw away the indices from the empty dimensions
    ...     idcs = idcs[0] if len(idcs) > 1 else idcs
    ...     out[idcs] = 1
    ...     return out

    Both versions work for arrays and meshgrids:

    >>> x = np.linspace(-5, 13, 10, dtype=float).reshape((2, 5))
    >>> x  # array representing 5 points in 2d
    array([[ -5.,  -3.,  -1.,   1.,   3.],
           [  5.,   7.,   9.,  11.,  13.]])
    >>> np.array_equal(step(x), step_vec(x))
    True

    >>> x = y = np.linspace(-1, 2, 5)
    >>> mg_sparse = np.meshgrid(x, y, indexing='ij', sparse=True)
    >>> np.array_equal(step(mg_sparse), step_vec(mg_sparse))
    True
    >>> mg_dense = np.meshgrid(x, y, indexing='ij', sparse=False)
    >>> np.array_equal(step(mg_dense), step_vec(mg_dense))
    True

    With output parameter:

    >>> @vectorize(dtype=float, outarg='positional')
    ... def step(x):
    ...     return 0 if x[0] <= 0 else 1
    >>> x = np.linspace(-5, 13, 10, dtype=float).reshape((2, 5))
    >>> out = np.empty(5, dtype=float)
    >>> step(x, out)  # returns out
    array([ 0.,  0.,  0.,  1.,  1.])
    """
    def vect_decorator(func):

        def _vect_wrapper_array(x, out, **kwargs):
            # Assume that x is an ndarray
            if out is None:
                out_shape = out_shape_from_array(x)
                if dtype is None:
                    out = [0] * out_shape[0]
                else:
                    out = np.empty(out_shape, dtype=dtype)

            for i, pt in enumerate(x.T):
                out[i] = func(pt, **kwargs)
            return out

        def _vect_wrapper_meshgrid(x, out, **kwargs):
            if out is None:
                out_shape = out_shape_from_meshgrid(x)
                if dtype is None:
                    out = [0] * out_shape[0]
                else:
                    out = np.empty(out_shape, dtype=dtype)

            order = meshgrid_input_order(x)
            vecs = vecs_from_meshgrid(x, order=order)
            for i, pt in enumerate(product(*vecs)):
                out.flat[i] = func(pt, **kwargs)
            return out

        def _vect_wrapper(x, out, **kwargs):
            # Find out dimension first
            if isinstance(x, np.ndarray):  # array
                if x.ndim == 1:
                    dim = 1
                elif x.ndim == 2:
                    dim = len(x)
                else:
                    raise ValueError('only 1- or 2-dimensional arrays '
                                     'supported.')
            else:  # meshgrid
                dim = len(x)

            if is_valid_input_array(x, dim):
                return _vect_wrapper_array(x, out, **kwargs)
            elif is_valid_input_meshgrid(x, dim):
                return _vect_wrapper_meshgrid(x, out, **kwargs)
            else:
                raise TypeError('invalid vectorized input type.')

        @wraps(func)
        def vect_wrapper_no_out(x, **kwargs):
            if 'out' in kwargs:
                raise TypeError("{}() got an unexpected keyword 'out'."
                                "".format(func.__name__))
            return _vect_wrapper(x, None, **kwargs)

        @wraps(func)
        def vect_wrapper_pos_out(x, out, **kwargs):
            return _vect_wrapper(x, out, **kwargs)

        @wraps(func)
        def vect_wrapper_opt_out(x, out=None, **kwargs):
            return _vect_wrapper(x, out, **kwargs)

        outarg_ = str(outarg).lower()

        if outarg_ == 'none':
            return vect_wrapper_no_out
        elif outarg_ == 'positional':
            return vect_wrapper_pos_out
        elif outarg_ == 'optional':
            return vect_wrapper_opt_out
        else:
            raise ValueError('output arg type {!r} not understood.'
                             ''.format(outarg))
    return vect_decorator


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
