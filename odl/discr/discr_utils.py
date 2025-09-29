# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Helpers for discretization-related functionality.

Most functions deal with interpolation of arrays, sampling of functions and
providing a single interface for the sampler by wrapping functions or
arrays of functions appropriately.
"""

from __future__ import absolute_import, division, print_function

import inspect
import sys
from builtins import object
from functools import partial
from itertools import product
from warnings import warn

from typing import Callable, List, Tuple
from odl.set.domain import IntervalProd

import numpy as np

from odl.array_API_support import asarray, lookup_array_backend, ArrayBackend, get_array_and_backend
from odl.array_API_support.utils import is_array_supported

from odl.util.npy_compat import AVOID_UNNECESSARY_COPY

from odl.util.dtype_utils import _universal_dtype_identifier, is_floating_dtype
from odl.util import (
    dtype_repr, is_real_dtype, is_string, is_valid_input_array,
    is_valid_input_meshgrid, out_shape_from_array, out_shape_from_meshgrid,
    writable_array)

__all__ = (
    'point_collocation',
    'nearest_interpolator',
    'linear_interpolator',
    'per_axis_interpolator',
    'sampling_function',
)

SUPPORTED_INTERP = ['nearest', 'linear']


def point_collocation(func, points, out=None, **kwargs):
    """Sample a function on a grid of points.

    This function represents the simplest way of discretizing a function.
    It does little more than calling the function on a single point or a
    set of points, and returning the result.

    Parameters
    ----------
    func : callable
        Function to be sampled. It is expected to work with single points,
        meshgrids and point arrays, and to support an optional ``out``
        argument.
        Usually, ``func`` is the return value of `make_func_for_sampling`.
    points : point, meshgrid or array of points
        The point(s) where to sample.
    out : numpy.ndarray, optional
        Array to which the result should be written.
    kwargs :
        Additional arguments that are passed on to ``func``.

    Returns
    -------
    out : numpy.ndarray
        Array holding the values of ``func`` at ``points``. If ``out`` was
        given, the returned object is a reference to it.

    Examples
    --------
    Sample a 1D function:

    >>> from odl.discr.grid import sparse_meshgrid
    >>> domain = odl.IntervalProd(0, 5)
    >>> func = sampling_function(lambda x: x ** 2, domain)
    >>> mesh = sparse_meshgrid([1, 2, 3])
    >>> point_collocation(func, mesh)
    array([ 1.,  4.,  9.])

    By default, inputs are checked against ``domain`` to be in bounds. This
    can be switched off by passing ``bounds_check=False``:

    >>> mesh = sparse_meshgrid([-1, 0, 4])
    >>> point_collocation(func, mesh, bounds_check=False)
    array([  1.,   0.,  16.])

    In two or more dimensions, the function to be sampled can be written as
    if its arguments were the components of a point, and an implicit loop
    around the call would iterate over all points:

    >>> domain = odl.IntervalProd([0, 0], [5, 5])
    >>> xs = [1, 2]
    >>> ys = [3, 4, 5]
    >>> mesh = sparse_meshgrid(xs, ys)
    >>> func = sampling_function(lambda x: x[0] - x[1], domain)
    >>> point_collocation(func, mesh)
    array([[-2., -3., -4.],
           [-1., -2., -3.]])

    It is possible to return results that require broadcasting, and to use
    *optional* function parameters:

    >>> def f(x, c=0):
    ...     return x[0] + c
    >>> func = sampling_function(f, domain)
    >>> point_collocation(func, mesh)  # uses default c=0
    array([[ 1.,  1.,  1.],
           [ 2.,  2.,  2.]])
    >>> point_collocation(func, mesh, c=2)
    array([[ 3.,  3.,  3.],
           [ 4.,  4.,  4.]])

    The ``point_collocation`` function also supports vector- and tensor-valued
    functions. They can be given either as a single function returning an
    array-like of results, or as an array-like of member functions:

    >>> domain = odl.IntervalProd([0, 0], [5, 5])
    >>> xs = [1, 2]
    >>> ys = [3, 4]
    >>> mesh = sparse_meshgrid(xs, ys)
    >>> def vec_valued(x):
    ...     return (x[0] - 1., 0., x[0] + x[1])  # broadcasting
    >>> # For a function with several output components, we must specify the
    >>> # shape explicitly in the `out_dtype` parameter
    >>> func1 = sampling_function(
    ...     vec_valued, domain, out_dtype=(float, (3,))
    ... )
    >>> point_collocation(func1, mesh)
    array([[[ 0.,  0.],
            [ 1.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.],
            [ 0.,  0.]],
    <BLANKLINE>
           [[ 4.,  5.],
            [ 5.,  6.]]])
    >>> list_of_funcs = [  # equivalent to `vec_valued`
    ...     lambda x: x[0] - 1,
    ...     0,                   # constants are allowed
    ...     lambda x: x[0] + x[1]
    ... ]
    >>> # For an array of functions, the output shape can be inferred
    >>> func2 = sampling_function(list_of_funcs, domain)
    >>> point_collocation(func2, mesh)
    array([[[ 0.,  0.],
            [ 1.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.],
            [ 0.,  0.]],
    <BLANKLINE>
           [[ 4.,  5.],
            [ 5.,  6.]]])

    Notes
    -----
    This function expects its input functions to be written in a
    vectorization-conforming manner to ensure fast evaluation.
    See the `ODL vectorization guide`_ for a detailed introduction.

    See Also
    --------
    make_func_for_sampling : wrap a function
    odl.discr.grid.RectGrid.meshgrid
    numpy.meshgrid

    References
    ----------
    .. _ODL vectorization guide:
       https://odlgroup.github.io/odl/guide/in_depth/vectorization_guide.html
    """
    if out is None:
        out = func(points, **kwargs)
    else:
        func(points, out=out, **kwargs)
    return out


def _normalize_interp(interp, ndim):
    """Turn interpolation type into a tuple with one entry per axis."""
    interp_in = interp

    if is_string(interp):
        interp = str(interp).lower()
        interp_byaxis = (interp,) * ndim
    else:
        interp_byaxis = tuple(str(itp).lower() for itp in interp)
        if len(interp_byaxis) != ndim:
            raise ValueError(
                'length of `interp` ({}) does not match number of axes ({})'
                ''.format(len(interp_byaxis), ndim)
            )

    if not all(
        interp in SUPPORTED_INTERP for interp in interp_byaxis
    ):
        raise ValueError(
            'invalid `interp` {!r}; supported are: {}'
            ''.format(interp_in, SUPPORTED_INTERP)
        )

    return interp_byaxis


def _check_interp_input(x, f):
    """Return transformed ``x``, its input type and whether it's scalar.

    On bad input, raise ``ValueError``.
    """
    errmsg_1d = (
        'bad input: expected scalar, array-like of shape (1,), (n,) or '
        '(1, n), or a meshgrid of length 1; got {!r}'
        ''.format(x)
    )

    errmsg_nd = (
        'bad input: expected scalar, array-like of shape ({0},) or '
        '({0}, n), or a meshgrid of length {0}; got {1!r}'
        ''.format(f.ndim, x)
    )

    if is_valid_input_meshgrid(x, f.ndim):
        x_is_scalar = False
        x_type = 'meshgrid'
    else:
        x = np.asarray(x)
        if f.ndim == 1 and x.shape == ():
            x_is_scalar = True
            x = x.reshape((1, 1))
        elif f.ndim == 1 and x.ndim == 1:
            x_is_scalar = False
            x = x.reshape((1, x.size))
        elif f.ndim > 1 and x.shape == (f.ndim,):
            x_is_scalar = True
            x = x.reshape((f.ndim, 1))
        else:
            x_is_scalar = False

        if not is_valid_input_array(x, f.ndim):
            errmsg = errmsg_1d if f.ndim == 1 else errmsg_nd
            raise ValueError(errmsg)

        x_type = 'array'

    return x, x_type, x_is_scalar


def nearest_interpolator(f, coord_vecs):
    """Return the nearest neighbor interpolator for discrete values.

    Given points ``x[1] < x[2] < ... < x[N]``, and function values
    ``f[1], ..., f[N]``, nearest neighbor interpolation at ``x`` is defined
    as ::

        I(x) = f[j]

    with ``j`` such that ``|x - x[j]|`` is minimal.
    The ambiguity at the midpoints is resolved by preferring the right
    neighbor. In higher dimensions, this principle is applied per axis.
    The returned interpolator is the piecewise constant function ``x -> I(x)``.

    Parameters
    ----------
    f : numpy.ndarray
        Function values that should be interpolated.
    coord_vecs : sequence of numpy.ndarray
        Coordinate vectors of the rectangular grid on which interpolation
        should be based. They must be sorted in ascending order. Usually
        they are obtained as ``grid.coord_vectors`` from a `RectGrid`.

    Returns
    -------
    interpolator : function
        Python function that will interpolate the given values when called
        with a point or multiple points (vectorized).

    Examples
    --------
    We interpolate a 1d function. If called with a single point, the
    interpolator returns a single value, and with multiple points at once,
    an array of values is returned:

    >>> part = odl.uniform_partition(0, 2, 5)
    >>> part.coord_vectors  # grid points
    (array([ 0.2,  0.6,  1. ,  1.4,  1.8]),)
    >>> f = [1, 2, 3, 4, 5]
    >>> interpolator = nearest_interpolator(f, part.coord_vectors)
    >>> interpolator(0.3)  # closest to 0.2 -> value 1
    1
    >>> interpolator([0.6, 1.3, 1.9])  # closest to [0.6, 1.4, 1.8]
    array([2, 4, 5])

    In 2 dimensions, we can either use a (transposed) list of points or
    a meshgrid:

    >>> part = odl.uniform_partition([0, 0], [1, 5], shape=(2, 4))
    >>> part.coord_vectors  # grid points
    (array([ 0.25,  0.75]), array([ 0.625,  1.875,  3.125,  4.375]))
    >>> f = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8]],
    ...              dtype=float)
    >>> interpolator = nearest_interpolator(f, part.coord_vectors)
    >>> interpolator([1, 1])  # single point
    5.0
    >>> x = np.array([[0.5, 2.0],
    ...               [0.0, 4.5],
    ...               [0.0, 3.0]]).T  # 3 points at once
    >>> interpolator(x)
    array([ 6.,  4.,  3.])
    >>> from odl.discr.grid import sparse_meshgrid
    >>> mesh = sparse_meshgrid([0.0, 0.4, 1.0], [1.5, 3.5])
    >>> interpolator(mesh)  # 3x2 grid of points
    array([[ 2.,  3.],
           [ 2.,  3.],
           [ 6.,  7.]])

    With nearest neighbor interpolation, we can also use non-scalar data
    types like strings:

    >>> part = odl.uniform_partition(0, 3, 6)
    >>> part.coord_vectors  # grid points
    (array([ 0.25,  0.75,  1.25,  1.75,  2.25,  2.75]),)
    >>> f = ['s', 't', 'r', 'i', 'n', 'g']
    >>> interpolator = nearest_interpolator(f, part.coord_vectors)
    >>> print(interpolator(0.9))
    t

    See Also
    --------
    linear_interpolator : (bi-/tri-/...)linear interpolation
    per_axis_interpolator : potentially different interpolation in each axis

    Notes
    -----
    - **Important:** if called on a point array, the points are
      assumed to be sorted in ascending order in each dimension
      for efficiency reasons.
    - Nearest neighbor interpolation is the only scheme which works
      with data of non-numeric data type since it does not involve any
      arithmetic operations on the values, in contrast to other
      interpolation methods.
    """
    f = np.asarray(f)

    # TODO(kohr-h): pass reasonable options on to the interpolator
    def nearest_interp(x, out=None):
        """Interpolating function with vectorization."""
        x, x_type, x_is_scalar = _check_interp_input(x, f)
        interpolator = _NearestInterpolator(coord_vecs, f, input_type=x_type)

        res = interpolator(x, out=out)
        if x_is_scalar:
            res = res.item()
        return res

    return nearest_interp


def linear_interpolator(f, coord_vecs):
    """Return the linear interpolator for discrete function values.

    Parameters
    ----------
    f : numpy.ndarray
        Function values that should be interpolated.
    coord_vecs : sequence of numpy.ndarray
        Coordinate vectors of the rectangular grid on which interpolation
        should be based. They must be sorted in ascending order. Usually
        they are obtained as ``grid.coord_vectors`` from a `RectGrid`.

    Returns
    -------
    interpolator : function
        Python function that will interpolate the given values when called
        with a point or multiple points (vectorized).

    Examples
    --------
    We interpolate a 1d function. If called with a single point, the
    interpolator returns a single value, and with multiple points at once,
    an array of values is returned:

    >>> part = odl.uniform_partition(0, 2, 5)
    >>> part.coord_vectors  # grid points
    (array([ 0.2,  0.6,  1. ,  1.4,  1.8]),)
    >>> f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> interpolator = linear_interpolator(f, part.coord_vectors)
    >>> interpolator(0.3)  # 0.75 * 1 + 0.25 * 2 = 1.25
    1.25
    >>> # At 1.9, the value is interpolated between the last value 5.0 and
    >>> # 0.0. The extra interpolation node is placed at the same distance
    >>> # as the second-to-last, i.e., at 2.2. Hence, the interpolated value
    >>> # is 0.75 * 5.0 + 0.25 * 0.0 = 3.75.
    >>> interpolator([0.6, 1.3, 1.9])
    array([ 2.  ,  3.75,  3.75])

    In 2 dimensions, we can either use a (transposed) list of points or
    a meshgrid:

    >>> part = odl.uniform_partition([0, 0], [1, 5], shape=(2, 4))
    >>> part.coord_vectors  # grid points
    (array([ 0.25,  0.75]), array([ 0.625,  1.875,  3.125,  4.375]))
    >>> f = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8]],
    ...              dtype=float)
    >>> interpolator = linear_interpolator(f, part.coord_vectors)
    >>> interpolator([1, 1])  # single point
    2.65
    >>> x = np.array([[0.5, 2.0],
    ...               [0.0, 4.5],
    ...               [0.0, 3.0]]).T  # 3 points at once
    >>> interpolator(x)
    array([ 4.1 ,  1.8 ,  1.45])
    >>> from odl.discr.grid import sparse_meshgrid
    >>> mesh = sparse_meshgrid([0.0, 0.5, 1.0], [1.5, 3.5])
    >>> interpolator(mesh)  # 3x2 grid of points
    array([[ 0.85,  1.65],
           [ 3.7 ,  5.3 ],
           [ 2.85,  3.65]])
    """
    f = asarray(f)

    # TODO(kohr-h): pass reasonable options on to the interpolator
    def linear_interp(x, out=None):
        """Interpolating function with vectorization."""
        x, x_type, x_is_scalar = _check_interp_input(x, f)
        interpolator = _LinearInterpolator(
            coord_vecs,
            f,
            input_type=x_type
        )

        res = interpolator(x, out=out)
        if x_is_scalar:
            res = res.item()
        return res

    return linear_interp


def per_axis_interpolator(f, coord_vecs, interp):
    """Return a per axis defined interpolator for discrete values.

    With this function, the interpolation scheme can be chosen for each axis
    separately.

    Parameters
    ----------
    f : numpy.ndarray
        Function values that should be interpolated.
    coord_vecs : sequence of numpy.ndarray
        Coordinate vectors of the rectangular grid on which interpolation
        should be based. They must be sorted in ascending order. Usually
        they are obtained as ``grid.coord_vectors`` from a `RectGrid`.
    interp : str or sequence of str
        Indicates which interpolation scheme to use for which axis.
        A single string is interpreted as a global scheme for all
        axes.

    Examples
    --------
    Choose linear interpolation in the first axis and nearest neighbor in
    the second:

    >>> part = odl.uniform_partition([0, 0], [1, 5], shape=(2, 4))
    >>> part.coord_vectors
    (array([ 0.25,  0.75]), array([ 0.625,  1.875,  3.125,  4.375]))
    >>> f = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8]],
    ...              dtype=float)
    >>> interpolator = per_axis_interpolator(
    ...     f, part.coord_vectors, ['linear', 'nearest']
    ... )
    >>> interpolator([1, 1])  # single point
    2.5
    >>> x = np.array([[0.5, 2.0],
    ...               [0.0, 4.5],
    ...               [0.0, 3.0]]).T  # 3 points at once
    >>> interpolator(x)
    array([ 4. ,  2. ,  1.5])
    >>> from odl.discr.grid import sparse_meshgrid
    >>> mesh = sparse_meshgrid([0.0, 0.5, 1.0], [1.5, 3.5])
    >>> interpolator(mesh)  # 3x2 grid of points
    array([[ 1. ,  1.5],
           [ 4. ,  5. ],
           [ 3. ,  3.5]])
    """
    f = asarray(f)

    interp = _normalize_interp(interp, f.ndim)

    def per_axis_interp(x, out=None):
        """Interpolating function with vectorization."""
        x, x_type, x_is_scalar = _check_interp_input(x, f)
        interpolator = _PerAxisInterpolator(
            coord_vecs, f, interp=interp, input_type=x_type
        )

        res = interpolator(x, out=out)
        if x_is_scalar:
            res = res.item()
        return res

    return per_axis_interp


class _Interpolator(object):
    r"""Abstract interpolator class.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_
    class.

    The init method does not convert to floating point to
    support arbitrary data type for nearest neighbor interpolation.

    Subclasses need to override ``_evaluate`` for concrete
    implementations.
    """

    def __init__(self, coord_vecs, values, input_type):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid.
        values : `array-like`
            Grid values to use for interpolation.
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``.
        """
        values = np.asarray(values)
        typ_ = str(input_type).lower()
        if typ_ not in ('array', 'meshgrid'):
            raise ValueError('`input_type` ({}) not understood'
                             ''.format(input_type))

        if len(coord_vecs) != values.ndim:
            raise ValueError(
                'there are {} point arrays, but `values` has {} dimensions'
                ''.format(len(coord_vecs), values.ndim)
            )
        for i, p in enumerate(coord_vecs):
            if not np.asarray(p).ndim == 1:
                raise ValueError('the points in dimension {} must be '
                                 '1-dimensional'.format(i))
            if values.shape[i] != len(p):
                raise ValueError('there are {} points and {} values in '
                                 'dimension {}'.format(len(p),
                                                       values.shape[i], i))

        self.coord_vecs = tuple(np.asarray(p) for p in coord_vecs)
        self.values = values
        self.input_type = input_type

    def __call__(self, x, out=None):
        """Do the interpolation.

        Parameters
        ----------
        x : `meshgrid` (i.e., tuple of arrays) if `input_type` is meshgrid,
            else `numpy.ndarray`.
            Evaluation points of the interpolator.
        out : `numpy.ndarray`, optional
            Array to which the results are written. Needs to have
            correct shape according to input ``x``.

        Returns
        -------
        out : `numpy.ndarray`
            Interpolated values. If ``out`` was given, the returned
            object is a reference to it.
        """
        if self.input_type == 'meshgrid':
            # Given a meshgrid, the evaluation will be on a ragged array.
            x = np.asarray(x, dtype=object)
        else:
            x = np.asarray(x)

        ndim = len(self.coord_vecs)
        scalar_out = False

        if self.input_type == 'array':
            if ndim == 1:
                scalar_out = x.ndim == 0
            else:
                scalar_out = x.shape == (ndim,)
            # Make a (1, n) array from one with shape (n,)
            x = x.reshape([ndim, -1])
            out_shape = out_shape_from_array(x)
        else:
            if len(x) != ndim:
                raise ValueError('number of vectors in x is {} instead of '
                                 'the grid dimension {}'
                                 ''.format(len(x), ndim))
            out_shape = out_shape_from_meshgrid(x)

        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('`out` {!r} not a `numpy.ndarray` '
                                'instance'.format(out))
            if out.shape != out_shape:
                raise ValueError('output shape {} not equal to expected '
                                 'shape {}'.format(out.shape, out_shape))
            if out.dtype != self.values.dtype:
                raise ValueError('output dtype {} not equal to expected '
                                 'dtype {}'
                                 ''.format(out.dtype, self.values.dtype))

        indices, norm_distances = self._find_indices(x)
        values = self._evaluate(indices, norm_distances, out)
        if scalar_out:
            return values.item()
        else:
            return values

    def _find_indices(self, x):
        """Find indices and distances of the given nodes.

        Can be overridden by subclasses to improve efficiency.
        """
        # find relevant edges between which xi are situated
        index_vecs = []
        # compute distance to lower edge in unity units
        norm_distances = []

        # iterate through dimensions
        for xi, cvec in zip(x, self.coord_vecs):
            try:
                xi = np.asarray(xi).astype(self.values.dtype, casting='safe')
            except TypeError:
                warn("Unable to infer accurate dtype for"
                  +" interpolation coefficients, defaulting to `float`.")
                xi = np.asarray(xi, dtype=float)

            idcs = np.searchsorted(cvec, xi) - 1

            idcs[idcs < 0] = 0
            idcs[idcs > cvec.size - 2] = cvec.size - 2
            index_vecs.append(idcs)

            norm_distances.append((xi - cvec[idcs]) /
                                  (cvec[idcs + 1] - cvec[idcs]))

        return index_vecs, norm_distances

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluation method, needs to be overridden."""
        raise NotImplementedError('abstract method')


class _NearestInterpolator(_Interpolator):

    """Nearest neighbor interpolator.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.

    This implementation is faster than the more generic one in the
    `_PerAxisPointwiseInterpolator`.
    """

    def __init__(self, coord_vecs, values, input_type):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        """
        super(_NearestInterpolator, self).__init__(
            coord_vecs, values, input_type
        )

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluate nearest interpolation."""
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            idx_res.append(np.where(yi < .5, i, i + 1))
        idx_res = tuple(idx_res)
        if out is not None:
            out[:] = self.values[idx_res]
            return out
        else:
            return self.values[idx_res]


def _compute_nearest_weights_edge(idcs, ndist):
    """Helper for nearest interpolation mimicing the linear case."""
    # Get out-of-bounds indices from the norm_distances. Negative
    # means "too low", larger than or equal to 1 means "too high"
    lo = (ndist < 0)
    hi = (ndist > 1)

    # For "too low" nodes, the lower neighbor gets weight zero;
    # "too high" gets 1.
    w_lo = np.where(ndist < 0.5, 1.0, 0.0)
    w_lo[lo] = 0
    w_lo[hi] = 1

    # For "too high" nodes, the upper neighbor gets weight zero;
    # "too low" gets 1.
    w_hi = np.where(ndist < 0.5, 0.0, 1.0)
    w_hi[lo] = 1
    w_hi[hi] = 0

    # For upper/lower out-of-bounds nodes, we need to set the
    # lower/upper neighbors to the last/first grid point
    edge = [idcs, idcs + 1]
    edge[0][hi] = -1
    edge[1][lo] = 0

    return w_lo, w_hi, edge


def _compute_linear_weights_edge(idcs, ndist):
    """Helper for linear interpolation."""
    ndist = np.asarray(ndist)

    # Get out-of-bounds indices from the norm_distances. Negative
    # means "too low", larger than or equal to 1 means "too high"
    lo = np.where(ndist < 0)
    hi = np.where(ndist > 1)

    # For "too low" nodes, the lower neighbor gets weight zero;
    # "too high" gets 2 - yi (since yi >= 1)
    w_lo = (1 - ndist)
    w_lo[lo] = 0
    w_lo[hi] += 1

    # For "too high" nodes, the upper neighbor gets weight zero;
    # "too low" gets 1 + yi (since yi < 0)
    w_hi = np.copy(ndist)
    w_hi[lo] += 1
    w_hi[hi] = 0

    # For upper/lower out-of-bounds nodes, we need to set the
    # lower/upper neighbors to the last/first grid point
    edge = [idcs, idcs + 1]
    edge[0][hi] = -1
    edge[1][lo] = 0

    return w_lo, w_hi, edge


def _create_weight_edge_lists(indices, norm_distances, interp):
    # Pre-calculate indices and weights (per axis)
    low_weights = []
    high_weights = []
    edge_indices = []
    for i, (idcs, yi, s) in enumerate(zip(indices, norm_distances, interp)):
        if s == 'nearest':
            w_lo, w_hi, edge = _compute_nearest_weights_edge(idcs, yi)
        elif s == 'linear':
            w_lo, w_hi, edge = _compute_linear_weights_edge(idcs, yi)
        else:
            raise ValueError('invalid `interp` {}'.format(interp))

        low_weights.append(w_lo)
        high_weights.append(w_hi)
        edge_indices.append(edge)

    return low_weights, high_weights, edge_indices


class _PerAxisInterpolator(_Interpolator):
    """Interpolator where the scheme is set per axis.

    This allows to use e.g. nearest neighbor interpolation in the
    first dimension and linear in dimensions 2 and 3.
    """

    def __init__(self, coord_vecs, values, input_type, interp):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        interp : sequence of str
            Indicates which interpolation scheme to use for which axis
        """
        super(_PerAxisInterpolator, self).__init__(
            coord_vecs, values, input_type)
        self.interp = interp

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluate per-axis interpolation.

        Modified for in-place evaluation and treatment of out-of-bounds
        points by implicitly assuming 0 at the next node.
        """
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        if out is None:
            out_shape = out_shape_from_meshgrid(norm_distances)
            out_dtype = self.values.dtype
            out = np.zeros(out_shape, dtype=out_dtype)
        else:
            out[:] = 0.0

        # Weights and indices (per axis)
        low_weights, high_weights, edge_indices = _create_weight_edge_lists(
            indices, norm_distances, self.interp)

        # Iterate over all possible combinations of [i, i+1] for each
        # axis, resulting in a loop of length 2**ndim
        for lo_hi, edge in zip(product(*([['l', 'h']] * len(indices))),
                               product(*edge_indices)):
            weight = np.array([1.0], dtype=self.values.dtype)
            # TODO(kohr-h): determine best summation order from array strides
            for lh, w_lo, w_hi in zip(lo_hi, low_weights, high_weights):

                # We don't multiply in-place to exploit the cheap operations
                # in the beginning: sizes grow gradually as following:
                # (n, 1, 1, ...) -> (n, m, 1, ...) -> ...
                # Hence, it is faster to build up the weight array instead
                # of doing full-size operations from the beginning.
                if lh == 'l':
                    weight = weight * w_lo
                else:
                    weight = weight * w_hi
            out += np.asarray(self.values[edge]) * weight[vslice]
        return np.array(out, copy=AVOID_UNNECESSARY_COPY, ndmin=1)


class _LinearInterpolator(_PerAxisInterpolator):
    """Linear (i.e. bi-/tri-/multi-linear) interpolator.

    Convenience class.
    """

    def __init__(self, coord_vecs, values, input_type):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        """
        super(_LinearInterpolator, self).__init__(
            coord_vecs,
            values,
            input_type,
            interp=['linear'] * len(coord_vecs),
        )


def _check_func_out_arg(func):
    """Check if ``func`` has an (optional) ``out`` argument.

    Also verify that the signature of ``func`` has no ``*args`` since
    they make argument propagation a huge hassle.

    Note: this function only works for objects that can be inspected
    with the ``inspect`` module, i.e., Python functions and callables,
    but not, e.g., NumPy UFuncs.

    Parameters
    ----------
    func : callable
        Object that should be inspected.

    Returns
    -------
    has_out : bool
        ``True`` if the signature has an ``out`` argument, ``False``
        otherwise.
    out_is_optional : bool
        ``True`` if ``out`` is present and optional in the signature,
        ``False`` otherwise.

    Raises
    ------
    TypeError
        If ``func``'s signature has ``*args``.
    """
    if sys.version_info.major > 2:
        spec = inspect.getfullargspec(func)
        kw_only = spec.kwonlyargs
    else:
        spec = inspect.getargspec(func)
        kw_only = ()

    if spec.varargs is not None:
        raise TypeError('*args not allowed in function signature')

    pos_args = spec.args
    pos_defaults = () if spec.defaults is None else spec.defaults

    if 'out' in pos_args:
        has_out = True
        out_optional = (
            pos_args.index('out') >= len(pos_args) - len(pos_defaults)
        )
    elif 'out' in kw_only:
        has_out = out_optional = True
    else:
        has_out = out_optional = False

    return has_out, out_optional


def _func_out_type(func):
    """Determine the output argument type (if any) of a function-like object.

    This function is intended to work with all types of callables
    that are used as input to `sampling_function`.
    """
    # Numpy `UFuncs` and similar objects (e.g. Numba `DUFuncs`)
    if hasattr(func, 'nin') and hasattr(func, 'nout'):
        if func.nin != 1:
            raise ValueError(
                'ufunc {} takes {} input arguments, expected 1'
                ''.format(func.__name__, func.nin)
            )
        if func.nout > 1:
            raise ValueError(
                'ufunc {} returns {} outputs, expected 0 or 1'
                ''.format(func.__name__, func.nout)
            )
        has_out = out_optional = (func.nout == 1)
    elif inspect.isfunction(func):
        has_out, out_optional = _check_func_out_arg(func)
    elif callable(func):
        has_out, out_optional = _check_func_out_arg(func.__call__)
    else:
        raise TypeError('object {!r} not callable'.format(func))

    return has_out, out_optional


def _broadcast_nested_list(arr_lists, element_shape, ndim, backend: ArrayBackend):
    """ A generalisation of `np.broadcast_to`, applied to an arbitrarily
    deep list (or tuple) eventually containing arrays or scalars. """
    if isinstance(arr_lists, backend.array_type) or np.isscalar(arr_lists):
        if ndim == 1:
            # As usual, 1d is tedious to deal with. This
            # code deals with extra dimensions in result
            # components that stem from using x instead of
            # x[0] in a function.
            # Without this, broadcasting fails.
            shp = getattr(arr_lists, 'shape', ())
            if shp and shp[0] == 1:
                arr_lists = arr_lists.reshape(arr_lists.shape[1:])
        return backend.array_namespace.broadcast_to(arr_lists, element_shape)
    else:
        return [_broadcast_nested_list(row, element_shape, ndim, backend=backend)
                 for row in arr_lists]

def _send_nested_list_to_backend(
        arr_lists, backend : ArrayBackend, device, dtype
        ):
    if backend.impl == 'numpy':
        return arr_lists

    if isinstance(arr_lists, np.ndarray) or np.isscalar(arr_lists):
        return backend.array_constructor(arr_lists, device=device, dtype=dtype)

    elif isinstance(arr_lists, (tuple,list)):
        return [_send_nested_list_to_backend(arr, backend, device, dtype) for arr in arr_lists]

    else:
        raise TypeError(f'Type of input {type(arr_lists)} not supported.')

def sampling_function(
    func : Callable | list | tuple,
    domain : IntervalProd,
    out_dtype : str | None,
    impl: str ='numpy',
    device: str ='cpu'
    ):
    def _infer_dtype(out_dtype : str | None):
        if out_dtype is None:
            out_dtype = 'float64'
        else:
            assert is_floating_dtype(out_dtype)
        # This is to replicate the old behaviour:
        # np.dtype('float64') = ()
        val_shape = ()
        return out_dtype, val_shape

    def _sanitise_callable(func: Callable) -> Callable:
        # Get default implementations if necessary
        has_out, out_optional = _func_out_type(func)

        if has_out:
            raise NotImplementedError('Currently, not implemented for out-of-place functions')

        return func

    def _sanitise_array_of_callables(funcs : List | Tuple):
        raise NotImplementedError('The sampling function cannot be instantiated with a list-like of callables.')

    def _sanitise_input_function(func: Callable | list | tuple):
        '''
        This function aims at unpacking the input function `func`.
        The former API expects a callable or array-like (of callables)
        A callable (or each callable) must take a single input and may
        accept one output parameter called ``out``, and should return
        its result.
        '''
        if isinstance(func, Callable):
            return _sanitise_callable(func)
        elif isinstance(func, (list, tuple)):
            return _sanitise_array_of_callables(func)
        else:
            raise NotImplementedError('The function to sample must be either a Callable or an array-like (list, tuple) of callables.')

    ### We begin by sanitising the inputs:
    # 1) the dtype
    out_dtype = _infer_dtype(out_dtype)
    # 2) the func_or_arr
    func = _sanitise_input_function(func)

    ### We then create the function
    return _make_single_use_func(func, domain, out_dtype, impl, device)



def old_sampling_function(func_or_arr, domain, out_dtype=None, impl: str ='numpy', device: str ='cpu'):
    """Return a function that can be used for sampling.

    For examples on this function's usage, see `point_collocation`.

    Parameters
    ----------
    func_or_arr : callable or array-like
        Either a single callable object (possibly with multiple output
        components), or an array or callables and constants.
        A callable (or each callable) must take a single input and may
        accept one output parameter called ``out``, and should return
        its result.
    domain : IntervalProd
        Set in which inputs to the function are assumed to lie. It is used
        to determine the type of input (point/meshgrid/array) based on
        ``domain.ndim``, and (unless switched off) to check whether all
        inputs are in bounds.
    out_dtype : optional
        Data type of a *single* output of ``func_or_arr``, i.e., when
        called with a single point as input. In particular:

        - If ``func_or_arr`` is a scalar-valued function, ``out_dtype`` is
          expected to be a basic dtype with empty shape.
        - If ``func_or_arr`` is a vector- or tensor-valued function,
          ``out_dtype`` should be a shaped data type, e.g., ``(float, (3,))``
          for a vector-valued function with 3 components.
        - If ``func_or_arr`` is an array-like, ``out_dtype`` should be a
          shaped dtype whose shape matches that of ``func_or_arr``. It can
          also be ``None``, in which case the shape is inferred, and the
          scalar data type is set to ``float``.

    Returns
    -------
    func : function
        Wrapper function that has no optional ``out`` argument.
    """
    if out_dtype is None:
        val_shape = None
        scalar_out_dtype = np.dtype('float64')
    else:
        out_dtype = np.dtype(out_dtype)
        val_shape = out_dtype.shape
        scalar_out_dtype = out_dtype.base

    # Provide default implementations of missing function signature types

    def _default_oop(func_ip, x, **kwargs):
        """Default out-of-place variant of an in-place-only function."""
        if is_valid_input_array(x, domain.ndim):
            scalar_out_shape = out_shape_from_array(x)
        elif is_valid_input_meshgrid(x, domain.ndim):
            scalar_out_shape = out_shape_from_meshgrid(x)
        else:
            raise TypeError('invalid input `x`')

        out_shape = val_shape + scalar_out_shape
        out = np.empty(out_shape, dtype=scalar_out_dtype)
        func_ip(x, out=out, **kwargs)
        return out

    def _default_ip(func_oop, x, out, **kwargs):
        """Default in-place variant of an out-of-place-only function."""
        result = func_oop(x, **kwargs)
        try:
            result = np.array(result, copy=AVOID_UNNECESSARY_COPY)
        except ValueError:
            # Different shapes encountered, need to broadcast
            if is_valid_input_array(x, domain.ndim):
                scalar_out_shape = out_shape_from_array(x)
            elif is_valid_input_meshgrid(x, domain.ndim):
                scalar_out_shape = out_shape_from_meshgrid(x)
            else:
                raise TypeError('invalid input `x`')

            bcast_results = _broadcast_nested_list(
                                   result, scalar_out_shape, domain.ndim)

            # New array that is flat in the `out_shape` axes, reshape it
            # to the final `out_shape + scalar_shape`, using the same
            # order ('C') as the initial `result.ravel()`.
            result = np.array(bcast_results, dtype=scalar_out_dtype)
            result = result.reshape(val_shape + scalar_out_shape)

        # The following code is required to remove extra axes, e.g., when
        # the result has shape (2, 1, 3) but should have shape (2, 3).
        # For those cases, broadcasting doesn't apply.
        try:
            reshaped = result.reshape(out.shape)
        except ValueError:
            # This is the case when `result` must be broadcast
            out[:] = result
        else:
            out[:] = reshaped

        return out

    # Now prepare the in-place and out-of-place functions for the final
    # wrapping.

    if callable(func_or_arr):
        # Assume scalar float out dtype for single function
        if out_dtype is None:
            out_dtype = np.dtype('float64')

        # Got a (single) function, possibly need to vectorize
        func = func_or_arr

        # Get default implementations if necessary
        has_out, out_optional = _func_out_type(func)
        if not has_out:
            # Out-of-place-only
            func_oop = func
        elif out_optional:
            # Dual-use
            func_oop = func
        else:
            # In-place-only
            func_oop = partial(_default_oop, func)

    else:
        # This is for the case that an array-like of callables is provided.
        # We need to convert this into a single function that returns an
        # array.

        arr = np.array(func_or_arr, dtype=object)

        if val_shape is None:
            # Infer value shape if `out_dtype is None`
            val_shape = arr.shape
        elif arr.shape != val_shape:
            # Otherwise, check that the value shape matches the dtype shape
            raise ValueError(
                'invalid `func_or_arr` {!r}: expected `None`, a callable or '
                'an array-like of callables whose shape matches '
                '`out_dtype.shape` {}'.format(func_or_arr, val_shape)
            )

        out_dtype = np.dtype((scalar_out_dtype, val_shape))

        arr = arr.ravel().tolist()

        def array_wrapper_func(x, out=None, **kwargs):
            """Function wrapping an array of callables and constants.

            This wrapper does the following for out-of-place
            evaluation (when ``out=None``):

            1. Collect the results of all function evaluations into
               a list, handling all kinds of sequence entries
               (normal function, ufunc, constant, etc.).
            2. Broadcast all results to the desired shape that is
               determined by the space's ``out_shape`` and the
               shape(s) of the input.
            3. Form a big array containing the final result.

            The in-place version is simpler because broadcasting
            happens automatically when assigning to the components
            of ``out``. Hence, we only have

            1. Assign the result of the evaluation of the i-th
               function to ``out_flat[i]``, possibly using the
               ``out`` parameter of the function.
            """
            if is_valid_input_meshgrid(x, domain.ndim):
                scalar_out_shape = out_shape_from_meshgrid(x)
            elif is_valid_input_array(x, domain.ndim):
                scalar_out_shape = out_shape_from_array(x)
            else:
                raise RuntimeError('bad input')

            if out is None:
                # Out-of-place evaluation

                # Collect results of member functions into a list.
                # Put simply, all that happens here is
                # `results.append(f(x))`, just for a bunch of cases
                # and with or without `out`.
                results = []
                for f in arr:
                    if np.isscalar(f):
                        # Constant function
                        results.append(f)
                    elif not callable(f):
                        raise TypeError(
                            'element {!r} of `func_or_arr` not callable'
                            ''.format(f)
                        )
                    elif hasattr(f, 'nin') and hasattr(f, 'nout'):
                        # ufunc-like object
                        results.append(f(x, **kwargs))
                    else:
                        has_out, _ = _func_out_type(f)
                        if has_out:
                            out = np.empty(
                                scalar_out_shape, dtype=scalar_out_dtype
                            )
                            f(x, out=out, **kwargs)
                            results.append(out)
                        else:
                            results.append(f(x, **kwargs))

                # Broadcast to required shape and convert to array.
                # This will raise an error if the shape of some member
                # array is wrong, since in that case the resulting
                # dtype would be `object`.
                bcast_results = []
                for res in results:
                    try:
                        reshaped = np.reshape(res, scalar_out_shape)
                    except ValueError:
                        bcast_results.append(
                            np.broadcast_to(res, scalar_out_shape))
                    else:
                        bcast_results.append(reshaped)

                out_arr = np.array(
                    bcast_results, dtype=scalar_out_dtype
                )

                return out_arr.reshape(val_shape + scalar_out_shape)

            else:
                # In-place evaluation

                # This is a precaution in case out is not contiguous
                with writable_array(out) as out_arr:
                    # Flatten tensor axes to work on one tensor
                    # component (= scalar function) at a time
                    out_comps = out_arr.reshape((-1,) + scalar_out_shape)
                    for f, out_comp in zip(arr, out_comps):
                        if np.isscalar(f):
                            out_comp[:] = f
                        else:
                            has_out, _ = _func_out_type(f)
                            if has_out:
                                f(x, out=out_comp, **kwargs)
                            else:
                                out_comp[:] = f(x, **kwargs)

        func_oop = array_wrapper_func

    return _make_single_use_func(func_oop, domain, out_dtype, impl=impl, device=device)


def _make_single_use_func(func_oop, domain, out_dtype, impl: str ='numpy', device: str ='cpu'):
    """Return a unifying wrapper function with optional ``out`` argument."""

    # Default to `ndim=1` for unusual domains that do not define a dimension
    # (like `Strings(3)`)
    ndim = getattr(domain, 'ndim', 1)
    if out_dtype is None:
        # Don't let `np.dtype` convert `None` to `float64`
        raise TypeError('`out_dtype` cannot be `None`')

    out_dtype = np.dtype(out_dtype)
    val_shape = out_dtype.shape
    scalar_out_dtype = out_dtype.base

    tensor_valued = val_shape != ()

    def dual_use_func(x, **kwargs):
        """Wrapper function with optional ``out`` argument.

        This function closes over two other functions, one for in-place,
        the other for out-of-place evaluation. Its purpose is to unify their
        interfaces to a single one with optional ``out`` argument, and to
        automate all details of input/output checking, broadcasting and
        type casting.

        The closure also contains ``domain``, an `IntervalProd` where points
        should lie, and the expected ``out_dtype``.

        For usage examples, see `point_collocation`.

        Parameters
        ----------
        x : point, `meshgrid` or `numpy.ndarray`
            Input argument for the function evaluation. Conditions
            on ``x`` depend on its type:

            - point: must be castable to an element of the enclosed ``domain``.
            - meshgrid: length must be ``domain.ndim``, and the arrays must
              be broadcastable against each other.
            - array: shape must be ``(ndim, N)``, where ``ndim`` equals
              ``domain.ndim``.

        out : `numpy.ndarray`, optional
            Output argument holding the result of the function evaluation.
            Its shape must be ``out_dtype.shape + np.broadcast(*x).shape``.

        Other Parameters
        ----------------
        bounds_check : bool, optional
            If ``True``, check if all input points lie in ``domain``. This
            requires ``domain`` to implement `Set.contains_all`.
            Default: ``True``

        Returns
        -------
        out : `numpy.ndarray`
            Result of the function evaluation. If ``out`` was provided,
            the returned object is a reference to it.

        Raises
        ------
        TypeError
            If ``x`` is not a valid vectorized evaluation argument.

            If ``out`` is neither ``None`` nor a `numpy.ndarray` of
            adequate shape and data type.

        ValueError
            If ``bounds_check == True`` and some evaluation points fall
            outside the valid domain.
        """
        bounds_check = kwargs.pop('bounds_check', True)
        if bounds_check and not hasattr(domain, 'contains_all'):
            raise AttributeError(
                'bounds check not possible for domain {!r}, missing '
                '`contains_all()` method'
                ''.format(domain)
            )

        # Check for input type and determine output shape
        if is_valid_input_meshgrid(x, ndim):
            scalar_in = False
            scalar_out_shape = out_shape_from_meshgrid(x)

            # Avoid operations on tuples like x * 2 by casting to array
            if ndim == 1:
                x = x[0][None, ...]
        elif is_valid_input_array(x, ndim):
            x = np.asarray(x)
            scalar_in = False
            scalar_out_shape = out_shape_from_array(x)

        elif x in domain:
            x = np.atleast_2d(x).T  # make a (d, 1) array
            scalar_in = True
            scalar_out_shape = (1,)

        else:
            # Unknown input
            txt_1d = ' or (n,)' if ndim == 1 else ''
            raise TypeError(
                'argument {!r} not a valid function input. '
                'Expected an element of the domain {domain!r}, an array-like '
                'with shape ({domain.ndim}, n){} or a length-{domain.ndim} '
                'meshgrid tuple.'
                ''.format(x, txt_1d, domain=domain)
            )

        # Check bounds if specified
        if bounds_check and not domain.contains_all(x):
            raise ValueError('input contains points outside the domain {!r}'
                             ''.format(domain))

        backend = lookup_array_backend(impl)
        array_ns = backend.array_namespace
        backend_scalar_out_dtype = backend.available_dtypes[_universal_dtype_identifier(scalar_out_dtype)]

        x = _send_nested_list_to_backend(
            x, backend, device, backend_scalar_out_dtype)

        if scalar_in:
            out_shape = val_shape
        else:
            out_shape = val_shape + scalar_out_shape

        if ndim == 1:
            try:
                out = func_oop(x, **kwargs)
            except (TypeError, IndexError):
                # TypeError is raised if a meshgrid was used but the
                # function expected an array (1d only). In this case we try
                # again with the first meshgrid vector.
                # IndexError is raised in expressions like x[x > 0] since
                # "x > 0" evaluates to 'True', i.e. 1, and that index is
                # out of range for a meshgrid tuple of length 1 :-). To get
                # the real errors with indexing, we check again for the
                # same scenario (scalar output when not valid) as in the
                # first case.
                out = func_oop(x[0], **kwargs)

        else:
            # Here we don't catch exceptions since they are likely true
            # errors
            out = func_oop(x, **kwargs)

        def _process_array(out):
            if isinstance(out, backend.array_type) or np.isscalar(out):
                # Cast to proper dtype if needed, also convert to array if out  is a scalar.
                out = backend.array_constructor(
                    out,
                    dtype=backend_scalar_out_dtype,
                    device=device
                    )
                if scalar_in:
                    out = array_ns.squeeze(out,0)
                elif ndim == 1 and out.shape == (1,) + out_shape:
                    out = out.reshape(out_shape)

                if out_shape != () and out.shape != out_shape:
                    # Broadcast the returned element, but not in the
                    # scalar case. The resulting array may be read-only,
                    # in which case we copy.
                    out = array_ns.broadcast_to(out, out_shape)
                    out = backend.array_constructor(out, copy=True)
                return out

            elif isinstance(out, (tuple, list)):
                result = []
                assert len(out) != 0
                for sub_out in out:
                    result.append(_process_array(sub_out))
                return result

        return _process_array(out)

    return dual_use_func


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
