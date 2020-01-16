# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Helpers for discretization-related functionality."""

from __future__ import absolute_import, division, print_function

from itertools import product

import numpy as np

from odl.util import (
    is_string, is_valid_input_array, is_valid_input_meshgrid,
    out_shape_from_array, out_shape_from_meshgrid)

__all__ = (
    'point_collocation',
    'nearest_interpolator',
    'linear_interpolator',
    'per_axis_interpolator',
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
    >>> fspace = odl.FunctionSpace(domain)
    >>> func = fspace.element(lambda x: x ** 2)
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
    >>> fspace = odl.FunctionSpace(domain)
    >>> xs = [1, 2]
    >>> ys = [3, 4, 5]
    >>> mesh = sparse_meshgrid(xs, ys)
    >>> func = fspace.element(lambda x: x[0] - x[1])
    >>> point_collocation(func, mesh)
    array([[-2., -3., -4.],
           [-1., -2., -3.]])

    It is possible to return results that require broadcasting, and to use
    *optional* function parameters:

    >>> def f(x, c=0):
    ...     return x[0] + c
    >>> func = fspace.element(f)
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
    >>> # Need to tell the wrapper that we want a 3-component function
    >>> fspace = odl.FunctionSpace(domain, out_dtype=(float, (3,)))
    >>> xs = [1, 2]
    >>> ys = [3, 4]
    >>> mesh = sparse_meshgrid(xs, ys)
    >>> def vec_valued(x):
    ...     return (x[0] - 1, 0, x[0] + x[1])  # broadcasting
    >>> func1 = fspace.element(vec_valued)
    >>> point_collocation(func1, mesh)
    array([[[ 0.,  0.],
            [ 1.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.],
            [ 0.,  0.]],
    <BLANKLINE>
           [[ 4.,  5.],
            [ 5.,  6.]]])
    >>> list_of_funcs = [
    ...     lambda x: x[0] - 1,
    ...     0,                   # constants are allowed
    ...     lambda x: x[0] + x[1]
    ... ]
    >>> func2 = fspace.element(list_of_funcs)
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
    make_func_for_sampling :
        wrap a function so it can handle all valid types of input
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


def _all_interp_equal(interp_byaxis):
    """Whether all entries are equal, with ``False`` for length 0."""
    if len(interp_byaxis) == 0:
        return False
    return all(itp == interp_byaxis[0] for itp in interp_byaxis)


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
                ''.format(len(interp_byaxis, ndim))
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
        I(x) = f[j]  with j such that |x - x[j]| is minimal.
    The ambiguity at the midpoints is resolved by preferring one of the
    neighbors. In higher dimensions, this principle is applied per axis.
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
    """Return the linear interpolator for discrete values.

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
    >>> f = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> interpolator = linear_interpolator(f, part.coord_vectors)
    >>> interpolator(0.3)  # 0.75 * 1 + 0.25 * 2 = 1.25
    1.25
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
    f = np.asarray(f)

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
    f = np.asarray(f)

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
    <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_ class.

    The init method does not convert to floating point to
    support arbitrary data type for nearest neighbor interpolation.

    Subclasses need to override ``_evaluate`` for concrete
    implementations.
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
        values = np.asarray(values)
        typ_ = str(input_type).lower()
        if typ_ not in ('array', 'meshgrid'):
            raise ValueError('`input_type` ({}) not understood'
                             ''.format(input_type))

        if len(coord_vecs) > values.ndim:
            raise ValueError('there are {} point arrays, but `values` has {} '
                             'dimensions'.format(len(coord_vecs),
                                                 values.ndim))
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
        x : `meshgrid` or `numpy.ndarray`
            Evaluation points of the interpolator
        out : `numpy.ndarray`, optional
            Array to which the results are written. Needs to have
            correct shape according to input ``x``.

        Returns
        -------
        out : `numpy.ndarray`
            Interpolated values. If ``out`` was given, the returned
            object is a reference to it.
        """
        ndim = len(self.coord_vecs)
        if self.input_type == 'array':
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
        return self._evaluate(indices, norm_distances, out)

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
    r"""Nearest neighbor interpolator.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_ class.

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
    # Precalculate indices and weights (per axis)
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
        """Evaluate linear interpolation.

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
            weight = 1.0
            # TODO: determine best summation order from array strides
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
        return np.array(out, copy=False, ndmin=1)


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


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
