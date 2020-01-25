# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `discr_utils`."""

from __future__ import division

from functools import partial

import numpy as np
import pytest

import odl
from odl.discr.discr_utils import (
    linear_interpolator, make_func_for_sampling, nearest_interpolator,
    per_axis_interpolator, point_collocation)
from odl.discr.grid import sparse_meshgrid
from odl.util.testutils import all_almost_equal, all_equal, simple_fixture


# --- Helper functions --- #


def _test_eq(x, y):
    """Test equality of x and y."""
    assert x == y
    assert not x != y
    assert hash(x) == hash(y)


def _test_neq(x, y):
    """Test non-equality of x and y."""
    assert x != y
    assert not x == y
    assert hash(x) != hash(y)


def _points(domain, num):
    """Helper to generate ``num`` points in ``domain``."""
    min_pt = domain.min_pt
    max_pt = domain.max_pt
    ndim = domain.ndim
    points = np.random.uniform(low=0, high=1, size=(ndim, num))
    for i in range(ndim):
        points[i, :] = min_pt[i] + (max_pt[i] - min_pt[i]) * points[i]
    return points


def _meshgrid(domain, shape):
    """Helper to generate a ``shape`` meshgrid of points in ``domain``."""
    min_pt = domain.min_pt
    max_pt = domain.max_pt
    ndim = domain.ndim
    coord_vecs = []
    for i in range(ndim):
        vec = np.random.uniform(low=min_pt[i], high=max_pt[i], size=shape[i])
        vec.sort()
        coord_vecs.append(vec)
    return sparse_meshgrid(*coord_vecs)


class FuncList(list):  # So we can set __name__
    pass


# --- pytest fixtures (general) --- #


out_dtype = simple_fixture(
    'out_dtype',
    ['float32', 'float64', 'complex64'],
    fmt=' {name} = {value!r} '
)
domain_ndim = simple_fixture('domain_ndim', [1, 2])


# --- pytest fixtures (scalar test functions) --- #


def func_nd_oop(x):
    return sum(x)


def func_nd_ip(x, out):
    out[:] = sum(x)


def func_nd_dual(x, out=None):
    if out is None:
        return sum(x)
    else:
        out[:] = sum(x)


def func_nd_bcast_ref(x):
    return x[0] + 0 * sum(x[1:])


def func_nd_bcast_oop(x):
    return x[0]


def func_nd_bcast_ip(x, out):
    out[:] = x[0]


def func_nd_bcast_dual(x, out=None):
    if out is None:
        return x[0]
    else:
        out[:] = x[0]


func_nd_ref = func_nd_oop
func_nd_params = [(func_nd_ref, f)
                  for f in [func_nd_oop, func_nd_ip, func_nd_dual]]
func_nd_params.extend([(func_nd_bcast_ref, func_nd_bcast_oop),
                       (func_nd_bcast_ref, func_nd_bcast_ip)])

func_nd = simple_fixture('func_nd', func_nd_params,
                         fmt=' {name} = {value[1].__name__} ')


def func_nd_other(x):
    return sum(x) + 1


def func_param_nd_oop(x, c):
    return sum(x) + c


def func_param_nd_ip(x, out, c):
    out[:] = sum(x) + c


def func_param_switched_nd_ip(x, c, out):
    out[:] = sum(x) + c


def func_param_bcast_nd_ref(x, c):
    return x[0] + c + 0 * sum(x[1:])


def func_param_bcast_nd_oop(x, c):
    return x[0] + c


def func_param_bcast_nd_ip(x, out, c):
    out[:] = x[0] + c


func_param_nd_ref = func_param_nd_oop
func_param_nd_params = [(func_param_nd_ref, f)
                        for f in [func_param_nd_oop, func_param_nd_ip,
                                  func_param_switched_nd_ip]]
func_param_nd_params.extend(
    [(func_param_bcast_nd_ref, func_param_bcast_nd_oop),
     (func_param_bcast_nd_ref, func_param_bcast_nd_ip)])
func_param_nd = simple_fixture('func_with_param', func_param_nd_params,
                               fmt=' {name} = {value[1].__name__} ')


def func_1d_ref(x):
    return x[0] * 2


def func_1d_oop(x):
    return x * 2


def func_1d_ip(x, out):
    out[:] = x * 2


func_1d_params = [
    (func_1d_ref, func_1d_oop),
    (func_1d_ref, func_1d_ip),
    (lambda x: -x[0], np.negative),
]
func_1d = simple_fixture('func_1d', func_1d_params,
                         fmt=' {name} = {value[1].__name__} ')


def func_complex_nd_oop(x):
    return sum(x) + 1j


# --- pytest fixtures (vector-valued test functions) --- #


def func_vec_nd_ref(x):
    return np.array([sum(x) + 1, sum(x) - 1])


def func_vec_nd_oop(x):
    return (sum(x) + 1, sum(x) - 1)


func_nd_oop_seq = FuncList([lambda x: sum(x) + 1, lambda x: sum(x) - 1])
func_nd_oop_seq.__name__ = 'func_nd_oop_seq'


def func_vec_nd_ip(x, out):
    out[0] = sum(x) + 1
    out[1] = sum(x) - 1


def comp0_nd(x, out):
    out[:] = sum(x) + 1


def comp1_nd(x, out):
    out[:] = sum(x) - 1


def func_vec_nd_dual(x, out=None):
    if out is None:
        return (sum(x) + 1, sum(x) - 1)
    else:
        out[0] = sum(x) + 1
        out[1] = sum(x) - 1


func_nd_ip_seq = FuncList([comp0_nd, comp1_nd])
func_nd_ip_seq.__name__ = 'func_nd_ip_seq'

func_vec_nd_params = [(func_vec_nd_ref, f)
                      for f in [func_vec_nd_oop, func_nd_oop_seq,
                                func_vec_nd_ip, func_nd_ip_seq]]
func_vec_nd = simple_fixture('func_vec_nd', func_vec_nd_params,
                             fmt=' {name} = {value[1].__name__} ')


def func_vec_nd_other(x):
    return np.array([sum(x) + 2, sum(x) + 3])


def func_vec_1d_ref(x):
    return np.array([x[0] * 2, x[0] + 1])


def func_vec_1d_oop(x):
    return (x * 2, x + 1)


func_1d_oop_seq = FuncList([lambda x: x * 2, lambda x: x + 1])
func_1d_oop_seq.__name__ = 'func_1d_oop_seq'


def func_vec_1d_ip(x, out):
    out[0] = x * 2
    out[1] = x + 1


def comp0_1d(x, out):
    out[:] = x * 2


def comp1_1d(x, out):
    out[:] = x + 1


func_1d_ip_seq = FuncList([comp0_1d, comp1_1d])
func_1d_ip_seq.__name__ = 'func_1d_ip_seq'

func_vec_1d_params = [(func_vec_1d_ref, f)
                      for f in [func_vec_1d_oop, func_1d_oop_seq,
                                func_vec_1d_ip, func_1d_ip_seq]]
func_vec_1d = simple_fixture('func_vec_1d', func_vec_1d_params,
                             fmt=' {name} = {value[1].__name__} ')


def func_vec_complex_nd_oop(x):
    return (sum(x) + 1j, sum(x) - 1j)


# --- pytest fixtures (tensor-valued test functions) --- #


def func_tens_ref(x):
    # Reference function where all shapes in the list are correct
    # without broadcasting
    shp = np.broadcast(*x).shape
    return np.array([[x[0] - x[1], np.zeros(shp), x[1] + 0 * x[0]],
                     [np.ones(shp), x[0] + 0 * x[1], sum(x)]])


def func_tens_oop(x):
    # Output shape 2x3, input 2-dimensional. Broadcasting supported.
    return [[x[0] - x[1], 0, x[1]],
            [1, x[0], sum(x)]]


def func_tens_ip(x, out):
    # In-place version
    out[0, 0] = x[0] - x[1]
    out[0, 1] = 0
    out[0, 2] = x[1]
    out[1, 0] = 1
    out[1, 1] = x[0]
    out[1, 2] = sum(x)


# Array of functions. May contain constants. Should yield the same as func.
func_tens_oop_seq = FuncList([[lambda x: x[0] - x[1], 0, lambda x: x[1]],
                              [1, lambda x: x[0], lambda x: sum(x)]])
func_tens_oop_seq.__name__ = 'func_tens_oop_seq'


# In-place component functions, cannot use lambdas
def comp00(x, out):
    out[:] = x[0] - x[1]


def comp01(x, out):
    out[:] = 0


def comp02(x, out):
    out[:] = x[1]


def comp10(x, out):
    out[:] = 1


def comp11(x, out):
    out[:] = x[0]


def comp12(x, out):
    out[:] = sum(x)


func_tens_ip_seq = FuncList([[comp00, comp01, comp02],
                             [comp10, comp11, comp12]])
func_tens_ip_seq.__name__ = 'func_tens_ip_seq'


def func_tens_dual(x, out=None):
    if out is None:
        return [[x[0] - x[1], 0, x[1]],
                [1, x[0], sum(x)]]
    else:
        out[0, 0] = x[0] - x[1]
        out[0, 1] = 0
        out[0, 2] = x[1]
        out[1, 0] = 1
        out[1, 1] = x[0]
        out[1, 2] = sum(x)


func_tens_params = [(func_tens_ref, f)
                    for f in [func_tens_oop, func_tens_oop_seq,
                              func_tens_ip, func_tens_ip_seq]]
func_tens = simple_fixture('func_tens', func_tens_params,
                           fmt=' {name} = {value[1].__name__} ')


def func_tens_other(x):
    return np.array([[x[0] + x[1], sum(x), sum(x)],
                     [sum(x), 2 * x[0] - x[1], sum(x)]])


def func_tens_complex_oop(x):
    return [[x[0], 0, 1j * x[0]],
            [1j, x, sum(x) + 1j]]


# --- point_collocation tests --- #


def test_point_collocation_scalar_valued(domain_ndim, out_dtype, func_nd):
    """Check collocation of scalar-valued functions."""
    domain = odl.IntervalProd([0] * domain_ndim, [1] * domain_ndim)
    points = _points(domain, 3)
    mesh_shape = tuple(range(2, 2 + domain_ndim))
    mesh = _meshgrid(domain, mesh_shape)
    point = [0.5] * domain_ndim

    func_ref, func = func_nd

    true_values_points = func_ref(points)
    true_values_mesh = func_ref(mesh)
    true_value_point = func_ref(point)

    sampl_func = make_func_for_sampling(func, domain, out_dtype)
    collocator = partial(point_collocation, sampl_func)

    # Out of place
    result_points = collocator(points)
    result_mesh = collocator(mesh)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)
    assert result_points.dtype == out_dtype
    assert result_mesh.dtype == out_dtype
    assert result_points.flags.writeable
    assert result_mesh.flags.writeable

    # In place
    out_points = np.empty(3, dtype=out_dtype)
    out_mesh = np.empty(mesh_shape, dtype=out_dtype)
    collocator(points, out=out_points)
    collocator(mesh, out=out_mesh)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Single point evaluation
    result_point = collocator(point)
    assert all_almost_equal(result_point, true_value_point)


def test_point_collocation_scalar_valued_with_param(func_param_nd):
    """Check collocation of scalar-valued functions with parameters."""
    domain = odl.IntervalProd([0, 0], [1, 1])
    points = _points(domain, 3)
    mesh_shape = (2, 3)
    mesh = _meshgrid(domain, mesh_shape)

    func_ref, func = func_param_nd

    true_values_points = func_ref(points, c=2.5)
    true_values_mesh = func_ref(mesh, c=2.5)

    sampl_func = make_func_for_sampling(func, domain, out_dtype='float64')
    collocator = partial(point_collocation, sampl_func)

    # Out of place
    result_points = collocator(points, c=2.5)
    result_mesh = collocator(mesh, c=2.5)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)

    # In place
    out_points = np.empty(3, dtype='float64')
    out_mesh = np.empty(mesh_shape, dtype='float64')
    collocator(points, out=out_points, c=2.5)
    collocator(mesh, out=out_mesh, c=2.5)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Complex output
    true_values_points = func_ref(points, c=2j)
    true_values_mesh = func_ref(mesh, c=2j)

    sampl_func = make_func_for_sampling(func, domain, out_dtype='complex128')
    collocator = partial(point_collocation, sampl_func)

    result_points = collocator(points, c=2j)
    result_mesh = collocator(mesh, c=2j)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)


def test_point_collocation_vector_valued(func_vec_nd):
    """Check collocation of vector-valued functions."""
    domain = odl.IntervalProd([0, 0], [1, 1])
    points = _points(domain, 3)
    mesh_shape = (2, 3)
    mesh = _meshgrid(domain, mesh_shape)
    point = [0.5, 0.5]
    values_points_shape = (2, 3)
    values_mesh_shape = (2, 2, 3)

    func_ref, func = func_vec_nd

    true_values_points = func_ref(points)
    true_values_mesh = func_ref(mesh)
    true_value_point = func_ref(point)

    sampl_func = make_func_for_sampling(
        func, domain, out_dtype=('float64', (2,))
    )
    collocator = partial(point_collocation, sampl_func)

    # Out of place
    result_points = collocator(points)
    result_mesh = collocator(mesh)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)
    assert result_points.dtype == 'float64'
    assert result_mesh.dtype == 'float64'
    assert result_points.flags.writeable
    assert result_mesh.flags.writeable

    # In place
    out_points = np.empty(values_points_shape, dtype='float64')
    out_mesh = np.empty(values_mesh_shape, dtype='float64')
    collocator(points, out=out_points)
    collocator(mesh, out=out_mesh)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Single point evaluation
    result_point = collocator(point)
    assert all_almost_equal(result_point, true_value_point)
    out_point = np.empty((2,), dtype='float64')
    collocator(point, out=out_point)
    assert all_almost_equal(out_point, true_value_point)


def test_point_collocation_tensor_valued(func_tens):
    """Check collocation of tensor-valued functions."""
    domain = odl.IntervalProd([0, 0], [1, 1])
    points = _points(domain, 4)
    mesh_shape = (4, 5)
    mesh = _meshgrid(domain, mesh_shape)
    point = [0.5, 0.5]
    values_points_shape = (2, 3, 4)
    values_mesh_shape = (2, 3, 4, 5)
    value_point_shape = (2, 3)

    func_ref, func = func_tens

    true_result_points = np.array(func_ref(points))
    true_result_mesh = np.array(func_ref(mesh))
    true_result_point = np.array(func_ref(np.array(point)[:, None])).squeeze()

    sampl_func = make_func_for_sampling(
        func, domain, out_dtype=('float64', (2, 3))
    )
    collocator = partial(point_collocation, sampl_func)

    result_points = collocator(points)
    result_mesh = collocator(mesh)
    result_point = collocator(point)
    assert all_almost_equal(result_points, true_result_points)
    assert all_almost_equal(result_mesh, true_result_mesh)
    assert all_almost_equal(result_point, true_result_point)
    assert result_points.flags.writeable
    assert result_mesh.flags.writeable
    assert result_point.flags.writeable

    out_points = np.empty(values_points_shape, dtype='float64')
    out_mesh = np.empty(values_mesh_shape, dtype='float64')
    out_point = np.empty(value_point_shape, dtype='float64')
    collocator(points, out=out_points)
    collocator(mesh, out=out_mesh)
    collocator(point, out=out_point)
    assert all_almost_equal(out_points, true_result_points)
    assert all_almost_equal(out_mesh, true_result_mesh)
    assert all_almost_equal(out_point, true_result_point)


def test_fspace_elem_eval_unusual_dtypes():
    """Check evaluation with unusual data types (int and string)."""
    domain = odl.Strings(3)
    strings = np.array(['aa', 'b', 'cab', 'aba'])
    out_vec = np.empty((4,), dtype='int64')

    # Can be vectorized for arrays only
    sampl_func = make_func_for_sampling(
        lambda s: np.array([str(si).count('a') for si in s]),
        domain,
        out_dtype='int64'
    )
    collocator = partial(point_collocation, sampl_func)

    true_values = [2, 0, 1, 2]

    assert collocator('abc') == 1
    assert all_equal(collocator(strings), true_values)
    collocator(strings, out=out_vec)
    assert all_equal(out_vec, true_values)


def test_fspace_elem_eval_vec_1d(func_vec_1d):
    """Test evaluation in 1d since it's a corner case regarding shapes."""
    domain = odl.IntervalProd(0, 1)
    points = _points(domain, 3)
    mesh_shape = (4,)
    mesh = _meshgrid(domain, mesh_shape)
    point1 = 0.5
    point2 = [0.5]
    values_points_shape = (2, 3)
    values_mesh_shape = (2, 4)
    value_point_shape = (2,)

    func_ref, func = func_vec_1d

    true_result_points = np.array(func_ref(points))
    true_result_mesh = np.array(func_ref(mesh))
    true_result_point = np.array(func_ref(np.array([point1]))).squeeze()

    sampl_func = make_func_for_sampling(
        func, domain, out_dtype=('float64', (2,))
    )
    collocator = partial(point_collocation, sampl_func)

    result_points = collocator(points)
    result_mesh = collocator(mesh)
    result_point1 = collocator(point1)
    result_point2 = collocator(point2)
    assert all_almost_equal(result_points, true_result_points)
    assert all_almost_equal(result_mesh, true_result_mesh)
    assert all_almost_equal(result_point1, true_result_point)
    assert all_almost_equal(result_point2, true_result_point)

    out_points = np.empty(values_points_shape, dtype='float64')
    out_mesh = np.empty(values_mesh_shape, dtype='float64')
    out_point1 = np.empty(value_point_shape, dtype='float64')
    out_point2 = np.empty(value_point_shape, dtype='float64')
    collocator(points, out=out_points)
    collocator(mesh, out=out_mesh)
    collocator(point1, out=out_point1)
    collocator(point2, out=out_point2)
    assert all_almost_equal(out_points, true_result_points)
    assert all_almost_equal(out_mesh, true_result_mesh)
    assert all_almost_equal(out_point1, true_result_point)
    assert all_almost_equal(out_point2, true_result_point)


# --- interpolation tests --- #


def test_nearest_interpolation_1d_complex():
    """Test nearest neighbor interpolation in 1d with complex values."""
    coord_vecs = [[0.1, 0.3, 0.5, 0.7, 0.9]]
    f = np.array([0 + 1j, 1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j], dtype="complex128")
    interpolator = nearest_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator(0.35)  # closest to index 1 -> 1 + 2j
    assert val == 1.0 + 2.0j
    # Input array, with and without output array
    pts = np.array([0.39, 0.0, 0.65, 0.95])
    true_arr = [1 + 2j, 0 + 1j, 3 + 4j, 4 + 5j]
    assert all_equal(interpolator(pts), true_arr)
    # Should also work with a (1, N) array
    pts = pts[None, :]
    assert all_equal(interpolator(pts), true_arr)
    out = np.empty(4, dtype='complex128')
    interpolator(pts, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    # Same as array for 1d
    mg = sparse_meshgrid([0.39, 0.0, 0.65, 0.95])
    true_mg = [1 + 2j, 0 + 1j, 3 + 4j, 4 + 5j]
    assert all_equal(interpolator(mg), true_mg)
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_nearest_interpolation_2d():
    """Test nearest neighbor interpolation in 2d."""
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    f = np.array([[0, 1],
                  [2, 3],
                  [4, 5],
                  [6, 7]], dtype="float64")
    interpolator = nearest_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator([0.3, 0.6])  # closest to index (1, 1) -> 3
    assert val == 3.0
    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [1.0, 1.0]])
    true_arr = [3, 7]
    assert all_equal(interpolator(pts.T), true_arr)
    out = np.empty(2, dtype='float64')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0])
    # Indices: (1, 3) x (0, 1)
    true_mg = [[2, 3],
               [6, 7]]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_nearest_interpolation_2d_string():
    """Test nearest neighbor interpolation in 2d with string values."""
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    f = np.array([['m', 'y'],
                  ['s', 't'],
                  ['r', 'i'],
                  ['n', 'g']], dtype='U1')
    interpolator = nearest_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator([0.3, 0.6])  # closest to index (1, 1) -> 3
    assert val == u't'
    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [1.0, 1.0]])
    true_arr = np.array(['t', 'g'], dtype='U1')
    assert all_equal(interpolator(pts.T), true_arr)
    out = np.empty(2, dtype='U1')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0])
    # Indices: (1, 3) x (0, 1)
    true_mg = np.array([['s', 't'],
                        ['n', 'g']], dtype='U1')
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='U1')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_linear_interpolation_1d():
    """Test linear interpolation in 1d."""
    coord_vecs = [[0.1, 0.3, 0.5, 0.7, 0.9]]
    f = np.array([1, 2, 3, 4, 5], dtype="float64")
    interpolator = linear_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator(0.35)
    true_val = 0.75 * 2 + 0.25 * 3
    assert val == pytest.approx(true_val)

    # Input array, with and without output array
    pts = np.array([0.4, 0.0, 0.65, 0.95])
    true_arr = [2.5, 0.5, 3.75, 3.75]
    assert all_almost_equal(interpolator(pts), true_arr)


def test_linear_interpolation_2d():
    """Test linear interpolation in 2d."""
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    f = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]], dtype='float64')
    interpolator = linear_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator([0.3, 0.6])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val = (
        (1 - l1) * (1 - l2) * f[0, 0]
        + (1 - l1) * l2 * f[0, 1]
        + l1 * (1 - l2) * f[1, 0]
        + l1 * l2 * f[1, 1]
    )
    assert val == pytest.approx(true_val)

    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [0.1, 0.25],
                    [1.0, 1.0]])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val_1 = (
        (1 - l1) * (1 - l2) * f[0, 0]
        + (1 - l1) * l2 * f[0, 1]
        + l1 * (1 - l2) * f[1, 0]
        + l1 * l2 * f[1, 1]
    )
    l1 = (0.125 - 0.1) / (0.375 - 0.125)
    # l2 = 0
    true_val_2 = (1 - l1) * f[0, 0]  # only lower left contributes
    l1 = (1.0 - 0.875) / (0.875 - 0.625)
    l2 = (1.0 - 0.75) / (0.75 - 0.25)
    true_val_3 = (1 - l1) * (1 - l2) * f[3, 1]  # lower left only
    true_arr = [true_val_1, true_val_2, true_val_3]
    assert all_equal(interpolator(pts.T), true_arr)

    out = np.empty(3, dtype='float64')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)

    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 0.75])
    # Indices: (1, 3) x (0, 1)
    lx1 = (0.3 - 0.125) / (0.375 - 0.125)
    lx2 = (1.0 - 0.875) / (0.875 - 0.625)
    ly1 = (0.4 - 0.25) / (0.75 - 0.25)
    # ly2 = 0
    true_val_11 = (
        (1 - lx1) * (1 - ly1) * f[0, 0]
        + (1 - lx1) * ly1 * f[0, 1]
        + lx1 * (1 - ly1) * f[1, 0]
        + lx1 * ly1 * f[1, 1]
    )
    true_val_12 = (
        (1 - lx1) * f[0, 1]
        + lx1 * f[1, 1]  # ly2 = 0
    )
    true_val_21 = (
        (1 - lx2) * (1 - ly1) * f[3, 0]
        + (1 - lx2) * ly1 * f[3, 1]   # high node 1.0, no upper
    )
    true_val_22 = (1 - lx2) * f[3, 1]  # ly2 = 0, no upper for 1.0
    true_mg = [[true_val_11, true_val_12],
               [true_val_21, true_val_22]]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_per_axis_interpolation():
    """Test different interpolation schemes per axis."""
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    interp = ['linear', 'nearest']
    f = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]], dtype='float64')
    interpolator = per_axis_interpolator(f, coord_vecs, interp)

    # Evaluate at single point
    val = interpolator([0.3, 0.5])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    # 0.5 equally far from both neighbors -> NN chooses 0.75
    true_val = (1 - l1) * f[0, 1] + l1 * f[1, 1]
    assert val == pytest.approx(true_val)

    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [0.1, 0.25],
                    [1.0, 1.0]])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    true_val_1 = (1 - l1) * f[0, 1] + l1 * f[1, 1]
    l1 = (0.125 - 0.1) / (0.375 - 0.125)
    true_val_2 = (1 - l1) * f[0, 0]  # only lower left contributes
    l1 = (1.0 - 0.875) / (0.875 - 0.625)
    true_val_3 = (1 - l1) * f[3, 1]  # lower left only
    true_arr = [true_val_1, true_val_2, true_val_3]
    assert all_equal(interpolator(pts.T), true_arr)

    out = np.empty(3, dtype='float64')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)

    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 0.85])
    # Indices: (1, 3) x (0, 1)
    lx1 = (0.3 - 0.125) / (0.375 - 0.125)
    lx2 = (1.0 - 0.875) / (0.875 - 0.625)
    true_val_11 = (1 - lx1) * f[0, 0] + lx1 * f[1, 0]
    true_val_12 = ((1 - lx1) * f[0, 1] + lx1 * f[1, 1])
    true_val_21 = (1 - lx2) * f[3, 0]
    true_val_22 = (1 - lx2) * f[3, 1]
    true_mg = [[true_val_11, true_val_12],
               [true_val_21, true_val_22]]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_collocation_interpolation_identity():
    """Check if collocation is left-inverse to interpolation."""
    # Interpolation followed by collocation on the same grid should be
    # the identity
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    f = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]], dtype='float64')
    interpolators = [
        nearest_interpolator(f, coord_vecs),
        linear_interpolator(f, coord_vecs),
        per_axis_interpolator(f, coord_vecs, interp=['linear', 'nearest']),
    ]

    for interpolator in interpolators:
        mg = sparse_meshgrid(*coord_vecs)
        ident_f = point_collocation(interpolator, mg)
        assert all_almost_equal(ident_f, f)


if __name__ == '__main__':
    odl.util.test_file(__file__)
