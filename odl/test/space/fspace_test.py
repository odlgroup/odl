# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import inspect
import numpy as np
import pytest
import sys

import odl
from odl import FunctionSpace
from odl.discr.grid import sparse_meshgrid
from odl.util.testutils import all_almost_equal, all_equal, simple_fixture


# --- Helper functions --- #


PY2 = sys.version_info.major < 3
getargspec = inspect.getargspec if PY2 else inspect.getfullargspec


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


def _standard_setup_2d():
    rect = odl.IntervalProd([0, 0], [1, 2])
    points = _points(rect, num=5)
    mg = _meshgrid(rect, shape=(2, 3))
    return rect, points, mg


class FuncList(list):  # So we can set __name__
    pass


# --- pytest fixtures (general) --- #


out_dtype_params = ['float32', 'float64', 'complex64']
out_dtype = simple_fixture('out_dtype', out_dtype_params,
                           fmt=' {name} = {value!r} ')

out_shape = simple_fixture('out_shape', [(), (2,), (2, 3)])
domain_ndim = simple_fixture('domain_ndim', [1, 2])
vectorized = simple_fixture('vectorized', [True, False])
a = simple_fixture('a', [0.0, 1.0, -2.0])
b = simple_fixture('b', [0.0, 1.0, -2.0])
power = simple_fixture('power', [3, 1.0, 0.5, -2.0])


@pytest.fixture(scope='module')
def fspace_scal(domain_ndim, out_dtype):
    """Fixture returning a function space with given properties."""
    domain = odl.IntervalProd([0] * domain_ndim, [1] * domain_ndim)
    return FunctionSpace(domain, out_dtype=out_dtype)


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


func_1d_params = [(func_1d_ref, func_1d_oop), (func_1d_ref, func_1d_ip)]
func_1d_params.append((lambda x: -x[0], np.negative))
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


# --- FunctionSpace tests --- #


def test_fspace_init():
    """Check if all initialization patterns work."""
    intv = odl.IntervalProd(0, 1)
    FunctionSpace(intv)
    FunctionSpace(intv, out_dtype=float)
    FunctionSpace(intv, out_dtype=complex)
    FunctionSpace(intv, out_dtype=(float, (2, 3)))

    str3 = odl.Strings(3)
    FunctionSpace(str3, out_dtype=int)

    # Make sure repr shows something
    assert repr(FunctionSpace(intv, out_dtype=(float, (2, 3))))


def test_fspace_attributes():
    """Check attribute access and correct values."""
    intv = odl.IntervalProd(0, 1)

    # Scalar-valued function spaces
    fspace = FunctionSpace(intv)
    fspace_r = FunctionSpace(intv, out_dtype=float)
    fspace_c = FunctionSpace(intv, out_dtype=complex)
    fspace_s = FunctionSpace(intv, out_dtype='U1')
    scalar_spaces = (fspace, fspace_r, fspace_c, fspace_s)

    assert fspace.domain == intv
    assert fspace.field == odl.RealNumbers()
    assert fspace_r.field == odl.RealNumbers()
    assert fspace_c.field == odl.ComplexNumbers()
    assert fspace_s.field is None

    assert fspace.out_dtype == float
    assert fspace_r.out_dtype == float
    assert fspace_r.real_out_dtype == float
    assert fspace_r.complex_out_dtype == complex
    assert fspace_c.out_dtype == complex
    assert fspace_c.real_out_dtype == float
    assert fspace_c.complex_out_dtype == complex
    assert fspace_s.out_dtype == np.dtype('U1')
    assert fspace_s.real_out_dtype is None

    assert all(spc.scalar_out_dtype == spc.out_dtype for spc in scalar_spaces)
    assert all(spc.out_shape == () for spc in scalar_spaces)
    assert all(not spc.tensor_valued for spc in scalar_spaces)

    # Vector-valued function space
    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))
    assert fspace_vec.field == odl.RealNumbers()
    assert fspace_vec.out_dtype == np.dtype((float, (2,)))
    assert fspace_vec.scalar_out_dtype == float
    assert fspace_vec.out_shape == (2,)
    assert fspace_vec.tensor_valued


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


def test_equals():
    """Test equality check and hash."""
    intv = odl.IntervalProd(0, 1)
    intv2 = odl.IntervalProd(-1, 1)
    fspace = FunctionSpace(intv)
    fspace_r = FunctionSpace(intv, out_dtype=float)
    fspace_c = FunctionSpace(intv, out_dtype=complex)
    fspace_intv2 = FunctionSpace(intv2)
    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))

    _test_eq(fspace, fspace)
    _test_eq(fspace, fspace_r)
    _test_eq(fspace_c, fspace_c)

    _test_neq(fspace, fspace_c)
    _test_neq(fspace, fspace_intv2)
    _test_neq(fspace_r, fspace_vec)


def test_fspace_astype():
    """Check that converting function spaces to new out_dtype works."""
    rspace = FunctionSpace(odl.IntervalProd(0, 1))
    cspace = FunctionSpace(odl.IntervalProd(0, 1), out_dtype=complex)
    rspace_s = FunctionSpace(odl.IntervalProd(0, 1), out_dtype='float32')
    cspace_s = FunctionSpace(odl.IntervalProd(0, 1), out_dtype='complex64')

    assert rspace.astype('complex64') == cspace_s
    assert rspace.astype('complex128') == cspace
    assert rspace.astype('complex128') is rspace.complex_space
    assert rspace.astype('float32') == rspace_s
    assert rspace.astype('float64') is rspace.real_space

    assert cspace.astype('float32') == rspace_s
    assert cspace.astype('float64') == rspace
    assert cspace.astype('float64') is cspace.real_space
    assert cspace.astype('complex64') == cspace_s
    assert cspace.astype('complex128') is cspace.complex_space


# --- FunctionSpaceElement tests --- #


def test_fspace_elem_vectorized_init(vectorized):
    """Check init of fspace elements with(out) vectorization."""
    intv = odl.IntervalProd(0, 1)

    fspace_scal = FunctionSpace(intv)
    fspace_scal.element(func_nd_oop, vectorized=vectorized)

    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))
    fspace_vec.element(func_vec_nd_oop, vectorized=vectorized)
    fspace_vec.element(func_nd_oop_seq, vectorized=vectorized)


def test_fspace_scal_elem_eval(fspace_scal, func_nd):
    """Check evaluation of scalar-valued function elements."""
    points = _points(fspace_scal.domain, 3)
    mesh_shape = tuple(range(2, 2 + fspace_scal.domain.ndim))
    mesh = _meshgrid(fspace_scal.domain, mesh_shape)
    point = [0.5] * fspace_scal.domain.ndim

    func_ref, func = func_nd

    true_values_points = func_ref(points)
    true_values_mesh = func_ref(mesh)
    true_value_point = func_ref(point)

    func_elem = fspace_scal.element(func)

    # Out of place
    result_points = func_elem(points)
    result_mesh = func_elem(mesh)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)
    assert result_points.dtype == fspace_scal.scalar_out_dtype
    assert result_mesh.dtype == fspace_scal.scalar_out_dtype

    # In place
    out_points = np.empty(3, dtype=fspace_scal.scalar_out_dtype)
    out_mesh = np.empty(mesh_shape, dtype=fspace_scal.scalar_out_dtype)
    func_elem(points, out=out_points)
    func_elem(mesh, out=out_mesh)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Single point evaluation
    result_point = func_elem(point)
    assert all_almost_equal(result_point, true_value_point)


def test_fspace_scal_elem_with_param_eval(func_param_nd):
    """Check evaluation of scalar-valued function elements with parameters."""
    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace_scal = FunctionSpace(intv)
    points = _points(fspace_scal.domain, 3)
    mesh_shape = (2, 3)
    mesh = _meshgrid(fspace_scal.domain, mesh_shape)

    func_ref, func = func_param_nd

    true_values_points = func_ref(points, c=2.5)
    true_values_mesh = func_ref(mesh, c=2.5)

    func_elem = fspace_scal.element(func)

    # Out of place
    result_points = func_elem(points, c=2.5)
    result_mesh = func_elem(mesh, c=2.5)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)

    # In place
    out_points = np.empty(3, dtype=fspace_scal.scalar_out_dtype)
    out_mesh = np.empty(mesh_shape, dtype=fspace_scal.scalar_out_dtype)
    func_elem(points, out=out_points, c=2.5)
    func_elem(mesh, out=out_mesh, c=2.5)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Complex output
    fspace_complex = FunctionSpace(intv, out_dtype=complex)
    true_values_points = func_ref(points, c=2j)
    true_values_mesh = func_ref(mesh, c=2j)

    func_elem = fspace_complex.element(func)

    result_points = func_elem(points, c=2j)
    result_mesh = func_elem(mesh, c=2j)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)


def test_fspace_vec_elem_eval(func_vec_nd, out_dtype):
    """Check evaluation of scalar-valued function elements."""
    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))
    points = _points(fspace_vec.domain, 3)
    mesh_shape = (2, 3)
    mesh = _meshgrid(fspace_vec.domain, mesh_shape)
    point = [0.5, 0.5]
    values_points_shape = (2, 3)
    values_mesh_shape = (2, 2, 3)

    func_ref, func = func_vec_nd

    true_values_points = func_ref(points)
    true_values_mesh = func_ref(mesh)
    true_value_point = func_ref(point)

    func_elem = fspace_vec.element(func)

    # Out of place
    result_points = func_elem(points)
    result_mesh = func_elem(mesh)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)
    assert result_points.dtype == fspace_vec.scalar_out_dtype
    assert result_mesh.dtype == fspace_vec.scalar_out_dtype

    # In place
    out_points = np.empty(values_points_shape,
                          dtype=fspace_vec.scalar_out_dtype)
    out_mesh = np.empty(values_mesh_shape,
                        dtype=fspace_vec.scalar_out_dtype)
    func_elem(points, out=out_points)
    func_elem(mesh, out=out_mesh)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Single point evaluation
    result_point = func_elem(point)
    assert all_almost_equal(result_point, true_value_point)
    out_point = np.empty((2,), dtype=fspace_vec.scalar_out_dtype)
    func_elem(point, out=out_point)
    assert all_almost_equal(out_point, true_value_point)


def test_fspace_tens_eval(func_tens):
    """Test tensor-valued function evaluation."""
    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace_tens = FunctionSpace(intv, out_dtype=(float, (2, 3)))
    points = _points(fspace_tens.domain, 4)
    mesh_shape = (4, 5)
    mesh = _meshgrid(fspace_tens.domain, mesh_shape)
    point = [0.5, 0.5]
    values_points_shape = (2, 3, 4)
    values_mesh_shape = (2, 3, 4, 5)
    value_point_shape = (2, 3)

    func_ref, func = func_tens

    true_result_points = np.array(func_ref(points))
    true_result_mesh = np.array(func_ref(mesh))
    true_result_point = np.array(func_ref(np.array(point)[:, None])).squeeze()

    func_elem = fspace_tens.element(func)

    result_points = func_elem(points)
    result_mesh = func_elem(mesh)
    result_point = func_elem(point)
    assert all_almost_equal(result_points, true_result_points)
    assert all_almost_equal(result_mesh, true_result_mesh)
    assert all_almost_equal(result_point, true_result_point)

    out_points = np.empty(values_points_shape, dtype=float)
    out_mesh = np.empty(values_mesh_shape, dtype=float)
    out_point = np.empty(value_point_shape, dtype=float)
    func_elem(points, out=out_points)
    func_elem(mesh, out=out_mesh)
    func_elem(point, out=out_point)
    assert all_almost_equal(out_points, true_result_points)
    assert all_almost_equal(out_mesh, true_result_mesh)
    assert all_almost_equal(out_point, true_result_point)


def test_fspace_elem_eval_unusual_dtypes():
    """Check evaluation with unusual data types."""
    str3 = odl.Strings(3)
    fspace = FunctionSpace(str3, out_dtype=int)
    strings = np.array(['aa', 'b', 'cab', 'aba'])
    out_vec = np.empty((4,), dtype=int)

    # Vectorized for arrays only
    func_elem = fspace.element(
        lambda s: np.array([str(si).count('a') for si in s]))
    true_values = [2, 0, 1, 2]

    assert func_elem('abc') == 1
    assert all_equal(func_elem(strings), true_values)
    func_elem(strings, out=out_vec)
    assert all_equal(out_vec, true_values)


def test_fspace_elem_eval_vec_1d(func_vec_1d):
    """Test evaluation in 1d since it's a corner case regarding shapes."""
    intv = odl.IntervalProd(0, 1)
    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))
    points = _points(fspace_vec.domain, 3)
    mesh_shape = (4,)
    mesh = _meshgrid(fspace_vec.domain, mesh_shape)
    point1 = 0.5
    point2 = [0.5]
    values_points_shape = (2, 3)
    values_mesh_shape = (2, 4)
    value_point_shape = (2,)

    func_ref, func = func_vec_1d

    true_result_points = np.array(func_ref(points))
    true_result_mesh = np.array(func_ref(mesh))
    true_result_point = np.array(func_ref(np.array([point1]))).squeeze()

    func_elem = fspace_vec.element(func)

    result_points = func_elem(points)
    result_mesh = func_elem(mesh)
    result_point1 = func_elem(point1)
    result_point2 = func_elem(point2)
    assert all_almost_equal(result_points, true_result_points)
    assert all_almost_equal(result_mesh, true_result_mesh)
    assert all_almost_equal(result_point1, true_result_point)
    assert all_almost_equal(result_point2, true_result_point)

    out_points = np.empty(values_points_shape, dtype=float)
    out_mesh = np.empty(values_mesh_shape, dtype=float)
    out_point1 = np.empty(value_point_shape, dtype=float)
    out_point2 = np.empty(value_point_shape, dtype=float)
    func_elem(points, out=out_points)
    func_elem(mesh, out=out_mesh)
    func_elem(point1, out=out_point1)
    func_elem(point2, out=out_point2)
    assert all_almost_equal(out_points, true_result_points)
    assert all_almost_equal(out_mesh, true_result_mesh)
    assert all_almost_equal(out_point1, true_result_point)
    assert all_almost_equal(out_point2, true_result_point)


def test_fspace_elem_equality():
    """Test equality check of fspace elements."""
    intv = odl.IntervalProd(0, 1)
    fspace = FunctionSpace(intv)

    f_novec = fspace.element(func_nd_oop, vectorized=False)

    f_vec_oop = fspace.element(func_nd_oop, vectorized=True)
    f_vec_oop_2 = fspace.element(func_nd_oop, vectorized=True)

    f_vec_ip = fspace.element(func_nd_ip, vectorized=True)
    f_vec_ip_2 = fspace.element(func_nd_ip, vectorized=True)

    f_vec_dual = fspace.element(func_nd_dual, vectorized=True)
    f_vec_dual_2 = fspace.element(func_nd_dual, vectorized=True)

    assert f_novec == f_novec
    assert f_novec != f_vec_oop
    assert f_novec != f_vec_ip
    assert f_novec != f_vec_dual

    assert f_vec_oop == f_vec_oop
    assert f_vec_oop == f_vec_oop_2
    assert f_vec_oop != f_vec_ip
    assert f_vec_oop != f_vec_dual

    assert f_vec_ip == f_vec_ip
    assert f_vec_ip == f_vec_ip_2
    assert f_vec_ip != f_vec_dual

    assert f_vec_dual == f_vec_dual
    assert f_vec_dual == f_vec_dual_2

    fspace_tens = FunctionSpace(intv, out_dtype=(float, (2, 3)))

    f_tens_oop = fspace_tens.element(func_tens_oop)
    f_tens_oop2 = fspace_tens.element(func_tens_oop)

    f_tens_ip = fspace_tens.element(func_tens_ip)
    f_tens_ip2 = fspace_tens.element(func_tens_ip)

    f_tens_seq = fspace_tens.element(func_tens_oop_seq)
    f_tens_seq2 = fspace_tens.element(func_tens_oop_seq)

    assert f_tens_oop == f_tens_oop
    assert f_tens_oop == f_tens_oop2
    assert f_tens_oop != f_tens_ip
    assert f_tens_oop != f_tens_seq

    assert f_tens_ip == f_tens_ip
    assert f_tens_ip == f_tens_ip2
    assert f_tens_ip != f_tens_seq

    # Sequences are wrapped, will compare to not equal
    assert f_tens_seq == f_tens_seq
    assert f_tens_seq != f_tens_seq2


def test_fspace_elem_assign(out_shape):
    """Check assignment of fspace elements."""
    fspace = FunctionSpace(odl.IntervalProd(0, 1),
                           out_dtype=(float, out_shape))

    ndim = len(out_shape)
    if ndim == 0:
        f_oop = fspace.element(func_nd_oop)
        f_ip = fspace.element(func_nd_ip)
        f_dual = fspace.element(func_nd_dual)
    elif ndim == 1:
        f_oop = fspace.element(func_vec_nd_oop)
        f_ip = fspace.element(func_vec_nd_ip)
        f_dual = fspace.element(func_vec_nd_dual)
    elif ndim == 2:
        f_oop = fspace.element(func_tens_oop)
        f_ip = fspace.element(func_tens_ip)
        f_dual = fspace.element(func_tens_dual)
    else:
        assert False

    f_out = fspace.element()
    f_out.assign(f_oop)
    assert f_out == f_oop

    f_out = fspace.element()
    f_out.assign(f_ip)
    assert f_out == f_ip

    f_out = fspace.element()
    f_out.assign(f_dual)
    assert f_out == f_dual


def test_fspace_elem_copy(out_shape):
    """Check copying of fspace elements."""
    fspace = FunctionSpace(odl.IntervalProd(0, 1),
                           out_dtype=(float, out_shape))

    ndim = len(out_shape)
    if ndim == 0:
        f_oop = fspace.element(func_nd_oop)
        f_ip = fspace.element(func_nd_ip)
        f_dual = fspace.element(func_nd_dual)
    elif ndim == 1:
        f_oop = fspace.element(func_vec_nd_oop)
        f_ip = fspace.element(func_vec_nd_ip)
        f_dual = fspace.element(func_vec_nd_dual)
    elif ndim == 2:
        f_oop = fspace.element(func_tens_oop)
        f_ip = fspace.element(func_tens_ip)
        f_dual = fspace.element(func_tens_dual)
    else:
        assert False

    f_out = f_oop.copy()
    assert f_out == f_oop

    f_out = f_ip.copy()
    assert f_out == f_ip

    f_out = f_dual.copy()
    assert f_out == f_dual


def test_fspace_elem_real_imag_conj(out_shape):
    """Check taking real/imaginary parts of fspace elements."""
    fspace = FunctionSpace(odl.IntervalProd(0, 1),
                           out_dtype=(complex, out_shape))

    ndim = len(out_shape)
    if ndim == 0:
        f_elem = fspace.element(func_complex_nd_oop)
    elif ndim == 1:
        f_elem = fspace.element(func_vec_complex_nd_oop)
    elif ndim == 2:
        f_elem = fspace.element(func_tens_complex_oop)
    else:
        assert False

    points = _points(fspace.domain, 4)
    mesh_shape = (5,)
    mesh = _meshgrid(fspace.domain, mesh_shape)
    point = 0.5
    values_points_shape = out_shape + (4,)
    values_mesh_shape = out_shape + mesh_shape

    result_points = f_elem(points)
    result_point = f_elem(point)
    result_mesh = f_elem(mesh)

    assert all_almost_equal(f_elem.real(points), result_points.real)
    assert all_almost_equal(f_elem.real(point), result_point.real)
    assert all_almost_equal(f_elem.real(mesh), result_mesh.real)
    assert all_almost_equal(f_elem.imag(points), result_points.imag)
    assert all_almost_equal(f_elem.imag(point), result_point.imag)
    assert all_almost_equal(f_elem.imag(mesh), result_mesh.imag)
    assert all_almost_equal(f_elem.conj()(points), result_points.conj())
    assert all_almost_equal(f_elem.conj()(point), np.conj(result_point))
    assert all_almost_equal(f_elem.conj()(mesh), result_mesh.conj())

    out_points = np.empty(values_points_shape, dtype=float)
    out_mesh = np.empty(values_mesh_shape, dtype=float)

    f_elem.real(points, out=out_points)
    f_elem.real(mesh, out=out_mesh)

    assert all_almost_equal(out_points, result_points.real)
    assert all_almost_equal(out_mesh, result_mesh.real)

    f_elem.imag(points, out=out_points)
    f_elem.imag(mesh, out=out_mesh)

    assert all_almost_equal(out_points, result_points.imag)
    assert all_almost_equal(out_mesh, result_mesh.imag)

    out_points = np.empty(values_points_shape, dtype=complex)
    out_mesh = np.empty(values_mesh_shape, dtype=complex)

    f_elem.conj()(points, out=out_points)
    f_elem.conj()(mesh, out=out_mesh)

    assert all_almost_equal(out_points, result_points.conj())
    assert all_almost_equal(out_mesh, result_mesh.conj())


def test_fspace_zero(out_shape):
    """Check zero element."""

    fspace = FunctionSpace(odl.IntervalProd(0, 1),
                           out_dtype=(float, out_shape))

    points = _points(fspace.domain, 4)
    mesh_shape = (5,)
    mesh = _meshgrid(fspace.domain, mesh_shape)
    point = 0.5
    values_points_shape = out_shape + (4,)
    values_point_shape = out_shape
    values_mesh_shape = out_shape + mesh_shape

    f_zero = fspace.zero()

    assert all_equal(f_zero(points), np.zeros(values_points_shape))
    if not out_shape:
        assert f_zero(point) == 0.0
    else:
        assert all_equal(f_zero(point), np.zeros(values_point_shape))
    assert all_equal(f_zero(mesh), np.zeros(values_mesh_shape))

    out_points = np.empty(values_points_shape)
    out_mesh = np.empty(values_mesh_shape)

    f_zero(points, out=out_points)
    f_zero(mesh, out=out_mesh)

    assert all_equal(out_points, np.zeros(values_points_shape))
    assert all_equal(out_mesh, np.zeros(values_mesh_shape))


def test_fspace_one(out_shape):
    """Check one element."""

    fspace = FunctionSpace(odl.IntervalProd(0, 1),
                           out_dtype=(float, out_shape))

    points = _points(fspace.domain, 4)
    mesh_shape = (5,)
    mesh = _meshgrid(fspace.domain, mesh_shape)
    point = 0.5
    values_points_shape = out_shape + (4,)
    values_point_shape = out_shape
    values_mesh_shape = out_shape + mesh_shape

    f_one = fspace.one()

    assert all_equal(f_one(points), np.ones(values_points_shape))
    if not out_shape:
        assert f_one(point) == 1.0
    else:
        assert all_equal(f_one(point), np.ones(values_point_shape))
    assert all_equal(f_one(mesh), np.ones(values_mesh_shape))

    out_points = np.empty(values_points_shape)
    out_mesh = np.empty(values_mesh_shape)

    f_one(points, out=out_points)
    f_one(mesh, out=out_mesh)

    assert all_equal(out_points, np.ones(values_points_shape))
    assert all_equal(out_mesh, np.ones(values_mesh_shape))


def test_fspace_lincomb_scalar(a, b):
    """Check linear combination in function spaces."""
    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace = FunctionSpace(intv)
    points = _points(fspace.domain, 4)
    true_result = a * func_nd_oop(points) + b * func_nd_bcast_ref(points)

    # Note: Special cases and alignment are tested later in the special methods

    # Non-vectorized
    f_elem1_novec = fspace.element(func_nd_oop, vectorized=False)
    f_elem2_novec = fspace.element(func_nd_bcast_oop, vectorized=False)
    out_novec = fspace.element(vectorized=False)
    fspace.lincomb(a, f_elem1_novec, b, f_elem2_novec, out_novec)
    assert all_equal(out_novec(points), true_result)
    out_arr = np.empty(4)
    out_novec(points, out=out_arr)
    assert all_equal(out_arr, true_result)

    # Vectorized
    f_elem1_oop = fspace.element(func_nd_oop)
    f_elem2_oop = fspace.element(func_nd_bcast_oop)
    out_oop = fspace.element()
    fspace.lincomb(a, f_elem1_oop, b, f_elem2_oop, out_oop)
    assert all_equal(out_oop(points), true_result)
    out_arr = np.empty(4)
    out_oop(points, out=out_arr)
    assert all_equal(out_arr, true_result)

    f_elem1_ip = fspace.element(func_nd_ip)
    f_elem2_ip = fspace.element(func_nd_bcast_ip)
    out_ip = fspace.element()
    fspace.lincomb(a, f_elem1_ip, b, f_elem2_ip, out_ip)
    assert all_equal(out_ip(points), true_result)
    out_arr = np.empty(4)
    out_ip(points, out=out_arr)
    assert all_equal(out_arr, true_result)

    f_elem1_dual = fspace.element(func_nd_dual)
    f_elem2_dual = fspace.element(func_nd_bcast_dual)
    out_dual = fspace.element()
    fspace.lincomb(a, f_elem1_dual, b, f_elem2_dual, out_dual)
    assert all_equal(out_dual(points), true_result)
    out_arr = np.empty(4)
    out_dual(points, out=out_arr)
    assert all_equal(out_arr, true_result)

    # Mix vectorized and non-vectorized
    out = fspace.element()
    fspace.lincomb(a, f_elem1_oop, b, f_elem2_novec, out)
    assert all_equal(out(points), true_result)
    out_arr = np.empty(4)
    out(points, out=out_arr)
    assert all_equal(out_arr, true_result)

    # Alignment options
    # out = a * out + b * f2, out = f1.copy() -> same as before
    out = f_elem1_oop.copy()
    fspace.lincomb(a, out, b, f_elem2_oop, out)
    true_result_aligned = true_result
    assert all_equal(out(points), true_result_aligned)

    # out = a * f1 + b * out, out = f2.copy() -> same as before
    out = f_elem2_oop.copy()
    fspace.lincomb(a, f_elem1_oop, b, out, out)
    true_result_aligned = true_result
    assert all_equal(out(points), true_result_aligned)

    # out = a * out + b * out
    out = f_elem1_oop.copy()
    fspace.lincomb(a, out, b, out, out)
    true_result_aligned = (a + b) * f_elem1_oop(points)
    assert all_equal(out(points), true_result_aligned)

    # out = a * f1 + b * f1
    out = fspace.element()
    fspace.lincomb(a, f_elem1_oop, b, f_elem1_oop, out)
    true_result_aligned = (a + b) * f_elem1_oop(points)
    assert all_equal(out(points), true_result_aligned)


def test_fspace_lincomb_vec_tens(a, b, out_shape):
    """Check linear combination in function spaces."""
    if not out_shape:
        return

    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace = FunctionSpace(intv, out_dtype=(float, out_shape))
    points = _points(fspace.domain, 4)

    ndim = len(out_shape)
    if ndim == 1:
        f_elem1 = fspace.element(func_vec_nd_oop)
        f_elem2 = fspace.element(func_vec_nd_other)
        true_result = (a * func_vec_nd_ref(points) +
                       b * func_vec_nd_other(points))
    elif ndim == 2:
        f_elem1 = fspace.element(func_tens_oop)
        f_elem2 = fspace.element(func_tens_other)
        true_result = a * func_tens_ref(points) + b * func_tens_other(points)
    else:
        assert False

    out_func = fspace.element()
    fspace.lincomb(a, f_elem1, b, f_elem2, out_func)
    assert all_equal(out_func(points), true_result)
    out_arr = np.empty(out_shape + (4,))
    out_func(points, out=out_arr)
    assert all_equal(out_arr, true_result)


# NOTE: multiply and divide are tested via special methods


def test_fspace_elem_power(power, out_shape):
    """Check taking powers of fspace elements."""
    # Make sure test functions don't take negative values
    intv = odl.IntervalProd([1, 0], [2, 1])
    fspace = FunctionSpace(intv, out_dtype=(float, out_shape))
    points = _points(fspace.domain, 4)

    ndim = len(out_shape)
    if ndim == 0:
        f_elem = fspace.element(func_nd_oop)
        true_result = func_nd_ref(points) ** power
    elif ndim == 1:
        f_elem = fspace.element(func_vec_nd_oop)
        true_result = func_vec_nd_ref(points) ** power
    elif ndim == 2:
        f_elem = fspace.element(func_tens_oop)
        true_result = func_tens_ref(points) ** power
    else:
        assert False

    f_elem_pow = f_elem ** power
    assert all_almost_equal(f_elem_pow(points), true_result)
    out_arr = np.empty(out_shape + (4,))
    f_elem_pow(points, out_arr)
    assert all_almost_equal(out_arr, true_result)


def test_fspace_elem_arithmetic(arithmetic_op, out_shape):
    """Test arithmetic of fspace elements."""

    intv = odl.IntervalProd([1, 0], [2, 1])
    fspace = FunctionSpace(intv, out_dtype=(float, out_shape))
    points = _points(fspace.domain, 4)

    ndim = len(out_shape)
    if ndim == 0:
        f_elem1 = fspace.element(func_nd_oop)
        f_elem2 = fspace.element(func_nd_other)
    elif ndim == 1:
        f_elem1 = fspace.element(func_vec_nd_oop)
        f_elem2 = fspace.element(func_vec_nd_other)
    elif ndim == 2:
        f_elem1 = fspace.element(func_tens_oop)
        f_elem2 = fspace.element(func_tens_other)
    else:
        assert False

    result1 = f_elem1(points)
    result1_cpy = result1.copy()
    result2 = f_elem2(points)
    true_result_func = arithmetic_op(result1, result2)
    true_result_scal = arithmetic_op(result1_cpy, -2.0)

    f_elem1_cpy = f_elem1.copy()
    func_arith_func = arithmetic_op(f_elem1, f_elem2)
    func_arith_scal = arithmetic_op(f_elem1_cpy, -2.0)
    assert all_almost_equal(func_arith_func(points), true_result_func)
    assert all_almost_equal(func_arith_scal(points), true_result_scal)
    out_arr_func = np.empty(out_shape + (4,))
    out_arr_scal = np.empty(out_shape + (4,))
    func_arith_func(points, out=out_arr_func)
    func_arith_scal(points, out=out_arr_scal)
    assert all_almost_equal(out_arr_func, true_result_func)
    assert all_almost_equal(out_arr_scal, true_result_scal)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
