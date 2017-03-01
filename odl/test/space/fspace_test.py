# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import numpy as np
import pytest

import odl
from odl import FunctionSpace
from odl.discr.grid import sparse_meshgrid
from odl.util.testutils import (all_almost_equal, all_equal, almost_equal,
                                simple_fixture)


def test_fspace_init():
    intv = odl.IntervalProd(0, 1)
    FunctionSpace(intv)
    FunctionSpace(intv, range=odl.RealNumbers())
    FunctionSpace(intv, range=odl.ComplexNumbers())

    rect = odl.IntervalProd([0, 0], [1, 2])
    FunctionSpace(rect)
    FunctionSpace(rect, range=odl.RealNumbers())
    FunctionSpace(rect, range=odl.ComplexNumbers())

    cube = odl.IntervalProd([0, 0, 0], [1, 2, 3])
    FunctionSpace(cube)
    FunctionSpace(cube, range=odl.RealNumbers())
    FunctionSpace(cube, range=odl.ComplexNumbers())

    ndbox = odl.IntervalProd([0] * 10, np.arange(1, 11))
    FunctionSpace(ndbox)
    FunctionSpace(ndbox, range=odl.RealNumbers())
    FunctionSpace(ndbox, range=odl.ComplexNumbers())

    str3 = odl.Strings(3)
    ints = odl.Integers()
    FunctionSpace(str3, range=ints)


def test_fspace_simple_attributes():
    intv = odl.IntervalProd(0, 1)
    fspace = FunctionSpace(intv)
    fspace_r = FunctionSpace(intv, range=odl.RealNumbers())
    fspace_c = FunctionSpace(intv, range=odl.ComplexNumbers())

    assert fspace.domain == intv
    assert fspace.range == odl.RealNumbers()
    assert fspace_r.range == odl.RealNumbers()
    assert fspace_c.range == odl.ComplexNumbers()


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
    fspace_r = FunctionSpace(intv, range=odl.RealNumbers())
    fspace_c = FunctionSpace(intv, range=odl.ComplexNumbers())
    fspace_intv2 = FunctionSpace(intv2)

    _test_eq(fspace, fspace)
    _test_eq(fspace, fspace_r)
    _test_eq(fspace_c, fspace_c)

    _test_neq(fspace, fspace_c)
    _test_neq(fspace, fspace_intv2)


def _points(domain, num):
    min_pt = domain.min_pt
    max_pt = domain.max_pt
    ndim = domain.ndim
    points = np.random.uniform(low=0, high=1, size=(ndim, num))
    for i in range(ndim):
        points[i, :] = min_pt[i] + (max_pt[i] - min_pt[i]) * points[i]
    return points


def _meshgrid(domain, shape):
    min_pt = domain.min_pt
    max_pt = domain.max_pt
    ndim = domain.ndim
    coord_vecs = []
    for i in range(ndim):
        vec = np.random.uniform(low=min_pt[i], high=max_pt[i], size=shape[i])
        vec.sort()
        coord_vecs.append(vec)
    return sparse_meshgrid(*coord_vecs)


def test_fspace_vector_init():
    # 1d, real
    intv = odl.IntervalProd(0, 1)
    fspace = FunctionSpace(intv)
    fspace.element(func_1d_oop)
    fspace.element(func_1d_oop, vectorized=False)
    fspace.element(func_1d_oop, vectorized=True)
    fspace.element(func_1d_ip, vectorized=True)
    fspace.element(func_1d_dual, vectorized=True)

    # 2d, real
    rect = odl.IntervalProd([0, 0], [1, 2])
    fspace = FunctionSpace(rect)
    fspace.element(func_2d_novec, vectorized=False)
    fspace.element(func_2d_vec_oop)
    fspace.element(func_2d_vec_oop, vectorized=True)
    fspace.element(func_2d_vec_ip, vectorized=True)
    fspace.element(func_2d_vec_dual, vectorized=True)

    # 2d, complex
    fspace = FunctionSpace(rect, range=odl.ComplexNumbers())
    fspace.element(cfunc_2d_novec, vectorized=False)
    fspace.element(cfunc_2d_vec_oop)
    fspace.element(cfunc_2d_vec_oop, vectorized=True)
    fspace.element(cfunc_2d_vec_ip, vectorized=True)
    fspace.element(cfunc_2d_vec_dual, vectorized=True)


def test_fspace_vector_eval():
    str3 = odl.Strings(3)
    ints = odl.Integers()
    fspace = FunctionSpace(str3, ints)
    strings = np.array(['aa', 'b', 'cab', 'aba'])
    out_vec = np.empty((4,), dtype=int)

    # Vectorized for arrays only
    f_vec = fspace.element(lambda s: np.array([str(si).count('a')
                                               for si in s]))
    true_vec = [2, 0, 1, 2]

    assert f_vec('abc') == 1
    assert all_equal(f_vec(strings), true_vec)
    f_vec(strings, out=out_vec)
    assert all_equal(out_vec, true_vec)


def _standard_setup_2d():
    rect = odl.IntervalProd([0, 0], [1, 2])
    points = _points(rect, num=5)
    mg = _meshgrid(rect, shape=(2, 3))
    return rect, points, mg


def test_fspace_out_dtype():
    rect = odl.IntervalProd([0, 0], [3, 5])
    points = np.array([[0, 1], [0, 3], [3, 4], [2, 5]], dtype='int').T
    vec1 = np.array([0, 1, 3])[:, None]
    vec2 = np.array([1, 2, 4, 5])[None, :]
    mg = (vec1, vec2)

    true_arr = func_2d_vec_oop(points)
    true_mg = func_2d_vec_oop(mg)

    fspace = FunctionSpace(rect, out_dtype='int')
    f_vec = fspace.element(func_2d_vec_oop)

    assert all_equal(f_vec(points), true_arr)
    assert all_equal(f_vec(mg), true_mg)
    assert f_vec(points).dtype == np.dtype('int')
    assert f_vec(mg).dtype == np.dtype('int')


def test_fspace_astype():

    rspace = FunctionSpace(odl.IntervalProd(0, 1))
    cspace = FunctionSpace(odl.IntervalProd(0, 1), range=odl.ComplexNumbers())
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


def test_fspace_vector_eval_real():
    rect, points, mg = _standard_setup_2d()

    fspace = FunctionSpace(rect)
    f_novec = fspace.element(func_2d_novec, vectorized=False)
    f_vec_oop = fspace.element(func_2d_vec_oop, vectorized=True)
    f_vec_ip = fspace.element(func_2d_vec_ip, vectorized=True)
    f_vec_dual = fspace.element(func_2d_vec_dual, vectorized=True)

    true_arr = func_2d_vec_oop(points)
    true_mg = func_2d_vec_oop(mg)

    # Out-of-place
    assert f_novec([0.5, 1.5]) == func_2d_novec([0.5, 1.5])
    assert f_vec_oop([0.5, 1.5]) == func_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_oop(points), true_arr)
    assert all_equal(f_vec_oop(mg), true_mg)

    # In-place standard implementation
    out_arr = np.empty((5,), dtype='float64')
    out_mg = np.empty((2, 3), dtype='float64')

    f_vec_oop(points, out=out_arr)
    f_vec_oop(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    with pytest.raises(TypeError):  # ValueError: invalid vectorized input
        f_vec_oop(points[0])

    # In-place-only
    out_arr = np.empty((5,), dtype='float64')
    out_mg = np.empty((2, 3), dtype='float64')

    f_vec_ip(points, out=out_arr)
    f_vec_ip(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # Standard out-of-place evaluation
    assert f_vec_ip([0.5, 1.5]) == func_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_ip(points), true_arr)
    assert all_equal(f_vec_ip(mg), true_mg)

    # Dual use
    assert f_vec_dual([0.5, 1.5]) == func_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_dual(points), true_arr)
    assert all_equal(f_vec_dual(mg), true_mg)

    out_arr = np.empty((5,), dtype='float64')
    out_mg = np.empty((2, 3), dtype='float64')

    f_vec_dual(points, out=out_arr)
    f_vec_dual(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)


def test_fspace_vector_eval_complex():
    rect, points, mg = _standard_setup_2d()

    fspace = FunctionSpace(rect, range=odl.ComplexNumbers())
    f_novec = fspace.element(cfunc_2d_novec, vectorized=False)
    f_vec_oop = fspace.element(cfunc_2d_vec_oop, vectorized=True)
    f_vec_ip = fspace.element(cfunc_2d_vec_ip, vectorized=True)
    f_vec_dual = fspace.element(cfunc_2d_vec_dual, vectorized=True)

    true_arr = cfunc_2d_vec_oop(points)
    true_mg = cfunc_2d_vec_oop(mg)

    # Out-of-place
    assert f_novec([0.5, 1.5]) == cfunc_2d_novec([0.5, 1.5])
    assert f_vec_oop([0.5, 1.5]) == cfunc_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_oop(points), true_arr)
    assert all_equal(f_vec_oop(mg), true_mg)

    # In-place standard implementation
    out_arr = np.empty((5,), dtype='complex128')
    out_mg = np.empty((2, 3), dtype='complex128')

    f_vec_oop(points, out=out_arr)
    f_vec_oop(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    with pytest.raises(TypeError):  # ValueError: invalid vectorized input
        f_vec_oop(points[0])

    # In-place-only
    out_arr = np.empty((5,), dtype='complex128')
    out_mg = np.empty((2, 3), dtype='complex128')

    f_vec_ip(points, out=out_arr)
    f_vec_ip(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # Standard out-of-place evaluation
    assert f_vec_ip([0.5, 1.5]) == cfunc_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_ip(points), true_arr)
    assert all_equal(f_vec_ip(mg), true_mg)

    # Dual use
    assert f_vec_dual([0.5, 1.5]) == cfunc_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_dual(points), true_arr)
    assert all_equal(f_vec_dual(mg), true_mg)

    out_arr = np.empty((5,), dtype='complex128')
    out_mg = np.empty((2, 3), dtype='complex128')

    f_vec_dual(points, out=out_arr)
    f_vec_dual(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)


def test_fspace_vector_with_params():
    rect, points, mg = _standard_setup_2d()

    def f(x, c):
        return sum(x) + c

    def f_out1(x, out, c):
        out[:] = sum(x) + c

    def f_out2(x, c, out):
        out[:] = sum(x) + c

    fspace = FunctionSpace(rect)
    true_result_arr = f(points, c=2)
    true_result_mg = f(mg, c=2)

    f_elem = fspace.element(f)
    assert all_equal(f_elem(points, c=2), true_result_arr)
    out_arr = np.empty((5,))
    f_elem(points, c=2, out=out_arr)
    assert all_equal(out_arr, true_result_arr)
    assert all_equal(f_elem(mg, c=2), true_result_mg)
    out_mg = np.empty((2, 3))
    f_elem(mg, c=2, out=out_mg)
    assert all_equal(out_mg, true_result_mg)

    f_out1_elem = fspace.element(f_out1)
    assert all_equal(f_out1_elem(points, c=2), true_result_arr)
    out_arr = np.empty((5,))
    f_out1_elem(points, c=2, out=out_arr)
    assert all_equal(out_arr, true_result_arr)
    assert all_equal(f_out1_elem(mg, c=2), true_result_mg)
    out_mg = np.empty((2, 3))
    f_out1_elem(mg, c=2, out=out_mg)
    assert all_equal(out_mg, true_result_mg)

    f_out2_elem = fspace.element(f_out2)
    assert all_equal(f_out2_elem(points, c=2), true_result_arr)
    out_arr = np.empty((5,))
    f_out2_elem(points, c=2, out=out_arr)
    assert all_equal(out_arr, true_result_arr)
    assert all_equal(f_out2_elem(mg, c=2), true_result_mg)
    out_mg = np.empty((2, 3))
    f_out2_elem(mg, c=2, out=out_mg)
    assert all_equal(out_mg, true_result_mg)


def test_fspace_vector_ufunc():
    intv = odl.IntervalProd(0, 1)
    points = _points(intv, num=5)
    mg = _meshgrid(intv, shape=(5,))

    fspace = FunctionSpace(intv)
    f_vec = fspace.element(np.sin)

    assert f_vec(0.5) == np.sin(0.5)
    assert all_equal(f_vec(points), np.sin(points.squeeze()))
    assert all_equal(f_vec(mg), np.sin(mg[0]))


def test_fspace_vector_equality():
    rect = odl.IntervalProd([0, 0], [1, 2])
    fspace = FunctionSpace(rect)

    f_novec = fspace.element(func_2d_novec, vectorized=False)

    f_vec_oop = fspace.element(func_2d_vec_oop, vectorized=True)
    f_vec_oop_2 = fspace.element(func_2d_vec_oop, vectorized=True)

    f_vec_ip = fspace.element(func_2d_vec_ip, vectorized=True)
    f_vec_ip_2 = fspace.element(func_2d_vec_ip, vectorized=True)

    f_vec_dual = fspace.element(func_2d_vec_dual, vectorized=True)
    f_vec_dual_2 = fspace.element(func_2d_vec_dual, vectorized=True)

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


def test_fspace_vector_assign():
    fspace = FunctionSpace(odl.IntervalProd(0, 1))

    f_novec = fspace.element(func_1d_oop, vectorized=False)
    f_vec_ip = fspace.element(func_1d_ip, vectorized=True)
    f_vec_dual = fspace.element(func_1d_dual, vectorized=True)

    f_out = fspace.element()
    f_out.assign(f_novec)
    assert f_out == f_novec

    f_out = fspace.element()
    f_out.assign(f_vec_ip)
    assert f_out == f_vec_ip

    f_out = fspace.element()
    f_out.assign(f_vec_dual)
    assert f_out == f_vec_dual


def test_fspace_vector_copy():
    fspace = FunctionSpace(odl.IntervalProd(0, 1))

    f_novec = fspace.element(func_1d_oop, vectorized=False)
    f_vec_ip = fspace.element(func_1d_ip, vectorized=True)
    f_vec_dual = fspace.element(func_1d_dual, vectorized=True)

    f_out = f_novec.copy()
    assert f_out == f_novec

    f_out = f_vec_ip.copy()
    assert f_out == f_vec_ip

    f_out = f_vec_dual.copy()
    assert f_out == f_vec_dual


def test_fspace_vector_real_imag():
    rect, _, mg = _standard_setup_2d()
    cspace = FunctionSpace(rect, range=odl.ComplexNumbers())
    f = cspace.element(cfunc_2d_vec_oop)

    # real / imag on complex functions
    assert all_equal(f.real(mg), cfunc_2d_vec_oop(mg).real)
    assert all_equal(f.imag(mg), cfunc_2d_vec_oop(mg).imag)
    out_mg = np.empty((2, 3))
    f.real(mg, out=out_mg)
    assert all_equal(out_mg, cfunc_2d_vec_oop(mg).real)
    f.imag(mg, out=out_mg)
    assert all_equal(out_mg, cfunc_2d_vec_oop(mg).imag)

    # real / imag on real functions, should be the function itself / zero
    rspace = FunctionSpace(rect)
    f = rspace.element(func_2d_vec_oop)
    assert all_equal(f.real(mg), f(mg))
    assert all_equal(f.imag(mg), rspace.zero()(mg))

    # Complex conjugate
    f = cspace.element(cfunc_2d_vec_oop)
    fbar = f.conj()
    assert all_equal(fbar(mg), cfunc_2d_vec_oop(mg).conj())
    out_mg = np.empty((2, 3), dtype='complex128')
    fbar(mg, out=out_mg)
    assert all_equal(out_mg, cfunc_2d_vec_oop(mg).conj())


def test_fspace_zero():
    rect, points, mg = _standard_setup_2d()

    # real
    fspace = FunctionSpace(rect)
    zero_vec = fspace.zero()

    assert zero_vec([0.5, 1.5]) == 0.0
    assert all_equal(zero_vec(points), np.zeros(5, dtype=float))
    assert all_equal(zero_vec(mg), np.zeros((2, 3), dtype=float))

    # complex
    fspace = FunctionSpace(rect, range=odl.ComplexNumbers())
    zero_vec = fspace.zero()

    assert zero_vec([0.5, 1.5]) == 0.0 + 1j * 0.0
    assert all_equal(zero_vec(points), np.zeros(5, dtype=complex))
    assert all_equal(zero_vec(mg), np.zeros((2, 3), dtype=complex))


def test_fspace_one():
    rect, points, mg = _standard_setup_2d()

    # real
    fspace = FunctionSpace(rect)
    one_vec = fspace.one()

    assert one_vec([0.5, 1.5]) == 1.0
    assert all_equal(one_vec(points), np.ones(5, dtype=float))
    assert all_equal(one_vec(mg), np.ones((2, 3), dtype=float))

    # complex
    fspace = FunctionSpace(rect, range=odl.ComplexNumbers())
    one_vec = fspace.one()

    assert one_vec([0.5, 1.5]) == 1.0 + 1j * 0.0
    assert all_equal(one_vec(points), np.ones(5, dtype=complex))
    assert all_equal(one_vec(mg), np.ones((2, 3), dtype=complex))


a = simple_fixture('a', [2.0, 0.0, -1.0])
b = simple_fixture('b', [2.0, 0.0, -1.0])


def test_fspace_lincomb(a, b):
    rect, points, mg = _standard_setup_2d()
    point = points.T[0]

    fspace = FunctionSpace(rect)

    # Note: Special cases and alignment are tested later in the magic methods

    # Not vectorized
    true_novec = a * func_2d_novec(point) + b * other_func_2d_novec(point)
    f_novec = fspace.element(func_2d_novec, vectorized=False)
    g_novec = fspace.element(other_func_2d_novec, vectorized=False)
    out_novec = fspace.element(vectorized=False)
    fspace.lincomb(a, f_novec, b, g_novec, out_novec)
    assert almost_equal(out_novec(point), true_novec)

    # Vectorized
    true_arr = (a * func_2d_vec_oop(points) +
                b * other_func_2d_vec_oop(points))
    true_mg = (a * func_2d_vec_oop(mg) + b * other_func_2d_vec_oop(mg))

    # Out-of-place
    f_vec_oop = fspace.element(func_2d_vec_oop, vectorized=True)
    g_vec_oop = fspace.element(other_func_2d_vec_oop, vectorized=True)
    out_vec = fspace.element()
    fspace.lincomb(a, f_vec_oop, b, g_vec_oop, out_vec)

    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)
    assert almost_equal(out_vec(point), true_novec)
    out_arr = np.empty((5,), dtype=float)
    out_mg = np.empty((2, 3), dtype=float)
    out_vec(points, out=out_arr)
    out_vec(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # In-place
    f_vec_ip = fspace.element(func_2d_vec_ip, vectorized=True)
    g_vec_ip = fspace.element(other_func_2d_vec_ip, vectorized=True)
    out_vec = fspace.element()
    fspace.lincomb(a, f_vec_ip, b, g_vec_ip, out_vec)

    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)
    assert almost_equal(out_vec(point), true_novec)
    out_arr = np.empty((5,), dtype=float)
    out_mg = np.empty((2, 3), dtype=float)
    out_vec(points, out=out_arr)
    out_vec(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # Dual use
    f_vec_dual = fspace.element(func_2d_vec_dual, vectorized=True)
    g_vec_dual = fspace.element(other_func_2d_vec_dual, vectorized=True)
    out_vec = fspace.element()
    fspace.lincomb(a, f_vec_dual, b, g_vec_dual, out_vec)

    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)
    assert almost_equal(out_vec(point), true_novec)
    out_arr = np.empty((5,), dtype=float)
    out_mg = np.empty((2, 3), dtype=float)
    out_vec(points, out=out_arr)
    out_vec(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # Mix of vectorized and non-vectorized -> manual vectorization
    fspace.lincomb(a, f_vec_dual, b, g_novec, out_vec)
    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)


# NOTE: multiply and divide are tested via magic methods

power = simple_fixture('power', [3, 1.0, 0.5, 6.0])


def test_fspace_power(power):
    rect, points, mg = _standard_setup_2d()
    point = points.T[0]
    out_arr = np.empty(5)
    out_mg = np.empty((2, 3))

    fspace = FunctionSpace(rect)

    # Not vectorized
    true_novec = func_2d_novec(point) ** power

    f_novec = fspace.element(func_2d_novec, vectorized=False)
    pow_novec = f_novec ** power
    assert almost_equal(pow_novec(point), true_novec)

    pow_novec = f_novec.copy()
    pow_novec **= power

    assert almost_equal(pow_novec(point), true_novec)

    # Vectorized
    true_arr = func_2d_vec_oop(points) ** power
    true_mg = func_2d_vec_oop(mg) ** power

    f_vec = fspace.element(func_2d_vec_dual, vectorized=True)
    pow_vec = f_vec ** power

    assert all_almost_equal(pow_vec(points), true_arr)
    assert all_almost_equal(pow_vec(mg), true_mg)

    pow_vec = f_vec.copy()
    pow_vec **= power

    assert all_almost_equal(pow_vec(points), true_arr)
    assert all_almost_equal(pow_vec(mg), true_mg)

    pow_vec(points, out=out_arr)
    pow_vec(mg, out=out_mg)

    assert all_almost_equal(out_arr, true_arr)
    assert all_almost_equal(out_mg, true_mg)


op = simple_fixture('op', ['+', '+=', '-', '-=', '*', '*=', '/', '/='])


var_params = ['vv', 'vs', 'sv']
var_ids = [' vec <op> vec ', ' vec <op> scal ', ' scal <op> vec ']


@pytest.fixture(scope="module", ids=var_ids, params=var_params)
def variant(request):
    return request.param


def _op(a, op, b):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a / b
    if op == '+=':
        a += b
        return a
    elif op == '-=':
        a -= b
        return a
    elif op == '*=':
        a *= b
        return a
    elif op == '/=':
        a /= b
        return a
    else:
        raise ValueError("bad operator '{}'.".format(op))


def test_fspace_vector_arithmetic(variant, op):
    if variant == 'sv' and '=' in op:  # makes no sense, quit
        return

    # Setup
    rect, points, mg = _standard_setup_2d()
    point = points.T[0]

    fspace = FunctionSpace(rect)
    a = -1.5
    b = 2.0
    array_out = np.empty((5,), dtype=float)
    mg_out = np.empty((2, 3), dtype=float)

    # Initialize a bunch of elements
    f_novec = fspace.element(func_2d_novec, vectorized=False)
    f_vec = fspace.element(func_2d_vec_dual, vectorized=True)
    g_novec = fspace.element(other_func_2d_novec, vectorized=False)
    g_vec = fspace.element(other_func_2d_vec_dual, vectorized=True)

    out_novec = fspace.element(vectorized=False)
    out_vec = fspace.element(vectorized=True)

    if variant[0] == 'v':
        true_l_novec = func_2d_novec(point)
        true_l_arr = func_2d_vec_oop(points)
        true_l_mg = func_2d_vec_oop(mg)

        test_l_novec = f_novec
        test_l_vec = f_vec
    else:  # 's'
        true_l_novec = true_l_arr = true_l_mg = a
        test_l_novec = test_l_vec = a

    if variant[1] == 'v':
        true_r_novec = other_func_2d_novec(point)
        true_r_arr = other_func_2d_vec_oop(points)
        true_r_mg = other_func_2d_vec_oop(mg)

        test_r_novec = g_novec
        test_r_vec = g_vec
    else:  # 's'
        true_r_novec = true_r_arr = true_r_mg = b
        test_r_novec = test_r_vec = b

    true_novec = _op(true_l_novec, op, true_r_novec)
    true_arr = _op(true_l_arr, op, true_r_arr)
    true_mg = _op(true_l_mg, op, true_r_mg)

    out_novec = _op(test_l_novec, op, test_r_novec)
    out_vec = _op(test_l_vec, op, test_r_vec)

    assert almost_equal(out_novec(point), true_novec)
    assert all_equal(out_vec(points), true_arr)
    out_vec(points, out=array_out)
    assert all_equal(array_out, true_arr)
    assert all_equal(out_vec(mg), true_mg)
    out_vec(mg, out=mg_out)
    assert all_equal(mg_out, true_mg)


# ---- Test function definitions ----

# 'ip' = in-place, 'oop' = out-of-place, 'dual' = dual-use

def func_1d_oop(x):
    return x ** 2


def func_1d_ip(x, out):
    out[:] = x ** 2


def func_1d_dual(x, out=None):
    if out is None:
        return x ** 2
    else:
        out[:] = x ** 2


def func_2d_novec(x):
    return x[0] ** 2 + x[1]


def func_2d_vec_oop(x):
    return x[0] ** 2 + x[1]


def func_2d_vec_ip(x, out):
    out[:] = x[0] ** 2 + x[1]


def func_2d_vec_dual(x, out=None):
    if out is None:
        return x[0] ** 2 + x[1]
    else:
        out[:] = x[0] ** 2 + x[1]


def cfunc_2d_novec(x):
    return x[0] ** 2 + 1j * x[1]


def cfunc_2d_vec_oop(x):
    return x[0] ** 2 + 1j * x[1]


def cfunc_2d_vec_ip(x, out):
    out[:] = x[0] ** 2 + 1j * x[1]


def cfunc_2d_vec_dual(x, out=None):
    if out is None:
        return x[0] ** 2 + 1j * x[1]
    else:
        out[:] = x[0] ** 2 + 1j * x[1]


def other_func_2d_novec(x):
    return x[0] + abs(x[1])


def other_func_2d_vec_oop(x):
    return x[0] + abs(x[1])


def other_func_2d_vec_ip(x, out):
    out[:] = x[0] + abs(x[1])


def other_func_2d_vec_dual(x, out=None):
    if out is None:
        return x[0] + abs(x[1])
    else:
        out[:] = x[0] + abs(x[1])


def other_cfunc_2d_novec(x):
    return 1j * x[0] + abs(x[1])


def other_cfunc_2d_vec_oop(x):
    return 1j * x[0] + abs(x[1])


def other_cfunc_2d_vec_ip(x, out):
    out[:] = 1j * x[0] + abs(x[1])


def other_cfunc_2d_vec_dual(x, out=None):
    if out is None:
        return 1j * x[0] + abs(x[1])
    else:
        out[:] = 1j * x[0] + abs(x[1])


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
