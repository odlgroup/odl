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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, str

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
from odl import L2
from odl.util.testutils import all_equal, all_almost_equal, almost_equal


# TODO:
# - vector unwrapping
# - 'F' order in zero and one
# - zero apply variant
# - lincomb_call with a == 0 and b != 0
# - lincomb_apply with both a == 0, both b cases
# - lincomb without out parameter
# - vec**n and vec **= n
# - +vec and -vec

# Define a bunch of example functions with different vectorization
def func_1d(x):
    return x**2


def func_1d_apply(x, out):
    out[:] = x**2


def func_2d_novec(x):
    return x[0]**2 + x[1]


def func_2d_vec(x):
    x0, x1 = x
    return x0**2 + x1


def func_2d_vec_apply(x, out):
    x0, x1 = x
    out[:] = x0**2 + x1


def cfunc_2d_novec(x):
    return x[0]**2 + 1j*x[1]


def cfunc_2d_vec(x):
    x0, x1 = x
    return x0**2 + 1j*x1


def cfunc_2d_vec_apply(x, out):
    x0, x1 = x
    out[:] = x0**2 + 1j*x1


def other_func_2d_novec(x):
    return x[0] + abs(x[1])


def other_func_2d_vec(x):
    x0, x1 = x
    return x0 + abs(x1)


def other_func_2d_vec_apply(x, out):
    x0, x1 = x
    out[:] = x0 + abs(x1)


def other_cfunc_2d_novec(x):
    return 1j*x[0] + abs(x[1])


def other_cfunc_2d_vec(x):
    x0, x1 = x
    return 1j*x0 + abs(x1)


def other_cfunc_2d_vec_apply(x, out):
    x0, x1 = x
    out[:] = 1j*x0 + abs(x1)


# -------- Here the actual tests begin --------


def test_l2_init():
    intv = odl.Interval(0, 1)
    L2(intv)
    L2(intv, field=odl.RealNumbers())
    L2(intv, field=odl.ComplexNumbers())

    rect = odl.Rectangle([0, 0], [1, 2])
    L2(rect)
    L2(rect, field=odl.RealNumbers())
    L2(rect, field=odl.ComplexNumbers())

    cube = odl.Cuboid([0, 0, 0], [1, 2, 3])
    L2(cube)
    L2(cube, field=odl.RealNumbers())
    L2(cube, field=odl.ComplexNumbers())

    ndbox = odl.IntervalProd([0]*10, np.arange(1, 11))
    L2(ndbox)
    L2(ndbox, field=odl.RealNumbers())
    L2(ndbox, field=odl.ComplexNumbers())


def test_l2_simple_attributes():
    intv = odl.Interval(0, 1)
    l2 = L2(intv)
    l2_r = L2(intv, field=odl.RealNumbers())
    l2_c = L2(intv, field=odl.ComplexNumbers())

    assert l2.domain == intv
    assert l2.range == odl.RealNumbers()
    assert l2_r.range == odl.RealNumbers()
    assert l2_c.range == odl.ComplexNumbers()


def test_l2_equality():
    intv = odl.Interval(0, 1)
    intv2 = odl.Interval(-1, 1)
    l2 = L2(intv)
    l2_r = L2(intv, field=odl.RealNumbers())
    l2_c = L2(intv, field=odl.ComplexNumbers())
    l2_intv2 = L2(intv2)

    assert l2 == l2_r
    assert l2 != l2_c
    assert l2 != l2_intv2


def _points(domain, num):
    beg = domain.begin
    end = domain.end
    ndim = domain.ndim
    points = np.random.uniform(low=0, high=1, size=(ndim, num))
    for i in range(ndim):
        points[i, :] = beg[i] + (end[i] - beg[i]) * points[i]
    return points


def _meshgrid(domain, shape):
    beg = domain.begin
    end = domain.end
    ndim = domain.ndim
    coord_vecs = []
    for i in range(ndim):
        vec = np.random.uniform(low=beg[i], high=end[i], size=shape[i])
        vec.sort()
        coord_vecs.append(vec)
    return np.meshgrid(*coord_vecs, indexing='ij', sparse=True,
                       copy=True)


def test_l2_vector_init():
    # 1d, real
    intv = odl.Interval(0, 1)
    l2 = L2(intv)
    l2.element(func_1d)
    l2.element(func_1d, vectorized=False)
    l2.element(func_1d, vectorized=True)
    l2.element(func_1d, func_1d_apply, vectorized=True)

    with pytest.raises(ValueError):
        l2.element(func_1d, func_1d_apply, vectorized=False)

    # 2d, real
    rect = odl.Rectangle([0, 0], [1, 2])
    l2 = L2(rect)
    l2.element(func_2d_novec, vectorized=False)
    l2.element(func_2d_vec)
    l2.element(func_2d_vec, vectorized=True)
    l2.element(func_2d_vec, func_2d_vec_apply, vectorized=True)

    with pytest.raises(ValueError):
        l2.element(func_2d_novec, func_2d_vec_apply, vectorized=False)

    # 2d, complex
    l2 = L2(rect, field=odl.ComplexNumbers())
    l2.element(cfunc_2d_novec, vectorized=False)
    l2.element(cfunc_2d_vec)
    l2.element(cfunc_2d_vec, vectorized=True)
    l2.element(cfunc_2d_vec, cfunc_2d_vec_apply, vectorized=True)


def _standard_setup_2d(small=False):
    rect = odl.Rectangle([0, 0], [1, 2])
    if small:
        points = _points(rect, num=5)
        mg = _meshgrid(rect, shape=(2, 3))
    else:
        points = _points(rect, num=50)
        mg = _meshgrid(rect, shape=(5, 10))
    return rect, points, mg


def test_l2_vector_assign():
    rect, points, mg = _standard_setup_2d()

    l2 = L2(rect)
    f_novec = l2.element(func_2d_novec, vectorized=False)
    f_vec = l2.element(func_2d_vec, vectorized=True)

    # Not vectorized
    f_out = l2.element()
    f_out.assign(f_novec)
    assert f_out == f_novec
    for point in points.T:
        assert f_out(point) == f_novec(point)

    # Vectorized
    f_out.assign(f_vec)  # Overwrites old `vectorized`
    assert f_out == f_vec
    assert all_equal(f_out(mg), f_vec(mg))
    assert all_equal(f_out(points), f_vec(points))


def test_l2_vector_copy():
    rect, points, mg = _standard_setup_2d()

    l2 = L2(rect)
    f_novec = l2.element(func_2d_novec, vectorized=False)
    f_vec = l2.element(func_2d_vec, vectorized=True)

    # Not vectorized
    f_out = f_novec.copy()
    assert f_out == f_novec
    for point in points.T:
        assert f_out(point) == f_novec(point)

    # Vectorized
    f_out = f_vec.copy()
    assert f_out == f_vec
    assert all_equal(f_out(mg), f_vec(mg))
    assert all_equal(f_out(points), f_vec(points))


def test_l2_vector_call():
    rect, points, mg = _standard_setup_2d()

    # real
    l2 = L2(rect)
    f_novec = l2.element(func_2d_novec, vectorized=False)
    f_vec = l2.element(func_2d_vec)  # Default: vectorized

    # Not vectorized
    for point in points.T:
        assert almost_equal(f_novec(point), func_2d_novec(point))
    # Vectorized
    assert all_almost_equal(f_vec(points), func_2d_vec(points))
    assert all_almost_equal(f_vec(mg), func_2d_vec(mg))

    with pytest.raises(TypeError):
        f_novec(points)
    with pytest.raises(TypeError):
        f_novec(mg)
    with pytest.raises(TypeError):  # ValueError: invalid vectorized input
        f_vec(points[0])

    # complex
    l2 = L2(rect, field=odl.ComplexNumbers())
    f_novec = l2.element(cfunc_2d_novec, vectorized=False)
    f_vec = l2.element(cfunc_2d_vec, vectorized=True)

    # Not vectorized
    for point in points.T:
        assert almost_equal(f_novec(point), cfunc_2d_novec(point))
    # Vectorized
    assert all_almost_equal(f_vec(points), cfunc_2d_vec(points))
    assert all_almost_equal(f_vec(mg), cfunc_2d_vec(mg))

    # Test bounds check
    points_outside_1 = np.array([[-1., 0], [0, 0]])
    points_outside_2 = np.array([[1., 0], [0, 2.5]])
    mg_outside_1 = np.meshgrid([-1, 0], [0.5, 1.5], indexing='ij',
                               sparse=True, copy=True)
    mg_outside_2 = np.meshgrid([0, 0.5], [0.5, 3.5], indexing='ij',
                               sparse=True, copy=True)

    with pytest.raises(TypeError):
        f_novec([-1, 0])
    with pytest.raises(TypeError):
        f_novec([0, 2.5])
    with pytest.raises(ValueError):
        f_vec(points_outside_1)
    with pytest.raises(ValueError):
        f_vec(points_outside_2)
    with pytest.raises(ValueError):
        f_vec(mg_outside_1)
    with pytest.raises(ValueError):
        f_vec(mg_outside_2)

    # Test disabling vectorized bounds check
    f_vec(points_outside_1, vec_bounds_check=False)
    f_vec(mg_outside_1, vec_bounds_check=False)


def test_l2_vector_apply():
    rect, points, mg = _standard_setup_2d()

    # real
    l2 = L2(rect)
    f_vec = l2.element(func_2d_vec, func_2d_vec_apply, vectorized=True)

    out = np.empty((50,), dtype=float)
    f_vec(points, out=out)
    assert all_almost_equal(out, func_2d_vec(points))
    out = np.empty((5, 10), dtype=float)
    f_vec(mg, out=out)
    assert all_almost_equal(out, func_2d_vec(mg))

    out = np.empty((5,), dtype=float)  # wrong shape
    with pytest.raises(ValueError):
        f_vec(points, out=out)
    with pytest.raises(ValueError):
        f_vec(mg, out=out)

    # complex
    l2 = L2(rect, field=odl.ComplexNumbers())
    f_vec = l2.element(cfunc_2d_vec, cfunc_2d_vec_apply, vectorized=True)

    out = np.empty((50,), dtype=complex)
    f_vec(points, out=out)
    assert all_almost_equal(out, cfunc_2d_vec(points))
    out = np.empty((5, 10), dtype=complex)
    f_vec(mg, out=out)
    assert all_almost_equal(out, cfunc_2d_vec(mg))


def test_l2_vector_equality():
    rect = odl.Rectangle([0, 0], [1, 2])
    l2 = L2(rect)

    f_novec = l2.element(func_2d_novec, vectorized=False)
    f_novec_2 = l2.element(func_2d_novec, vectorized=False)

    f_vec = l2.element(func_2d_vec, vectorized=True)
    f_vec_2 = l2.element(func_2d_vec, vectorized=True)
    f_vec_a = l2.element(func_2d_vec, func_2d_vec_apply, vectorized=True)
    f_vec_a_2 = l2.element(func_2d_vec, func_2d_vec_apply, vectorized=True)

    assert f_novec == f_novec
    assert f_novec == f_novec_2
    assert f_novec != f_vec

    assert f_vec == f_vec_2
    assert f_vec_a == f_vec_a_2
    assert f_vec != f_vec_a
    assert f_vec != f_novec


def test_l2_zero():
    rect, points, mg = _standard_setup_2d(small=True)

    # real
    l2 = L2(rect)

    zero_novec = l2.zero(vectorized=False)
    zero_vec = l2.zero(vectorized=True)

    for point in points.T:
        assert zero_novec(point) == 0.0

    assert all_equal(zero_vec(points), np.zeros(5, dtype=float))
    assert all_equal(zero_vec(mg), np.zeros((2, 3), dtype=float))

    # complex
    l2 = L2(rect, field=odl.ComplexNumbers())

    zero_novec = l2.zero(vectorized=False)
    zero_vec = l2.zero()

    for point in points.T:
        assert zero_novec(point) == 0.0 + 1j*0.0

    assert all_equal(zero_vec(points), np.zeros(5, dtype=complex))
    assert all_equal(zero_vec(mg), np.zeros((2, 3), dtype=complex))


def test_l2_one():
    rect, points, mg = _standard_setup_2d(small=True)

    # real
    l2 = L2(rect)

    one_novec = l2.one(vectorized=False)
    one_vec = l2.one(vectorized=True)

    for point in points.T:
        assert one_novec(point) == 1.0

    assert all_equal(one_vec(points), np.ones(5, dtype=float))
    assert all_equal(one_vec(mg), np.ones((2, 3), dtype=float))

    # complex
    l2 = L2(rect, field=odl.ComplexNumbers())

    one_novec = l2.one(vectorized=False)
    one_vec = l2.one()

    for point in points.T:
        assert one_novec(point) == 1.0 + 1j*0.0

    assert all_equal(one_vec(points), np.ones(5, dtype=complex))
    assert all_equal(one_vec(mg), np.ones((2, 3), dtype=complex))


def test_l2_lincomb():
    rect, points, mg = _standard_setup_2d(small=True)
    point = points.T[0]

    # REAL
    l2 = L2(rect)
    a = -1.5
    b = 2.0

    # Note: Special cases and alignment are tested later in the magic methods

    # Not vectorized
    true_novec = a * func_2d_novec(point) + b * other_func_2d_novec(point)
    f_novec = l2.element(func_2d_novec, vectorized=False)
    g_novec = l2.element(other_func_2d_novec, vectorized=False)
    out_novec = l2.element()
    l2.lincomb(a, f_novec, b, g_novec, out_novec)
    assert out_novec(point) == true_novec

    # Vectorized
    true_arr = (a * func_2d_vec(points) + b * other_func_2d_vec(points))
    true_mg = (a * func_2d_vec(mg) + b * other_func_2d_vec(mg))
    f_vec = l2.element(func_2d_vec, vectorized=True)
    g_vec = l2.element(other_func_2d_vec, vectorized=True)
    out_vec = l2.element(vectorized=True)
    l2.lincomb(a, f_vec, b, g_vec, out_vec)

    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)
    assert out_vec(point) == true_novec  # should work, too

    f_vec_a = l2.element(func_2d_vec, func_2d_vec_apply)
    g_vec_a = l2.element(other_func_2d_vec, other_func_2d_vec_apply)
    out_vec_a = l2.element()
    l2.lincomb(a, f_vec_a, b, g_vec_a, out_vec_a)
    # Check if out-of-place still works
    assert all_equal(out_vec(mg), true_mg)
    # In-place
    out = np.empty((2, 3), dtype=float)
    out_vec_a(mg, out=out)
    assert all_equal(out, true_mg)
    out = np.empty((5,), dtype=float)
    out_vec_a(points, out=out)
    assert all_equal(out, true_arr)

    # Mix of vectorized and non-vectorized -> manual vectorization
    l2.lincomb(a, f_vec_a, b, g_novec, out_vec_a)
    assert all_equal(out_vec_a(points), true_arr)
    assert all_equal(out_vec_a(mg), true_mg)
    out = np.empty((2, 3), dtype=float)
    out_vec_a(mg, out=out)
    assert all_equal(out, true_mg)
    out = np.empty((5,), dtype=float)
    out_vec_a(points, out=out)
    assert all_equal(out, true_arr)

    # COMPLEX
    l2 = L2(rect, field=odl.ComplexNumbers())
    a = -1.5 + 1j*7
    b = 2.0 - 1j

    # Not vectorized
    true_novec = a * cfunc_2d_novec(point) + b * other_cfunc_2d_novec(point)
    f_novec = l2.element(cfunc_2d_novec)
    g_novec = l2.element(other_cfunc_2d_novec)
    out_novec = l2.element()
    l2.lincomb(a, f_novec, b, g_novec, out_novec)
    assert out_novec(point) == true_novec

    # Vectorized
    true_arr = (a * cfunc_2d_vec(points) + b * other_cfunc_2d_vec(points))
    true_mg = (a * cfunc_2d_vec(mg) + b * other_cfunc_2d_vec(mg))
    f_vec = l2.element(cfunc_2d_vec, vectorized=True)
    g_vec = l2.element(other_cfunc_2d_vec, vectorized=True)
    out_vec = l2.element(vectorized=True)
    l2.lincomb(a, f_vec, b, g_vec, out_vec)

    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)

    f_vec_a = l2.element(cfunc_2d_vec, cfunc_2d_vec_apply, vectorized=True)
    g_vec_a = l2.element(other_cfunc_2d_vec, other_cfunc_2d_vec_apply,
                         vectorized=True)
    out_vec_a = l2.element()
    l2.lincomb(a, f_vec_a, b, g_vec_a, out_vec_a)
    # Check if out-of-place still works
    assert all_equal(out_vec(mg), true_mg)
    # In-place
    out = np.empty((2, 3), dtype=float)
    out_vec_a(mg, out=out)
    assert all_equal(out, true_mg)
    out = np.empty((5,), dtype=float)
    out_vec_a(points, out=out)
    assert all_equal(out, true_arr)


# NOTE: multiply and divide are tested via magic methods


def _test_l2_vector_op(op_str, pattern):
    if op_str not in ('+', '-', '*', '/'):
        raise ValueError('bad operator {!r}.'.format(op_str))

    if pattern not in ('sv', 'vv', 'vs', 'iv', 'is'):
        raise ValueError('bad pattern {!r}'.format(pattern))

    # Setup
    rect = odl.Rectangle([0, 0], [1, 2])
    l2 = L2(rect)
    points = _points(rect, num=5)
    point = points.T[0]
    mg = _meshgrid(rect, shape=(2, 3))
    a = -1.5
    b = 2.0
    array_out = np.empty((5,), dtype=float)
    mg_out = np.empty((2, 3), dtype=float)

    # Initialize a bunch of elements
    f_novec = l2.element(func_2d_novec, vectorized=False)
    f_vec = l2.element(func_2d_vec, vectorized=True)
    f_vec_a = l2.element(func_2d_vec, func_2d_vec_apply, vectorized=True)
    g_novec = l2.element(other_func_2d_novec, vectorized=False)
    g_vec = l2.element(other_func_2d_vec, vectorized=True)
    g_vec_a = l2.element(other_func_2d_vec, other_func_2d_vec_apply,
                         vectorized=True)

    out_novec = l2.element(vectorized=False)
    out_vec = l2.element(vectorized=True)
    out_vec_a = l2.element(vectorized=True)

    if pattern[0] in ('v', 'i'):
        true_l_novec = func_2d_novec(point)
        true_l_arr = func_2d_vec(points)
        true_l_mg = func_2d_vec(mg)

        test_l_novec = f_novec
        test_l_vec = f_vec
        test_l_vec_a = f_vec_a
    else:  # 's'
        true_l_novec = true_l_arr = true_l_mg = a
        test_l_novec = test_l_vec = test_l_vec_a = a

    if pattern[1] == 'v':
        true_r_novec = other_func_2d_novec(point)
        true_r_arr = other_func_2d_vec(points)
        true_r_mg = other_func_2d_vec(mg)

        test_r_novec = g_novec
        test_r_vec = g_vec
        test_r_vec_a = g_vec_a
    else:  # 's'
        true_r_novec = true_r_arr = true_r_mg = b
        test_r_novec = test_r_vec = test_r_vec_a = b

    if op_str == '+':
        true_novec = true_l_novec + true_r_novec
        true_arr = true_l_arr + true_r_arr
        true_mg = true_l_mg + true_r_mg

        if pattern[0] == 'i':
            test_l_novec += test_r_novec
            test_l_vec += test_r_vec
            test_l_vec_a += test_r_vec_a
        else:
            out_novec = test_l_novec + test_r_novec
            out_vec = test_l_vec + test_r_vec
            out_vec_a = test_l_vec_a + test_r_vec_a
    elif op_str == '-':
        true_novec = true_l_novec - true_r_novec
        true_arr = true_l_arr - true_r_arr
        true_mg = true_l_mg - true_r_mg

        if pattern[0] == 'i':
            test_l_novec -= test_r_novec
            test_l_vec -= test_r_vec
            test_l_vec_a -= test_r_vec_a
        else:
            out_novec = test_l_novec - test_r_novec
            out_vec = test_l_vec - test_r_vec
            out_vec_a = test_l_vec_a - test_r_vec_a
    elif op_str == '*':
        true_novec = true_l_novec * true_r_novec
        true_arr = true_l_arr * true_r_arr
        true_mg = true_l_mg * true_r_mg

        if pattern[0] == 'i':
            test_l_novec *= test_r_novec
            test_l_vec *= test_r_vec
            test_l_vec_a *= test_r_vec_a
        else:
            out_novec = test_l_novec * test_r_novec
            out_vec = test_l_vec * test_r_vec
            out_vec_a = test_l_vec_a * test_r_vec_a
    elif op_str == '/':
        true_novec = true_l_novec / true_r_novec
        true_arr = true_l_arr / true_r_arr
        true_mg = true_l_mg / true_r_mg

        if pattern[0] == 'i':
            test_l_novec /= test_r_novec
            test_l_vec /= test_r_vec
            test_l_vec_a /= test_r_vec_a
        else:
            out_novec = test_l_novec / test_r_novec
            out_vec = test_l_vec / test_r_vec
            out_vec_a = test_l_vec_a / test_r_vec_a

    if pattern[0] == 'i':
        assert test_l_novec(point) == true_novec
        assert all_equal(test_l_vec(points), true_arr)
        test_l_vec_a(points, out=array_out)
        assert all_equal(array_out, true_arr)
        assert all_equal(test_l_vec(mg), true_mg)
        test_l_vec_a(mg, out=mg_out)
        assert all_equal(mg_out, true_mg)
    else:
        assert out_novec(point) == true_novec
        assert all_equal(out_vec(points), true_arr)
        out_vec_a(points, out=array_out)
        assert all_equal(array_out, true_arr)
        assert all_equal(out_vec(mg), true_mg)
        out_vec_a(mg, out=mg_out)
        assert all_equal(mg_out, true_mg)


def test_l2_vector_add():
    _test_l2_vector_op('+', 'vv')  # vec + vec
    _test_l2_vector_op('+', 'vs')  # vec + scal
    _test_l2_vector_op('+', 'sv')  # scal + vec
    _test_l2_vector_op('+', 'iv')  # vec += vec
    _test_l2_vector_op('+', 'is')  # vec += scal


def test_l2_vector_sub():
    _test_l2_vector_op('-', 'vv')  # vec - vec
    _test_l2_vector_op('-', 'vs')  # vec - scal
    _test_l2_vector_op('-', 'sv')  # scal - vec
    _test_l2_vector_op('-', 'iv')  # vec -= vec
    _test_l2_vector_op('-', 'is')  # vec -= scal


def test_l2_vector_mul():
    _test_l2_vector_op('*', 'vv')  # vec * vec
    _test_l2_vector_op('*', 'vs')  # vec * scal
    _test_l2_vector_op('*', 'sv')  # scal * vec
    _test_l2_vector_op('*', 'iv')  # vec *= vec
    _test_l2_vector_op('*', 'is')  # vec *= scal


def test_l2_vector_div():
    _test_l2_vector_op('/', 'vv')  # vec / vec
    _test_l2_vector_op('/', 'vs')  # vec / scal
    _test_l2_vector_op('/', 'sv')  # scal / vec
    _test_l2_vector_op('/', 'iv')  # vec /= vec
    _test_l2_vector_op('/', 'is')  # vec /= scal


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
