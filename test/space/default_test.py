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


def func_2d_array(x):
    x0, x1 = x[:, 0], x[:, 1]
    return x0**2 + x1


def func_2d_array_apply(x, out):
    x0, x1 = x[:, 0], x[:, 1]
    out[:] = x0**2
    out += x1


def func_2d_mg(x):
    x0, x1 = x
    return x0**2 + x1


def func_2d_mg_apply(x, out):
    x0, x1 = x
    out[:] = x0**2 + x1


def cfunc_2d_novec(x):
    return x[0]**2 + 1j*x[1]


def cfunc_2d_array(x):
    x0, x1 = x[:, 0], x[:, 1]
    return x0**2 + 1j*x1


def cfunc_2d_array_apply(x, out):
    x0, x1 = x[:, 0], x[:, 1]
    out[:] = x0**2
    out += 1j*x1


def cfunc_2d_mg(x):
    x0, x1 = x
    return x0**2 + 1j*x1


def cfunc_2d_mg_apply(x, out):
    x0, x1 = x
    out[:] = x0**2 + 1j*x1


def other_func_2d_novec(x):
    return x[0] + abs(x[1])


def other_func_2d_array(x):
    x0, x1 = x[:, 0], x[:, 1]
    return x0 + abs(x1)


def other_func_2d_array_apply(x, out):
    x0, x1 = x[:, 0], x[:, 1]
    out[:] = abs(x1)
    out += x0


def other_func_2d_mg(x):
    x0, x1 = x
    return x0 + abs(x1)


def other_func_2d_mg_apply(x, out):
    x0, x1 = x
    out[:] = x0 + abs(x1)


def other_cfunc_2d_novec(x):
    return 1j*x[0] + abs(x[1])


def other_cfunc_2d_array(x):
    x0, x1 = x[:, 0], x[:, 1]
    return 1j*x0 + abs(x1)


def other_cfunc_2d_array_apply(x, out):
    x0, x1 = x[:, 0], x[:, 1]
    out[:] = abs(x1)
    out += 1j*x0


def other_cfunc_2d_mg(x):
    x0, x1 = x
    return 1j*x0 + abs(x1)


def other_cfunc_2d_mg_apply(x, out):
    x0, x1 = x
    out[:] = 1j*x0 + abs(x1)


#func_names = {'func_2d_novec': func_2d_novec,
#              'func_2d_array': func_2d_array,
#              'func_2d_array_apply': func_2d_array_apply,
#              'func_2d_mg': func_2d_mg,
#              'func_2d_mg_apply': func_2d_mg_apply,
#              'other_func_2d_novec': other_func_2d_novec,
#              'other_func_2d_array': other_func_2d_array,
#              'other_func_2d_array_apply': other_func_2d_array_apply,
#              'other_func_2d_mg': other_func_2d_mg,
#              'other_func_2d_mg_apply': other_func_2d_mg_apply}

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
    points = np.random.uniform(low=0, high=1, size=(num, ndim))
    for i in range(ndim):
        points[:, i] = beg[i] + (end[i] - beg[i]) * points[:, i]
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
    l2.element(func_1d, vectorization='none')
    l2.element(func_1d, vectorization='array')
    l2.element(func_1d, vectorization='meshgrid')
    l2.element(func_1d, func_1d_apply, vectorization='array')
    l2.element(func_1d, func_1d_apply, vectorization='meshgrid')

    with pytest.raises(ValueError):
        l2.element(func_1d, func_1d_apply, vectorization='none')

    # 2d, real
    rect = odl.Rectangle([0, 0], [1, 2])
    l2 = L2(rect)
    l2.element(func_2d_novec)
    l2.element(func_2d_novec, vectorization='none')
    l2.element(func_2d_array, vectorization='array')
    l2.element(func_2d_mg, vectorization='meshgrid')
    l2.element(func_2d_array, func_2d_array_apply, vectorization='array')
    l2.element(func_2d_mg, func_2d_mg_apply, vectorization='meshgrid')

    with pytest.raises(ValueError):
        l2.element(func_2d_novec, func_2d_array_apply,
                   vectorization='none')

    # 2d, complex
    l2 = L2(rect, field=odl.ComplexNumbers())
    l2.element(cfunc_2d_novec)
    l2.element(cfunc_2d_novec, vectorization='none')
    l2.element(cfunc_2d_array, vectorization='array')
    l2.element(cfunc_2d_mg, vectorization='meshgrid')
    l2.element(cfunc_2d_array, cfunc_2d_array_apply, vectorization='array')
    l2.element(cfunc_2d_mg, cfunc_2d_mg_apply, vectorization='meshgrid')


def test_l2_vector_assign():
    rect = odl.Rectangle([0, 0], [1, 2])
    points = _points(rect, num=50)
    mg = _meshgrid(rect, shape=(5, 10))

    l2 = L2(rect)
    f_novec = l2.element(func_2d_novec)
    f_array = l2.element(func_2d_array, vectorization='array')
    f_mg = l2.element(func_2d_mg, vectorization='meshgrid')

    # No vectorization
    f_out = l2.element()
    f_out.assign(f_novec)
    assert f_out == f_novec
    for point in points:
        assert f_out(point) == f_novec(point)

    # Array vectorization
    f_out = l2.element(vectorization='array')
    f_out.assign(f_array)
    assert f_out == f_array
    assert all_equal(f_out(points), f_array(points))

    # Meshgrid vectorization
    f_out = l2.element(vectorization='meshgrid')
    f_out.assign(f_mg)
    assert f_out == f_mg
    assert all_equal(f_out(mg), f_mg(mg))


def test_l2_vector_copy():
    rect = odl.Rectangle([0, 0], [1, 2])
    points = _points(rect, num=50)
    mg = _meshgrid(rect, shape=(5, 10))

    l2 = L2(rect)
    f_novec = l2.element(func_2d_novec)
    f_array = l2.element(func_2d_array, vectorization='array')
    f_mg = l2.element(func_2d_mg, vectorization='meshgrid')

    # No vectorization
    f_out = f_novec.copy()
    assert f_out == f_novec
    for point in points:
        assert f_out(point) == f_novec(point)

    # Array vectorization
    f_out = f_array.copy()
    assert f_out == f_array
    assert all_equal(f_out(points), f_array(points))

    # Meshgrid vectorization
    f_out = f_mg.copy()
    assert f_out == f_mg
    assert all_equal(f_out(mg), f_mg(mg))


def test_l2_vector_call():
    rect = odl.Rectangle([0, 0], [1, 2])
    points = _points(rect, num=50)
    mg = _meshgrid(rect, shape=(5, 10))

    # real
    l2 = L2(rect)
    f_novec = l2.element(func_2d_novec)
    f_array = l2.element(func_2d_array, vectorization='array')
    f_mg = l2.element(func_2d_mg, vectorization='meshgrid')

    # non-vectorized
    for p in points:
        assert almost_equal(f_novec(p), func_2d_novec(p))
    # array version
    assert all_almost_equal(f_array(points), func_2d_array(points))
    # meshgrid version
    assert all_almost_equal(f_mg(mg), func_2d_mg(mg))

    with pytest.raises(TypeError):
        f_novec(points)
    with pytest.raises(TypeError):
        f_novec(mg)

    with pytest.raises(ValueError):  # ValueError: wrong shape
        f_array(points[0])
    with pytest.raises(TypeError):
        f_array(mg)

    with pytest.raises(TypeError):
        f_mg(points[0])
    with pytest.raises(ValueError):  # ValueError: wrong number of vecs
        f_mg(points)

    # complex
    l2 = L2(rect, field=odl.ComplexNumbers())
    f_novec = l2.element(cfunc_2d_novec)
    f_array = l2.element(cfunc_2d_array, vectorization='array')
    f_mg = l2.element(cfunc_2d_mg, vectorization='meshgrid')

    # non-vectorized
    for p in points:
        assert almost_equal(f_novec(p), cfunc_2d_novec(p))
    # array version
    assert all_almost_equal(f_array(points), cfunc_2d_array(points))
    # meshgrid version
    assert all_almost_equal(f_mg(mg), cfunc_2d_mg(mg))

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
        f_array(points_outside_1)
    with pytest.raises(ValueError):
        f_array(points_outside_2)
    with pytest.raises(ValueError):
        f_mg(mg_outside_1)
    with pytest.raises(ValueError):
        f_mg(mg_outside_2)

    # Test disabling vectorized bounds check
    f_array(points_outside_1, vec_bounds_check=False)
    f_mg(mg_outside_1, vec_bounds_check=False)


def test_l2_vector_apply():
    rect = odl.Rectangle([0, 0], [1, 2])
    points = _points(rect, num=50)
    mg = _meshgrid(rect, shape=(5, 10))

    # real
    l2 = L2(rect)
    f_array = l2.element(func_2d_array, func_2d_array_apply,
                         vectorization='array')
    f_mg = l2.element(func_2d_mg, func_2d_mg_apply,
                      vectorization='meshgrid')

    # array version
    out = np.empty((50,), dtype=float)
    f_array.apply(points, out)
    assert all_almost_equal(out, func_2d_array(points))
    # meshgrid version
    out = np.empty((5, 10), dtype=float)
    f_mg.apply(mg, out)
    assert all_almost_equal(out, func_2d_mg(mg))

    out = np.empty((5,), dtype=float)  # wrong shape
    with pytest.raises(ValueError):
        f_array.apply(points, out)
    with pytest.raises(ValueError):
        f_mg.apply(mg, out)

    # complex
    l2 = L2(rect, field=odl.ComplexNumbers())
    f_array = l2.element(cfunc_2d_array, cfunc_2d_array_apply,
                         vectorization='array')
    f_mg = l2.element(cfunc_2d_mg, cfunc_2d_mg_apply,
                      vectorization='meshgrid')

    # array version
    out = np.empty((50,), dtype=complex)
    f_array.apply(points, out)
    assert all_almost_equal(out, cfunc_2d_array(points))
    # meshgrid version
    out = np.empty((5, 10), dtype=complex)
    f_mg.apply(mg, out)
    assert all_almost_equal(out, cfunc_2d_mg(mg))


def test_l2_vector_equality():
    rect = odl.Rectangle([0, 0], [1, 2])
    l2 = L2(rect)

    f_novec = l2.element(func_2d_novec)
    f_novec_2 = l2.element(func_2d_novec)

    f_array = l2.element(func_2d_array, vectorization='array')
    f_array_2 = l2.element(func_2d_array, vectorization='array')
    f_array_a = l2.element(func_2d_array, func_2d_array_apply,
                           vectorization='array')
    f_array_a_2 = l2.element(func_2d_array, func_2d_array_apply,
                             vectorization='array')

    f_mg = l2.element(func_2d_mg, vectorization='meshgrid')
    f_mg_2 = l2.element(func_2d_mg, vectorization='meshgrid')
    f_mg_a = l2.element(func_2d_mg, func_2d_mg_apply,
                        vectorization='meshgrid')
    f_mg_a_2 = l2.element(func_2d_mg, func_2d_mg_apply,
                          vectorization='meshgrid')

    assert f_novec == f_novec
    assert f_novec == f_novec_2
    assert f_novec != f_array
    assert f_novec != f_mg

    assert f_array == f_array_2
    assert f_array_a == f_array_a_2
    assert f_array != f_array_a
    assert f_array != f_mg

    assert f_mg == f_mg_2
    assert f_mg_a == f_mg_a_2
    assert f_mg != f_mg_a
    assert f_mg != f_array


def test_l2_zero():
    rect = odl.Rectangle([0, 0], [1, 2])
    points = _points(rect, num=5)
    mg = _meshgrid(rect, shape=(2, 3))

    # real
    l2 = L2(rect)

    zero_novec = l2.zero()
    zero_array = l2.zero(vectorization='array')
    zero_mg = l2.zero(vectorization='meshgrid')

    for p in points:
        assert zero_novec(p) == 0.0

    assert all_equal(zero_array(points), np.zeros(5, dtype=float))
    assert all_equal(zero_mg(mg), np.zeros((2, 3), dtype=float))

    # complex
    l2 = L2(rect, field=odl.ComplexNumbers())

    zero_novec = l2.zero()
    zero_array = l2.zero(vectorization='array')
    zero_mg = l2.zero(vectorization='meshgrid')

    for p in points:
        assert zero_novec(p) == 0.0 + 1j*0.0

    assert all_equal(zero_array(points), np.zeros(5, dtype=complex))
    assert all_equal(zero_mg(mg), np.zeros((2, 3), dtype=complex))


def test_l2_one():
    rect = odl.Rectangle([0, 0], [1, 2])
    points = _points(rect, num=5)
    mg = _meshgrid(rect, shape=(2, 3))

    # real
    l2 = L2(rect)

    one_novec = l2.one()
    one_array = l2.one(vectorization='array')
    one_mg = l2.one(vectorization='meshgrid')

    for p in points:
        assert one_novec(p) == 1.0

    assert all_equal(one_array(points), np.ones(5, dtype=float))
    assert all_equal(one_mg(mg), np.ones((2, 3), dtype=float))

    # complex
    l2 = L2(rect, field=odl.ComplexNumbers())

    one_novec = l2.one()
    one_array = l2.one(vectorization='array')
    one_mg = l2.one(vectorization='meshgrid')

    for p in points:
        assert one_novec(p) == 1.0 + 1j*0.0

    assert all_equal(one_array(points), np.ones(5, dtype=complex))
    assert all_equal(one_mg(mg), np.ones((2, 3), dtype=complex))


def test_l2_lincomb():
    rect = odl.Rectangle([0, 0], [1, 2])
    points = _points(rect, num=5)
    point = points[0]
    mg = _meshgrid(rect, shape=(2, 3))

    # REAL
    l2 = L2(rect)
    a = -1.5
    b = 2.0

    # Note: Special cases and alignment are tested later in the magic methods

    # No vectorization
    true_novec = a * func_2d_novec(point) + b * other_func_2d_novec(point)
    f_novec = l2.element(func_2d_novec)
    g_novec = l2.element(other_func_2d_novec)
    out_novec = l2.element()
    l2.lincomb(a, f_novec, b, g_novec, out_novec)
    assert out_novec(point) == true_novec

    # Array vectorization
    true_array = (a * func_2d_array(points) + b * other_func_2d_array(points))
    f_array = l2.element(func_2d_array, vectorization='array')
    g_array = l2.element(other_func_2d_array, vectorization='array')
    out_array = l2.element(vectorization='array')

    l2.lincomb(a, f_array, b, g_array, out_array)
    assert all_equal(out_array(points), true_array)

    f_array_a = l2.element(func_2d_array, func_2d_array_apply,
                           vectorization='array')
    g_array_a = l2.element(other_func_2d_array, other_func_2d_array_apply,
                           vectorization='array')
    out_array_a = l2.element(vectorization='array')

    l2.lincomb(a, f_array_a, b, g_array_a, out_array_a)
    # Check if out-of-place still works
    assert all_equal(out_array(points), true_array)
    # In-place
    out = np.empty((5,), dtype=float)
    out_array_a.apply(points, out=out)
    assert all_equal(out, true_array)

    # Meshgrid vectorization
    true_mg = (a * func_2d_mg(mg) + b * other_func_2d_mg(mg))
    f_mg = l2.element(func_2d_mg, vectorization='meshgrid')
    g_mg = l2.element(other_func_2d_mg, vectorization='meshgrid')
    out_mg = l2.element(vectorization='meshgrid')

    l2.lincomb(a, f_mg, b, g_mg, out_mg)
    assert all_equal(out_mg(mg), true_mg)

    f_mg_a = l2.element(func_2d_mg, func_2d_mg_apply, vectorization='meshgrid')
    g_mg_a = l2.element(other_func_2d_mg, other_func_2d_mg_apply,
                        vectorization='meshgrid')
    out_mg_a = l2.element(vectorization='meshgrid')

    l2.lincomb(a, f_mg_a, b, g_mg_a, out_mg_a)
    # Check if out-of-place still works
    assert all_equal(out_mg(mg), true_mg)
    # In-place
    out = np.empty((2, 3), dtype=float)
    out_mg_a.apply(mg, out=out)
    assert all_equal(out, true_mg)

    # COMPLEX
    l2 = L2(rect, field=odl.ComplexNumbers())
    a = -1.5 + 1j*7
    b = 2.0 - 1j

    # No vectorization
    true_novec = a * cfunc_2d_novec(point) + b * other_cfunc_2d_novec(point)
    f_novec = l2.element(cfunc_2d_novec)
    g_novec = l2.element(other_cfunc_2d_novec)
    out_novec = l2.element()
    l2.lincomb(a, f_novec, b, g_novec, out_novec)
    assert out_novec(point) == true_novec

    # Array vectorization
    true_array = (a * cfunc_2d_array(points) +
                  b * other_cfunc_2d_array(points))
    f_array = l2.element(cfunc_2d_array, vectorization='array')
    g_array = l2.element(other_cfunc_2d_array, vectorization='array')
    out_array = l2.element(vectorization='array')

    l2.lincomb(a, f_array, b, g_array, out_array)
    assert all_equal(out_array(points), true_array)

    f_array_a = l2.element(cfunc_2d_array, cfunc_2d_array_apply,
                           vectorization='array')
    g_array_a = l2.element(other_cfunc_2d_array, other_cfunc_2d_array_apply,
                           vectorization='array')
    out_array_a = l2.element(vectorization='array')

    l2.lincomb(a, f_array_a, b, g_array_a, out_array_a)
    # Check if out-of-place still works
    assert all_equal(out_array(points), true_array)
    # In-place
    out = np.empty((5,), dtype=complex)
    out_array_a.apply(points, out=out)
    assert all_equal(out, true_array)

    # Meshgrid vectorization
    true_mg = a * cfunc_2d_mg(mg) + b * other_cfunc_2d_mg(mg)
    f_mg = l2.element(cfunc_2d_mg, vectorization='meshgrid')
    g_mg = l2.element(other_cfunc_2d_mg, vectorization='meshgrid')
    out_mg = l2.element(vectorization='meshgrid')

    l2.lincomb(a, f_mg, b, g_mg, out_mg)
    assert all_equal(out_mg(mg), true_mg)

    f_mg_a = l2.element(cfunc_2d_mg, cfunc_2d_mg_apply,
                        vectorization='meshgrid')
    g_mg_a = l2.element(other_cfunc_2d_mg, other_cfunc_2d_mg_apply,
                        vectorization='meshgrid')
    out_mg_a = l2.element(vectorization='meshgrid')

    l2.lincomb(a, f_mg_a, b, g_mg_a, out_mg_a)
    # Check if out-of-place still works
    assert all_equal(out_mg(mg), true_mg)
    # In-place
    out = np.empty((2, 3), dtype=complex)
    out_mg_a.apply(mg, out=out)
    assert all_equal(out, true_mg)

    with pytest.raises(ValueError):
        l2.lincomb(a, f_mg, b, g_array, out_mg)

    with pytest.raises(ValueError):
        l2.lincomb(a, f_novec, b, g_novec, out_mg)


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
    point = points[0]
    mg = _meshgrid(rect, shape=(2, 3))
    a = -1.5
    b = 2.0
    array_out = np.empty((5,), dtype=float)
    mg_out = np.empty((2, 3), dtype=float)

    f_novec = l2.element(func_2d_novec, vectorization='none')
    f_array = l2.element(func_2d_array, vectorization='array')
    f_array_a = l2.element(func_2d_array, func_2d_array_apply,
                           vectorization='array')
    f_mg = l2.element(func_2d_mg, vectorization='meshgrid')
    f_mg_a = l2.element(func_2d_mg, func_2d_mg_apply,
                        vectorization='meshgrid')
    g_novec = l2.element(other_func_2d_novec, vectorization='none')
    g_array = l2.element(other_func_2d_array, vectorization='array')
    g_array_a = l2.element(other_func_2d_array, other_func_2d_array_apply,
                           vectorization='array')
    g_mg = l2.element(other_func_2d_mg, vectorization='meshgrid')
    g_mg_a = l2.element(other_func_2d_mg, other_func_2d_mg_apply,
                        vectorization='meshgrid')

    out_novec = l2.element(vectorization='novec')
    out_array = l2.element(vectorization='array')
    out_array_a = l2.element(vectorization='array')
    out_mg = l2.element(vectorization='meshgrid')
    out_mg_a = l2.element(vectorization='meshgrid')

    if pattern[0] in ('v', 'i'):
        true_l_novec = func_2d_novec(point)
        true_l_array = func_2d_array(points)
        true_l_mg = func_2d_mg(mg)

        test_l_novec = f_novec
        test_l_array = f_array
        test_l_array_a = f_array_a
        test_l_mg = f_mg
        test_l_mg_a = f_mg_a
    else:  # 's'
        true_l_novec = true_l_array = true_l_mg = a
        test_l_novec = test_l_array = test_l_array_a = test_l_mg = test_l_mg_a = a

    if pattern[1] == 'v':
        true_r_novec = other_func_2d_novec(point)
        true_r_array = other_func_2d_array(points)
        true_r_mg = other_func_2d_mg(mg)

        test_r_novec = g_novec
        test_r_array = g_array
        test_r_array_a = g_array_a
        test_r_mg = g_mg
        test_r_mg_a = g_mg_a
    else:  # 's'
        true_r_novec = true_r_array = true_r_mg = b
        test_r_novec = test_r_array = test_r_array_a = test_r_mg = test_r_mg_a = a

    if op_str == '+':
        if pattern[0] == 'i':
            true_novec = true_l_novec + true_r_novec
            true_array = true_l_array + true_r_array
            true_mg = true_l_mg + true_r_mg

            test_l_novec += test_r_novec
            test_l_array += test_r_array
            test_l_array_a += test_r_array_a
            test_l_mg += test_r_mg
            test_l_mg_a += test_r_mg_a
        else:
            out_novec = test_l_novec + test_r_novec
            out_array = test_l_array + test_r_array
            out_array_a = test_l_array_a + test_r_array_a
            out_mg = test_l_mg + test_r_mg
            out_mg_a = test_l_mg_a + test_r_mg_a
    elif op_str == '-':
        if pattern[0] == 'i':
            true_novec = true_l_novec - true_r_novec
            true_array = true_l_array - true_r_array
            true_mg = true_l_mg - true_r_mg

            test_l_novec -= test_r_novec
            test_l_array -= test_r_array
            test_l_array_a -= test_r_array_a
            test_l_mg -= test_r_mg
            test_l_mg_a -= test_r_mg_a
        else:
            out_novec = test_l_novec - test_r_novec
            out_array = test_l_array - test_r_array
            out_array_a = test_l_array_a - test_r_array_a
            out_mg = test_l_mg - test_r_mg
            out_mg_a = test_l_mg_a - test_r_mg_a
    elif op_str == '*':
        if pattern[0] == 'i':
            true_novec = true_l_novec * true_r_novec
            true_array = true_l_array * true_r_array
            true_mg = true_l_mg * true_r_mg

            test_l_novec *= test_r_novec
            test_l_array *= test_r_array
            test_l_array_a *= test_r_array_a
            test_l_mg *= test_r_mg
            test_l_mg_a *= test_r_mg_a
        else:
            out_novec = test_l_novec * test_r_novec
            out_array = test_l_array * test_r_array
            out_array_a = test_l_array_a * test_r_array_a
            out_mg = test_l_mg * test_r_mg
            out_mg_a = test_l_mg_a * test_r_mg_a
    elif op_str == '/':
        if pattern[0] == 'i':
            true_novec = true_l_novec / true_r_novec
            true_array = true_l_array / true_r_array
            true_mg = true_l_mg / true_r_mg

            test_l_novec /= test_r_novec
            test_l_array /= test_r_array
            test_l_array_a /= test_r_array_a
            test_l_mg /= test_r_mg
            test_l_mg_a /= test_r_mg_a
        else:
            out_novec = test_l_novec / test_r_novec
            out_array = test_l_array / test_r_array
            out_array_a = test_l_array_a / test_r_array_a
            out_mg = test_l_mg / test_r_mg
            out_mg_a = test_l_mg_a / test_r_mg_a

    if pattern[0] == 'i':
        assert test_l_novec(point) == true_novec
        assert all_equal(test_l_array(points), true_array)
        test_l_array.apply(points, out=array_out)
        assert all_equal(array_out, true_array)
        assert all_equal(test_l_mg(mg), true_mg)
        test_l_mg.apply(mg, out=mg_out)
        assert all_equal(mg_out, true_mg)
    else:
        assert out_novec(point) == true_novec
        assert all_equal(out_array(points), true_array)
        out_array_a.apply(points, out=array_out)
        assert all_equal(array_out, true_array)
        assert all_equal(out_mg(mg), true_mg)
        out_mg_a.apply(mg, out=mg_out)
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
