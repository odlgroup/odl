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

"""Unit tests for convolutions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import numpy as np
from scipy.signal import convolve

import odl
from odl.util.testutils import all_almost_equal


# ---- RealSpaceConvolution ---- #


def test_real_conv_init_errors():
    # Test if init code runs and errors are raised properly

    discr = odl.uniform_discr([-1, 0], [1, 1], [5, 10])
    subdiscr = odl.uniform_discr_fromdiscr(discr, nsamples=[3, 3])

    full_ker = discr.one()
    full_ker_arr = full_ker.asarray()
    small_ker = subdiscr.one()
    small_ker_arr = small_ker.asarray()

    odl.trafos.RealSpaceConvolution(discr, full_ker)
    odl.trafos.RealSpaceConvolution(discr, full_ker_arr)
    odl.trafos.RealSpaceConvolution(discr, small_ker)
    odl.trafos.RealSpaceConvolution(discr, small_ker_arr)

    cdiscr = odl.uniform_discr([-1, 0], [1, 1], [5, 10], dtype=complex)
    odl.trafos.RealSpaceConvolution(cdiscr, small_ker_arr)

    with pytest.raises(TypeError):
        odl.trafos.RealSpaceConvolution(odl.rn(5), [1, 1, 1])

    with pytest.raises(ValueError):
        fspace = odl.FunctionSpace(odl.Interval(0, 1))
        grid = odl.TensorGrid([0, 0.4, 1])
        part = odl.RectPartition(fspace.domain, grid)
        nonuni_discr = odl.DiscreteLp(fspace, part, odl.rn(3))
        odl.trafos.RealSpaceConvolution(nonuni_discr, [1, 1, 1])

    with pytest.raises(NotImplementedError):
        odl.trafos.RealSpaceConvolution(discr, small_ker_arr, range=discr)


def test_real_conv_range():

    # midpoint: [0.0, 0.5]
    discr = odl.uniform_discr([-1, 0], [1, 1], [5, 10])
    # midpoint: [0.6, 0.15]
    subdiscr = odl.uniform_discr_fromdiscr(discr, min_corner=[0, 0],
                                           nsamples=[3, 3])

    full_ker = discr.one()
    full_ker_arr = full_ker.asarray()
    small_ker = subdiscr.one()
    small_ker_arr = small_ker.asarray()

    def func_ker(x):
        return sum(x)

    # arrays and callables lead to domain == range
    conv_full_arr = odl.trafos.RealSpaceConvolution(discr, full_ker_arr)
    assert conv_full_arr.range == conv_full_arr.domain
    conv_small_arr = odl.trafos.RealSpaceConvolution(discr, small_ker_arr)
    assert conv_small_arr.range == conv_small_arr.domain
    conv_func = odl.trafos.RealSpaceConvolution(discr, func_ker)
    assert conv_func.range == conv_func.domain

    # elements of discr a subspace induce a translation
    conv_full = odl.trafos.RealSpaceConvolution(discr, full_ker)
    true_range = odl.uniform_discr_fromdiscr(discr, min_corner=[None, 0.5])
    assert conv_full.range.partition.approx_equals(true_range.partition,
                                                   atol=1e-6)
    conv_small = odl.trafos.RealSpaceConvolution(discr, small_ker)
    true_range = odl.uniform_discr_fromdiscr(discr, min_corner=[-0.4, 0.15])
    assert conv_small.range.partition.approx_equals(true_range.partition,
                                                    atol=1e-6)


def test_real_conv_kernel():

    discr = odl.uniform_discr([-1, 0], [1, 1], [5, 10])
    subdiscr = odl.uniform_discr_fromdiscr(discr, min_corner=[0, 0],
                                           nsamples=[3, 3])
    std_ker_discr = odl.uniform_discr([-1, -0.5], [1, 0.5], [5, 10])

    full_ker = discr.one()
    full_ker_arr = full_ker.asarray()
    small_ker = subdiscr.one()
    small_ker_arr = small_ker.asarray()

    def func_ker(x):
        return sum(x)

    conv_full = odl.trafos.RealSpaceConvolution(discr, full_ker)
    assert conv_full.kernel() == full_ker

    conv_full_arr = odl.trafos.RealSpaceConvolution(discr, full_ker_arr)
    assert np.array_equal(conv_full_arr.kernel(), full_ker_arr)
    assert np.allclose(conv_full_arr.kernel().space.partition.midpoint, 0)
    assert conv_full_arr.kernel().shape == discr.shape

    conv_small = odl.trafos.RealSpaceConvolution(discr, small_ker)
    assert conv_small.kernel() == small_ker

    conv_small_arr = odl.trafos.RealSpaceConvolution(discr, small_ker_arr)
    assert np.array_equal(conv_small_arr.kernel(), small_ker_arr)
    assert np.allclose(conv_small_arr.kernel().space.partition.midpoint, 0)
    assert conv_small_arr.kernel().shape == subdiscr.shape

    conv_func = odl.trafos.RealSpaceConvolution(discr, func_ker)
    assert conv_func.kernel().space == std_ker_discr
    assert all_almost_equal(conv_func.kernel(),
                            std_ker_discr.element(func_ker))

    def func_ker_p(x, **kwargs):
        p = kwargs.pop('p', 1.0)
        return x[0] + p * x[1]

    conv_func_p = odl.trafos.RealSpaceConvolution(discr, func_ker_p)
    assert all_almost_equal(conv_func_p.kernel(),
                            std_ker_discr.element(func_ker_p, p=1.0))

    conv_func_p = odl.trafos.RealSpaceConvolution(
        discr, func_ker_p, kernel_kwargs={'p': 0.5})
    assert all_almost_equal(conv_func_p.kernel(),
                            std_ker_discr.element(func_ker_p, p=0.5))


def test_real_conv_resampling():

    discr = odl.uniform_discr([-1, 0], [1, 1], [5, 10])
    ker_discr = odl.uniform_discr([-1, -0.25], [1, 0.25], [20, 2])
    kernel = ker_discr.one()

    cs_up = [0.1, 0.1]
    discr_up = odl.uniform_discr_fromdiscr(discr, cell_sides=cs_up)
    ker_discr_up = odl.uniform_discr_fromdiscr(ker_discr, cell_sides=cs_up)
    conv_up = odl.trafos.RealSpaceConvolution(discr, kernel, resample='up')
    assert conv_up._domain_resampling_op.domain == discr
    assert conv_up._domain_resampling_op.range == discr_up
    assert conv_up._kernel_resampling_op.domain == ker_discr
    assert conv_up._kernel_resampling_op.range == ker_discr_up

    cs_down = [0.4, 0.25]
    discr_down = odl.uniform_discr_fromdiscr(discr, cell_sides=cs_down)
    ker_discr_down = odl.uniform_discr_fromdiscr(ker_discr, cell_sides=cs_down)
    conv_down = odl.trafos.RealSpaceConvolution(discr, kernel, resample='down')
    assert conv_down._domain_resampling_op.domain == discr
    assert conv_down._domain_resampling_op.range == discr_down
    assert conv_down._kernel_resampling_op.domain == ker_discr
    assert conv_down._kernel_resampling_op.range == ker_discr_down

    cs_ud = [0.1, 0.25]
    discr_ud = odl.uniform_discr_fromdiscr(discr, cell_sides=cs_ud)
    ker_discr_ud = odl.uniform_discr_fromdiscr(ker_discr, cell_sides=cs_ud)
    conv_ud = odl.trafos.RealSpaceConvolution(discr, kernel,
                                              resample=['UP', 'DOWN'])
    assert conv_ud._domain_resampling_op.domain == discr
    assert conv_ud._domain_resampling_op.range == discr_ud
    assert conv_ud._kernel_resampling_op.domain == ker_discr
    assert conv_ud._kernel_resampling_op.range == ker_discr_ud

    cs_du = [0.4, 0.1]
    discr_du = odl.uniform_discr_fromdiscr(discr, cell_sides=cs_du)
    ker_discr_du = odl.uniform_discr_fromdiscr(ker_discr, cell_sides=cs_du)
    conv_du = odl.trafos.RealSpaceConvolution(discr, kernel,
                                              resample=['down', 'up'])
    assert conv_du._domain_resampling_op.domain == discr
    assert conv_du._domain_resampling_op.range == discr_du
    assert conv_du._kernel_resampling_op.domain == ker_discr
    assert conv_du._kernel_resampling_op.range == ker_discr_du


def test_real_conv_call_1d():

    discr = odl.uniform_discr(0, 1, 5)
    ker_array = np.ones(3)
    conv = odl.trafos.RealSpaceConvolution(discr, ker_array)

    def char(x):
        return (x[0] >= 0) & (x[0] <= 0.5)

    func = discr.element(char)
    func_arr = func.asarray()
    scaling = discr.cell_volume
    assert np.allclose(conv(func),
                       scaling * convolve(func_arr, ker_array, mode='same'))

    # Switched roles
    discr = odl.uniform_discr(0, 1, 3)
    scaling = discr.cell_volume
    conv = odl.trafos.RealSpaceConvolution(discr, func_arr)
    assert np.allclose(conv(ker_array),
                       scaling * convolve(ker_array, func_arr, mode='same'))


def test_real_conv_call_2d():

    discr = odl.uniform_discr([-1, 0], [1, 1], [5, 10])
    ker_array = np.ones((3, 3))
    conv = odl.trafos.RealSpaceConvolution(discr, ker_array)

    def rectangle(x):
        return (x[0] >= 0) & (x[0] <= 0.5) & (x[1] >= 0.5) & (x[1] <= 1)

    func = discr.element(rectangle)
    func_arr = func.asarray()
    scaling = discr.cell_volume
    assert np.allclose(conv(func),
                       scaling * convolve(func_arr, ker_array, mode='same'))


def test_real_conv_adjoint_1d():

    discr = odl.uniform_discr(0, 1, 5)
    ker_array = np.array([-1, 0, 1], dtype=float)
    ker_elem = discr.element([0, -1, 0, 1, 0])
    adj_ker_array = np.array([1, 0, -1], dtype=float)
    adj_ker_elem = discr.element([0, 1, 0, -1, 0])

    def char(x):
        return (x[0] >= 0) & (x[0] <= 0.5)

    conv_elem = odl.trafos.RealSpaceConvolution(discr, ker_elem)
    conv_elem_adj = conv_elem.adjoint
    dom_elem = discr.element(char)
    ran_elem = conv_elem.range.one()
    assert conv_elem_adj.domain == conv_elem.range
    assert conv_elem_adj.range == conv_elem.domain
    assert np.allclose(conv_elem_adj.kernel(), adj_ker_elem)
    assert np.isclose(conv_elem(dom_elem).inner(ran_elem),
                      dom_elem.inner(conv_elem_adj(ran_elem)))

    conv_arr = odl.trafos.RealSpaceConvolution(discr, ker_array)
    conv_arr_adj = conv_arr.adjoint
    dom_elem = discr.element(char)
    ran_elem = conv_arr.range.one()
    assert conv_arr_adj.domain == conv_arr.range
    assert conv_arr_adj.range == conv_arr.domain
    assert np.allclose(conv_arr_adj.kernel(), adj_ker_array)
    assert np.isclose(conv_arr(dom_elem).inner(ran_elem),
                      dom_elem.inner(conv_arr_adj(ran_elem)))


def test_real_conv_adjoint_2d():

    discr = odl.uniform_discr([-1, 0], [1, 1], [5, 10])
    ker_array = np.array([[0, 1, 0],
                          [-1, 0, 1],
                          [0, -1, 0]], dtype=float)
    adj_ker_array = np.array([[0, -1, 0],
                              [1, 0, -1],
                              [0, 1, 0]], dtype=float)

    def char(x):
        return (x[0] > 0) & (x[0] < 0.5) & (x[1] >= 0) & (x[1] <= 0.5)

    conv = odl.trafos.RealSpaceConvolution(discr, ker_array)
    conv_adj = conv.adjoint
    dom_elem = discr.element(char)
    ran_elem = conv.range.one()
    assert conv_adj.domain == conv.range
    assert conv_adj.range == conv.domain
    assert np.allclose(conv_adj.kernel(), adj_ker_array)
    assert np.isclose(conv(dom_elem).inner(ran_elem),
                      dom_elem.inner(conv_adj(ran_elem)))


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
