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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External module imports
import pytest
import numpy as np
try:
    import pywt
except ImportError:
    pass

# ODL imports
import odl
from odl.trafos.wavelet import (
    coeff_size_list_axes, coeff_size_list, pywt_list_to_array,
    pywt_dict_to_array, array_to_pywt_list, array_to_pywt_dict,
    WaveletTransform)
from odl.util.testutils import (all_almost_equal, all_equal,
                                skip_if_no_pywavelets)


@skip_if_no_pywavelets
def test_coeff_size_list():
    # Verify that the helper function does indeed work as expected
    shape = (16,)
    nscale = 3
    wbasis = pywt.Wavelet('db1')
    mode = 'per'
    size_list1d = coeff_size_list(shape, nscale, wbasis, mode)
    S1d = [(2,), (2,), (4,), (8,), (16,)]
    shape = (16, 16)
    size_list2d = coeff_size_list(shape, nscale, wbasis, mode)
    S2d = [(2, 2), (2, 2), (4, 4), (8, 8), (16, 16)]
    shape = (16, 16, 16)
    size_list3d = coeff_size_list(shape, nscale, wbasis, mode)
    S3d = [(2, 2, 2), (2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16)]
    assert all_equal(size_list1d, S1d)
    assert all_equal(size_list2d, S2d)
    assert all_equal(size_list3d, S3d)


@skip_if_no_pywavelets
def test_coeff_size_list_axes():
    # 2D
    n = 16
    wbasis = pywt.Wavelet('db1')
    mode = 'sym'
    axes = [0, 0]
    size_list = coeff_size_list_axes((n, n), wbasis, mode, axes)
    sl = [(4, 16), (4, 16), (16, 16)]
    assert all_equal(size_list, sl)
    coeff_dict = pywt.dwtn(np.random.rand(n, n), wbasis, mode, axes)
    assert all_equal(np.shape(coeff_dict['aa']), size_list[0])

    # 3D
    axes = [0, 0, 1]
    size_list = coeff_size_list_axes((n, n, n), wbasis, mode, axes)
    sl = [(4, 8, 16), (4, 8, 16), (16, 16, 16)]
    assert all_equal(size_list, sl)
    coeff_dict = pywt.dwtn(np.random.rand(n, n, n), wbasis, mode, axes)
    assert all_equal(np.shape(coeff_dict['aaa']), size_list[0])


@skip_if_no_pywavelets
def test_pywt_list_to_array_and_array_to_pywt_list():
    # Verify that the helper function does indeed work as expected
    wbasis = pywt.Wavelet('db1')
    mode = 'zpd'
    nscales = 2
    n = 16
    # 1D test
    size_list = coeff_size_list((n,), nscales, wbasis, mode)
    x = np.random.rand(n)
    coeff_list = pywt.wavedec(x, wbasis, mode, nscales)
    coeff_arr = pywt_list_to_array(coeff_list, size_list)
    assert isinstance(coeff_arr, (np.ndarray))
    length_of_array = np.prod(size_list[0])
    length_of_array += sum(np.prod(shape) for shape in size_list[1:-1])
    assert all_equal(len(coeff_arr), length_of_array)

    coeff_list2 = array_to_pywt_list(coeff_arr, size_list)
    assert all_equal(coeff_list, coeff_list2)
    reconstruction = pywt.waverec(coeff_list2, wbasis, mode)
    assert all_almost_equal(reconstruction, x)

    # 2D test
    size_list = coeff_size_list((n, n), nscales, wbasis, mode)
    x = np.random.rand(n, n)
    coeff_list = pywt.wavedec2(x, wbasis, mode, nscales)
    coeff_arr = pywt_list_to_array(coeff_list, size_list)
    assert isinstance(coeff_arr, (np.ndarray))
    length_of_array = np.prod(size_list[0])
    length_of_array += sum(3 * np.prod(shape) for shape in size_list[1:-1])
    assert all_equal(len(coeff_arr), length_of_array)

    coeff_list2 = array_to_pywt_list(coeff_arr, size_list)
    assert all_equal(coeff_list, coeff_list2)
    reconstruction = pywt.waverec2(coeff_list2, wbasis, mode)
    assert all_almost_equal(reconstruction, x)

    # 3D test
    size_list = coeff_size_list((n, n, n), nscales, wbasis, mode)
    x = np.random.rand(n, n, n)
    coeff_list = pywt.wavedecn(x, wbasis, mode, nscales)
    coeff_arr = pywt_list_to_array(coeff_list, size_list)
    assert isinstance(coeff_arr, (np.ndarray))
    length_of_array = np.prod(size_list[0])
    length_of_array += sum(7 * np.prod(shape) for shape in size_list[1:-1])
    assert len(coeff_arr) == length_of_array

    coeff_list2 = array_to_pywt_list(coeff_arr, size_list)
    reconstruction = pywt.waverecn(coeff_list2, wbasis, mode)
    assert all_equal(coeff_list, coeff_list2)
    assert all_almost_equal(reconstruction, x)


@skip_if_no_pywavelets
def test_pywt_dict_to_array_and_array_to_pywt_dict():
    # Verify that the helper function does indeed work as expected
    n = 16
    wbasis = pywt.Wavelet('db1')
    mode = 'zpd'
    # 2D
    axes = [0, 0]
    size_list = coeff_size_list_axes((n, n), wbasis, mode, axes)
    x = np.random.rand(n, n)
    coeff_dict = pywt.dwtn(x, wbasis, mode, axes)
    coeff_arr = pywt_dict_to_array(coeff_dict, size_list)
    assert isinstance(coeff_arr, (np.ndarray))
    length_of_array = np.prod(size_list[0])
    length_of_array += sum(3 * np.prod(shape) for shape in size_list[1:-1])
    assert len(coeff_arr) == length_of_array

    coeff_dict_converted = array_to_pywt_dict(coeff_arr, size_list)
    aa_orig = coeff_dict['aa']
    aa_converter = coeff_dict_converted['aa']
    assert all_almost_equal(aa_orig, aa_converter)
    reconstruction = pywt.idwtn(coeff_dict_converted, wbasis, mode, axes)
    assert all_almost_equal(reconstruction, x)

    # 3D
    axes = [0, 0, 0]
    size_list = coeff_size_list_axes((n, n, n), wbasis, mode, axes)

    x = np.random.rand(n, n, n)
    coeff_dict = pywt.dwtn(x, wbasis, mode, axes)
    coeff_arr = pywt_dict_to_array(coeff_dict, size_list)
    assert isinstance(coeff_arr, (np.ndarray))
    length_of_array = np.prod(size_list[0])
    length_of_array += sum(7 * np.prod(shape) for shape in size_list[1:-1])
    assert len(coeff_arr) == length_of_array

    coeff_dict_converted = array_to_pywt_dict(coeff_arr, size_list)
    aaa_orig = coeff_dict['aaa']
    aaa_converter = coeff_dict_converted['aaa']
    assert all_almost_equal(aaa_orig, aaa_converter)
    reconstruction = pywt.idwtn(coeff_dict_converted, wbasis, mode, axes)
    assert all_almost_equal(reconstruction, x)


@skip_if_no_pywavelets
def test_dwt():
    # Verify that the operator works as axpected
    # 1D test
    n = 16
    x = np.zeros(n)
    x[5:10] = 1
    wbasis = pywt.Wavelet('db1')
    nscales = 2
    mode = 'sym'
    size_list = coeff_size_list((n,), nscales, wbasis, mode)

    # Define a discretized domain
    disc_domain = odl.uniform_discr([-1], [1], [n], dtype='float32')
    disc_phantom = disc_domain.element(x)

    # Create the discrete wavelet transform operator.
    # Only the domain of the operator needs to be defined
    Wop = WaveletTransform(disc_domain, nscales, wbasis, mode)

    # Compute the discrete wavelet transform of discrete imput image
    coeffs = Wop(disc_phantom)

    # Determine the correct range for Wop and verify that coeffs
    # is an element of it
    ran_size = np.prod(size_list[0])
    ran_size += sum(np.prod(shape) for shape in size_list[1:-1])
    disc_range = disc_domain.dspace_type(ran_size, dtype=disc_domain.dtype)
    assert coeffs in disc_range

    # Compute the inverse wavelet transform
    reconstruction = Wop.inverse(coeffs)

    # Verify that reconstructions lie in correct discretized domain
    assert reconstruction in disc_domain
    assert all_almost_equal(reconstruction.asarray(), x)

    # ---------------------------------------------------------------
    # 2D test
    n = 16
    x = np.zeros((n, n))
    x[5:10, 5:10] = 1
    wbasis = pywt.Wavelet('db1')
    nscales = 2
    mode = 'sym'
    size_list = coeff_size_list((n, n), nscales, wbasis, mode)

    # Define a discretized domain
    disc_domain = odl.uniform_discr([-1] * 2, [1] * 2, [n] * 2,
                                    dtype='float32')
    disc_phantom = disc_domain.element(x)

    # Create the discrete wavelet transform operator.
    # Only the domain of the operator needs to be defined
    Wop = WaveletTransform(disc_domain, nscales, wbasis, mode)

    # Compute the discrete wavelet transform of discrete imput image
    coeffs = Wop(disc_phantom)

    # Determine the correct range for Wop and verify that coeffs
    # is an element of it
    ran_size = np.prod(size_list[0])
    ran_size += sum(3 * np.prod(shape) for shape in size_list[1:-1])
    disc_range = disc_domain.dspace_type(ran_size, dtype=disc_domain.dtype)
    assert coeffs in disc_range

    # Compute the inverse wavelet transform
    reconstruction = Wop.inverse(coeffs)

    # Verify that reconstructions lie in correct discretized domain
    assert reconstruction in disc_domain
    assert all_almost_equal(reconstruction.asarray(), x)

    # -------------------------------------------------------------
    # 3D test
    n = 16
    x = np.zeros((n, n, n))
    x[5:10, 5:10, 5:10] = 1
    wbasis = pywt.Wavelet('db2')
    nscales = 1
    mode = 'per'
    size_list = coeff_size_list((n, n, n), nscales, wbasis, mode)

    # Define a discretized domain
    disc_domain = odl.uniform_discr([-1] * 3, [1] * 3, [n] * 3,
                                    dtype='float32')
    disc_phantom = disc_domain.element(x)

    # Create the discrete wavelet transform operator related to 3D transform.
    Wop = WaveletTransform(disc_domain, nscales, wbasis, mode)
    # Compute the discrete wavelet transform of discrete imput image
    coeffs = Wop(disc_phantom)
    # Determine the correct range for Wop and verify that coeffs
    # is an element of it
    ran_size = np.prod(size_list[0])
    ran_size += sum(7 * np.prod(shape) for shape in size_list[1:-1])
    disc_range = disc_domain.dspace_type(ran_size, dtype=disc_domain.dtype)
    assert coeffs in disc_range

    # Compute the inverse wavelet transform
    reconstruction = Wop.inverse(coeffs)

    # Verify that reconstructions lie in correct discretized domain
    assert reconstruction in disc_domain
    assert all_almost_equal(reconstruction.asarray(), x)


@skip_if_no_pywavelets
def test_axes_option():
    # 2D
    n = 16
    x = np.ones((n, n))
    wbasis = pywt.Wavelet('db1')
    mode = 'sym'
    nscales = 1

    axes = [0, 0]

    size_list = coeff_size_list_axes((n, n), wbasis, mode, axes)

    # Define a discretized domain
    disc_domain = odl.uniform_discr([-1] * 2, [1] * 2, [n] * 2,
                                    dtype='float32')
    disc_phantom = disc_domain.element(x)

    # Create the discrete wavelet transform operator.
    Wop = WaveletTransform(disc_domain, nscales, wbasis, mode, axes=axes)

    # Compute the discrete wavelet transform of discrete imput image
    coeffs = Wop(disc_phantom)

    # Determine the correct range for Wop and verify that coeffs
    # is an element of it
    ran_size = np.prod(size_list[0])
    ran_size += sum(3 * np.prod(shape) for shape in size_list[1:-1])
    disc_range = disc_domain.dspace_type(ran_size, dtype=disc_domain.dtype)
    assert coeffs in disc_range

    # Check that A*A(x) == x
    reconstruction = Wop.adjoint(coeffs)
    # Verify that the output of Wop.inverse and Wop.adjoint are the same
    assert all_almost_equal(reconstruction.asarray(),
                            disc_phantom.asarray())
    # Verify that reconstructions lie in correct discretized domain
    assert reconstruction in disc_domain

    # 3D
    x = np.ones((n, n, n))
    axes = [0, 0, 2]
    size_list = coeff_size_list_axes((n, n, n), wbasis, mode, axes)
    # Define a discretized domain
    disc_domain = odl.uniform_discr([-1] * 3, [1] * 3, [n] * 3,
                                    dtype='float32')
    disc_phantom = disc_domain.element(x)
    # Create the discrete wavelet transform operator.
    Wop = WaveletTransform(disc_domain, nscales, wbasis, mode, axes=axes)
    # Compute the discrete wavelet transform of discrete imput image
    coeffs = Wop(disc_phantom)
    # Check that A*A(x) == x
    reconstruction = Wop.adjoint(coeffs)
    # Verify that the output of Wop.inverse and Wop.adjoint are the same
    assert all_almost_equal(reconstruction.asarray(),
                            disc_phantom.asarray())


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
