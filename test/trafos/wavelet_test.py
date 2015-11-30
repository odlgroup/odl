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

# External module imports
import pytest
import numpy as np
import pywt

# ODL imports
import odl
from odl.trafos.wavelet import (list_of_coeff_sizes, pywt_coeff_to_array2d,
                                array_to_pywt_coeff2d, pywt_coeff_to_array3d,
                                array_to_pywt_coeff3d, wavelet_decomposition3d,
                                wavelet_reconstruction3d, DiscreteWaveletTrafo)
from odl.util.testutils import all_almost_equal, all_equal


def test_list_of_coeffs_sizes():
    # Verify that the helper function does indeed work as expected
    shape = (64, 64)
    nscale = 3
    wbasis = pywt.Wavelet('db2')
    mode = 'per'
    size_list = list_of_coeff_sizes(shape, nscale, wbasis, mode)
    S2d = [(8, 8), (8, 8), (16, 16), (32, 32), (64, 64)]
    shape = (64, 64, 64)
    size_list3d = list_of_coeff_sizes(shape, nscale, wbasis, mode)
    S3d = [(8, 8, 8), (8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64)]
    assert all_equal(size_list, S2d)
    assert all_equal(size_list3d, S3d)


def test_wavelet_decomposition3d_et_reconstruction3d():
    # Test 3D wavelet decomposition and reconstruction and verify that
    # they perform as expected
    x = np.random.rand(16, 16, 16)
    mode = 'sym'
    wbasis = pywt.Wavelet('db5')
    nscales = 1
    wavelet_coeffs = wavelet_decomposition3d(x, wbasis, mode, nscales)
    aaa = wavelet_coeffs[0]
    reference = pywt.dwtn(x, wbasis, mode)
    aaa_reference = reference['aaa']
    assert all_almost_equal(aaa, aaa_reference)

    reconstruction = wavelet_reconstruction3d(wavelet_coeffs, wbasis, mode,
                                              nscales)
    reconstruction_reference = pywt.idwtn(reference, wbasis, mode)
    assert all_almost_equal(reconstruction, reconstruction_reference)
    assert all_almost_equal(reconstruction, x)
    assert all_almost_equal(reconstruction_reference, x)

    wbasis = pywt.Wavelet('db1')
    nscales = 3
    wavelet_coeffs = wavelet_decomposition3d(x, wbasis, mode, nscales)
    shape_true = (nscales+1, )
    assert all_equal(np.shape(wavelet_coeffs), shape_true)

    reconstruction = wavelet_reconstruction3d(wavelet_coeffs, wbasis, mode,
                                              nscales)
    assert all_almost_equal(reconstruction, x)


def test_pywt_coeff_to_array_et_array_to_pywt_coeff2d():
    # Verify that the helper function does indeed work as expected
    wbasis = pywt.Wavelet('db1')
    mode = 'zpd'
    nscales = 2
    n = 16
    size_list = list_of_coeff_sizes((n, n), nscales, wbasis, mode)
    x = np.random.rand(n, n)
    coeff_list = pywt.wavedec2(x, wbasis, mode, nscales)
    coeff_arr = pywt_coeff_to_array2d(coeff_list, size_list, nscales)
    assert isinstance(coeff_arr, (np.ndarray))
    length_of_array = np.prod(size_list[0])
    length_of_array += sum(3 * np.prod(shape) for shape in size_list[1:-1])
    assert all_equal(len(coeff_arr), length_of_array)

    coeff_list2 = array_to_pywt_coeff2d(coeff_arr, size_list, nscales)
    assert all_equal(coeff_list, coeff_list2)
    reconstruction = pywt.waverec2(coeff_list2, wbasis, mode)
    assert all_almost_equal(reconstruction, x)


def test_pywt_coeff_to_array_et_array_to_pywt_coeff3d():
    # Verify that the helper function does indeed work as expected
    wbasis = pywt.Wavelet('db1')
    mode = 'ppd'
    nscales = 2
    n = 16
    size_list = list_of_coeff_sizes((n, n, n), nscales, wbasis, mode)
    x = np.random.rand(n, n, n)
    coeff_dict = wavelet_decomposition3d(x, wbasis, mode, nscales)
    coeff_arr = pywt_coeff_to_array3d(coeff_dict, size_list, nscales)
    assert isinstance(coeff_arr, (np.ndarray))
    length_of_array = np.prod(size_list[0])
    length_of_array += sum(7 * np.prod(shape) for shape in size_list[1:-1])
    assert all_equal(len(coeff_arr), length_of_array)

    coeff_dict2 = array_to_pywt_coeff3d(coeff_arr, size_list, nscales)
    reconstruction = wavelet_reconstruction3d(coeff_dict2, wbasis, mode,
                                              nscales)
    assert all_equal(coeff_dict, coeff_dict)
    assert all_almost_equal(reconstruction, x)


def test_DiscreteWaveletTrafo():
    #Verify that the operator works as axpected
    # 2D test
    n = 16
    x = np.zeros((n, n))
    x[5:10, 5:10] = 1
    wbasis = pywt.Wavelet('db1')
    nscales = 2
    mode = 'sym'
    size_list = list_of_coeff_sizes((n, n), nscales, wbasis, mode)

    # Define a discretized domain
    domain = odl.FunctionSpace(odl.Rectangle([-1, -1], [1, 1]))
    nPoints = np.array([n, n])
    disc_domain = odl.uniform_discr(domain, nPoints)
    disc_phantom = disc_domain.element(x)

    # Create the discrete wavelet transform operator.
    # Only the domain of the operator needs to be defined
    Wop = DiscreteWaveletTrafo(disc_domain, nscales, wbasis, mode)

    # Compute the discrete wavelet transform of discrete imput image
    coeffs = Wop(disc_phantom)

    #Determine the correct range for Wop and verify that coeffs
    # is an element of it
    ran_size = np.prod(size_list[0])
    ran_size += sum(3 * np.prod(shape) for shape in size_list[1:-1])
    disc_range = disc_domain.dspace_type(ran_size, dtype=disc_domain.dtype)
    assert coeffs in disc_range

    # Compute the inverse wavelet transform
    reconstruction1 = Wop.inverse(coeffs)
    # With othogonal wavelets the inverse is the adjoint
    reconstruction2 = Wop.adjoint(coeffs)
    # Verify that the output of Wop.inverse and Wop.adjoint are the same
    assert all_almost_equal(reconstruction1.asarray(),
                            reconstruction2.asarray())

    # Verify that reconstructions lie in correct discretized domain
    assert reconstruction1 in disc_domain
    assert reconstruction2 in disc_domain
    assert all_almost_equal(reconstruction1.asarray(), x)
    assert all_almost_equal(reconstruction2.asarray(), x)

    # 3D test
    n = 16
    x = np.zeros((n, n, n))
    x[5:10, 5:10, 5:10] = 1
    wbasis = pywt.Wavelet('db2')
    nscales = 1
    mode = 'per'
    size_list = list_of_coeff_sizes((n, n, n), nscales, wbasis, mode)

    # Define a discretized domain
    domain = odl.FunctionSpace(odl.Cuboid([-1, -1, -1], [1, 1, 1]))
    nPoints = np.array([n, n, n])
    disc_domain = odl.uniform_discr(domain, nPoints)
    disc_phantom = disc_domain.element(x)

    # Create the discrete wavelet transform operator related to 3D transform.
    Wop = DiscreteWaveletTrafo(disc_domain, nscales, wbasis, mode)
    # Compute the discrete wavelet transform of discrete imput image
    coeffs = Wop(disc_phantom)
    #Determine the correct range for Wop and verify that coeffs
    # is an element of it
    ran_size = np.prod(size_list[0])
    ran_size += sum(7 * np.prod(shape) for shape in size_list[1:-1])
    disc_range = disc_domain.dspace_type(ran_size, dtype=disc_domain.dtype)
    assert coeffs in disc_range

    # Compute the inverse wavelet transform
    reconstruction1 = Wop.inverse(coeffs)
    # With othogonal wavelets the inverse is the adjoint
    reconstruction2 = Wop.adjoint(coeffs)

    # Verify that the output of Wop.inverse and Wop.adjoint are the same
    assert all_almost_equal(reconstruction1, reconstruction2)

    # Verify that reconstructions lie in correct discretized domain
    assert reconstruction1 in disc_domain
    assert reconstruction2 in disc_domain
    assert all_almost_equal(reconstruction1.asarray(), x)
    assert all_almost_equal(reconstruction2, disc_phantom)

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
