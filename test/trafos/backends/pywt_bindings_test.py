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

import pytest
import numpy as np

from odl.trafos.backends.pywt_bindings import (
    PYWT_AVAILABLE,
    pywt_coeff_shapes,
    pywt_flat_array_from_coeffs, pywt_coeffs_from_flat_array,
    pywt_multi_level_decomp, pywt_multi_level_recon)
from odl.util.testutils import all_almost_equal, all_equal

pytestmark = pytest.mark.skipif(not PYWT_AVAILABLE,
                                reason='`pywt` backend not available')


wavelet_params = ['db1', 'sym2']
wavelet_ids = [" wavelet = '{}' ".format(w) for w in wavelet_params]


@pytest.fixture(scope='module', params=wavelet_params, ids=wavelet_ids)
def wavelet(request):
    return request.param


mode_params = ['zpd', 'per']
mode_ids = [" mode = '{}' ".format(m) for m in mode_params]


@pytest.fixture(scope='module', params=mode_params, ids=mode_ids)
def mode(request):
    return request.param


ndim_params = [1, 2, 3]
ndim_ids = [' ndim = {} '.format(ndim) for ndim in ndim_params]


@pytest.fixture(scope='module', params=ndim_params, ids=ndim_ids)
def ndim(request):
    return request.param


nlevels_params = [1, 3]
nlevels_ids = [' nlevels = {} '.format(nlevels) for nlevels in nlevels_params]


@pytest.fixture(scope='module', params=nlevels_params, ids=nlevels_ids)
def nlevels(request):
    return request.param


@pytest.fixture(scope='module')
def shape_setup(ndim, wavelet, mode):
    nlevels = 2

    if ndim == 1:
        image_shape = (16,)
        if wavelet == 'db1':
            coeff_shapes = [(4,), (4,), (8,)]
        elif wavelet == 'sym2':
            if mode == 'zpd':
                coeff_shapes = [(6,), (6,), (9,)]
            elif mode == 'per':
                coeff_shapes = [(4,), (4,), (8,)]
            else:
                raise RuntimeError
        else:
            raise RuntimeError

    elif ndim == 2:
        image_shape = (16, 17)
        if wavelet == 'db1':
            coeff_shapes = [(4, 5), (4, 5), (8, 9)]
        elif wavelet == 'sym2':
            if mode == 'zpd':
                coeff_shapes = [(6, 6), (6, 6), (9, 10)]
            elif mode == 'per':
                coeff_shapes = [(4, 5), (4, 5), (8, 9)]
            else:
                raise RuntimeError
        else:
            raise RuntimeError

    elif ndim == 3:
        image_shape = (16, 17, 18)
        if wavelet == 'db1':
            coeff_shapes = [(4, 5, 5), (4, 5, 5), (8, 9, 9)]
        elif wavelet == 'sym2':
            if mode == 'zpd':
                coeff_shapes = [(6, 6, 6), (6, 6, 6), (9, 10, 10)]
            elif mode == 'per':
                coeff_shapes = [(4, 5, 5), (4, 5, 5), (8, 9, 9)]
            else:
                raise RuntimeError
    else:
        raise RuntimeError

    return wavelet, mode, nlevels, image_shape, coeff_shapes


dtype_params = ['float32', 'float64']
dtype_ids = [' dtype = {} '.format(dt) for dt in dtype_params]


@pytest.fixture(scope="module", ids=dtype_ids, params=dtype_params)
def dtype(request):
    return request.param


def test_pywt_coeff_shapes(shape_setup):
    wavelet, mode, nlevels, image_shape, coeff_shapes = shape_setup
    shapes = pywt_coeff_shapes(image_shape, wavelet, nlevels, mode)
    assert all_equal(shapes, coeff_shapes)


@pytest.fixture(scope='module')
def small_shapes(ndim):
    if ndim == 1:
        shapes = [(2,), (2,), (3,)]
    elif ndim == 2:
        shapes = [(2, 2), (2, 2), (3, 4)]
    elif ndim == 3:
        shapes = [(2, 2, 2), (2, 2, 2), (3, 4, 4)]
    else:
        raise RuntimeError

    return ndim, shapes


def _grouped_and_flat_arrays(shapes):
    """Return a grouped and flat list of arrays with specified shapes.

    The lists are constructed as if they were used in a wavelet transform,
    i.e. the array with shape ``shapes[0]`` appears once, while the
    others appear ``2 ** ndim - 1`` times each.
    """
    array = np.random.uniform(size=shapes[0])
    grouped_list = [array]
    flat_list = [array.ravel()]
    ndim = len(shapes[0])

    for shape in shapes[1:]:
        arrays = [np.random.uniform(size=shape)
                  for _ in range(2 ** ndim - 1)]
        grouped_list.append(tuple(arrays))
        flat_list.extend([arr.ravel() for arr in arrays])

    return grouped_list, flat_list


def test_pywt_coeff_list_conversion(small_shapes):
    ndim, shapes = small_shapes

    grouped_list, flat_list = _grouped_and_flat_arrays(shapes)

    true_flat_array = np.hstack(flat_list)
    flat_array = pywt_flat_array_from_coeffs(grouped_list)
    assert all_equal(flat_array, true_flat_array)

    coeff_list = pywt_coeffs_from_flat_array(flat_array, shapes)
    true_coeff_list = grouped_list
    assert all_equal(coeff_list, true_coeff_list)


def test_multilevel_decomp_and_recon(shape_setup, dtype):
    wavelet, mode, nlevels, image_shape, coeff_shapes = shape_setup

    # Test invertibility the decomposition
    image = np.random.uniform(size=image_shape).astype(dtype)
    wave_decomp = pywt_multi_level_decomp(image, wavelet, nlevels, mode)
    wave_recon = pywt_multi_level_recon(wave_decomp, wavelet, mode,
                                        image_shape)
    assert wave_recon.shape == image.shape
    assert all_almost_equal(wave_recon, image)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
