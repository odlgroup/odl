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
try:
    import pywt
except ImportError:
    pass
import numpy as np
from scipy.signal import convolve

import odl
from odl.trafos.backends.pywt_bindings import (
    PYWT_AVAILABLE,
    pywt_coeff_shapes,
    pywt_flat_array_from_coeffs, pywt_coeffs_from_flat_array,
    pywt_single_level_decomp,
    pywt_multi_level_decomp, pywt_multi_level_recon)
from odl.util.testutils import all_almost_equal, all_equal, noise_array


pytestmark = pytest.mark.skipif(not PYWT_AVAILABLE,
                                reason='`pywt` backend not available')


# --- pytest fixtures --- #


wavelet_params = ['db1', 'sym2']
wavelet_ids = [" wavelet = '{}' ".format(w) for w in wavelet_params]


@pytest.fixture(scope='module', params=wavelet_params, ids=wavelet_ids)
def wavelet(request):
    return request.param


pywt_mode_params = ['zero', 'periodization']
pywt_mode_ids = [" mode = '{}' ".format(m) for m in pywt_mode_params]


@pytest.fixture(scope='module', params=pywt_mode_params, ids=pywt_mode_ids)
def pywt_mode(request):
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
def shape_setup(ndim, wavelet, pywt_mode):
    nlevels = 2

    if ndim == 1:
        image_shape = (16,)
        if wavelet == 'db1':
            coeff_shapes = [(4,), (4,), (8,)]
        elif wavelet == 'sym2':
            if pywt_mode == 'zero':
                coeff_shapes = [(6,), (6,), (9,)]
            elif pywt_mode == 'periodization':
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
            if pywt_mode == 'zero':
                coeff_shapes = [(6, 6), (6, 6), (9, 10)]
            elif pywt_mode == 'periodization':
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
            if pywt_mode == 'zero':
                coeff_shapes = [(6, 6, 6), (6, 6, 6), (9, 10, 10)]
            elif pywt_mode == 'periodization':
                coeff_shapes = [(4, 5, 5), (4, 5, 5), (8, 9, 9)]
            else:
                raise RuntimeError
    else:
        raise RuntimeError

    return wavelet, pywt_mode, nlevels, image_shape, coeff_shapes


@pytest.fixture(scope='module')
def small_shapes(ndim):
    if ndim == 1:
        shapes = [(2,), 2, (3,)]  # contains integer shape
    elif ndim == 2:
        shapes = [(2, 2), (2, 2), (3, 4)]
    elif ndim == 3:
        shapes = [(2, 2, 2), (2, 2, 2), (3, 4, 4)]
    else:
        raise RuntimeError

    return ndim, shapes


# --- helper functions --- #


def _grouped_and_flat_arrays(shapes, dtype):
    """Return a grouped and flat list of arrays with specified shapes.

    The lists are constructed as if they were used in a wavelet transform,
    i.e. the array with shape ``shapes[0]`` appears once, while the
    others appear ``2 ** ndim - 1`` times each.
    """
    space = odl.discr_sequence_space(shape=shapes[0], dtype=dtype)
    array = noise_array(space).reshape(space.shape)
    grouped_list = [array]
    flat_list = [array.ravel()]
    ndim = space.ndim

    for shape in shapes[1:]:
        space = odl.discr_sequence_space(shape=shape, dtype=dtype)
        arrays = [noise_array(space).reshape(shape)
                  for _ in range(2 ** ndim - 1)]
        grouped_list.append(tuple(arrays))
        flat_list.extend([arr.ravel() for arr in arrays])

    return grouped_list, flat_list


# --- unit tests --- #


def test_pywt_coeff_shapes(shape_setup):
    wavelet, pywt_mode, nlevels, image_shape, coeff_shapes = shape_setup
    shapes = pywt_coeff_shapes(image_shape, wavelet, nlevels, pywt_mode)
    assert all_equal(shapes, coeff_shapes)


def test_pywt_coeff_list_conversion(small_shapes, floating_dtype):
    """Test if converstion flat array <-> coefficient list works."""
    ndim, shapes = small_shapes

    grouped_list, flat_list = _grouped_and_flat_arrays(shapes, dtype=float)

    true_flat_array = np.hstack(flat_list)
    flat_array = pywt_flat_array_from_coeffs(grouped_list)
    assert all_equal(flat_array, true_flat_array)

    coeff_list = pywt_coeffs_from_flat_array(flat_array, shapes)
    true_coeff_list = grouped_list
    assert all_equal(coeff_list, true_coeff_list)


def test_multilevel_recon_inverts_decomp(shape_setup, floating_dtype):
    """Test that reco is the inverse of decomp."""
    wavelet, pywt_mode, nlevels, image_shape, coeff_shapes = shape_setup

    image = np.random.uniform(size=image_shape).astype(floating_dtype)
    wave_decomp = pywt_multi_level_decomp(image, wavelet, nlevels, pywt_mode)
    wave_recon = pywt_multi_level_recon(wave_decomp, wavelet, pywt_mode,
                                        image_shape)
    assert wave_recon.shape == image.shape
    assert all_almost_equal(wave_recon, image)


def test_multilevel_decomp_inverts_recon(shape_setup):
    """Test that decomp is the inverse of recon."""
    dtype = 'float64'  # when fixed, use dtype fixture instead
    wavelet, pywt_mode, nlevels, image_shape, coeff_shapes = shape_setup

    if not ((ndim == 1 and
             wavelet == 'sym2' and
             pywt_mode == 'periodization') or
            (ndim == 1 and
             wavelet == 'db1' and
             pywt_mode in ('zero', 'periodization'))):
        # The reverse invertibility is not given since the wavelet
        # decomposition as implemented in PyWavelets, is not left-invertible.
        # Only some setups work by miracle.
        # TODO: investigate further
        pytest.xfail('not left-invertible')

    coeffs, _ = _grouped_and_flat_arrays(coeff_shapes, dtype)
    wave_recon = pywt_multi_level_recon(coeffs, wavelet, pywt_mode,
                                        recon_shape=image_shape)
    wave_decomp = pywt_multi_level_decomp(wave_recon, wavelet, nlevels,
                                          pywt_mode)
    assert all_almost_equal(coeffs, wave_decomp)


def test_explicit_example(floating_dtype):
    """Comparison with hand-calculated wavelet transform."""

    space = odl.uniform_discr([0, 0], [1, 1], (16, 15), dtype=floating_dtype)
    x = noise_array(space).reshape(space.shape)

    # We use a Daubechies-2 wavelet
    wavelet = pywt.Wavelet('db2')
    filter_l = np.array(wavelet.dec_lo)
    filter_h = np.array(wavelet.dec_hi)

    # Build the 2D filters
    filter_ll = filter_l[:, None] * filter_l[None, :]
    filter_lh = filter_l[:, None] * filter_h[None, :]
    filter_hl = filter_h[:, None] * filter_l[None, :]
    filter_hh = filter_h[:, None] * filter_h[None, :]

    # Convolve x with 2D filters (implicitly uses zero-padding)
    conv_ll = convolve(x, filter_ll)
    conv_lh = convolve(x, filter_lh)
    conv_hl = convolve(x, filter_hl)
    conv_hh = convolve(x, filter_hh)

    # Downsampling by factor 2, taking the odd indices, gives the wavelet
    # coefficients
    coeff_aa = conv_ll[1::2, 1::2]
    coeff_ad = conv_lh[1::2, 1::2]
    coeff_da = conv_hl[1::2, 1::2]
    coeff_dd = conv_hh[1::2, 1::2]

    # Compare with single-level wavelet trafo (zero padding)
    coeffs = pywt_single_level_decomp(x, wavelet='db2', mode='zero')
    approx, details = coeffs

    assert all_almost_equal(approx, coeff_aa)
    assert all_almost_equal(details, [coeff_ad, coeff_da, coeff_dd])

    # Second level, continuing with the level 1 approximation coefficient
    coeff_2_aa = convolve(coeff_aa, filter_ll)[1::2, 1::2]
    coeff_2_ad = convolve(coeff_aa, filter_lh)[1::2, 1::2]
    coeff_2_da = convolve(coeff_aa, filter_hl)[1::2, 1::2]
    coeff_2_dd = convolve(coeff_aa, filter_hh)[1::2, 1::2]

    # Compare with multi-level wavelet trafo (zero padding)
    coeffs = pywt_multi_level_decomp(x, wavelet='db2', mode='zero', nlevels=2)
    approx_2, details_2, details_1 = coeffs

    assert all_almost_equal(approx_2, coeff_2_aa)
    assert all_almost_equal(details_1, [coeff_ad, coeff_da, coeff_dd])
    assert all_almost_equal(details_2, [coeff_2_ad, coeff_2_da, coeff_2_dd])


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
