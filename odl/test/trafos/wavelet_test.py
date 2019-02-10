# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

import pytest

import odl
from odl.util.testutils import (
    all_almost_equal, noise_element, simple_fixture, skip_if_no_pywavelets)


# --- pytest fixtures --- #


wavelet = simple_fixture('wavelet', ['db1', 'sym2'])
pad_mode = simple_fixture('pad_mode', ['constant', 'pywt_periodic'])
ndim = simple_fixture('ndim', [1, 2, 3])
nlevels = simple_fixture('nlevels', [2, None])
axes = simple_fixture('axes', [-1, None])
wave_impl = simple_fixture(
    'wave_impl',
    [pytest.param('pywt', marks=skip_if_no_pywavelets)]
)


@pytest.fixture(scope='module')
def shape_setup(ndim, nlevels, wavelet, pad_mode):
    if ndim == 1:
        image_shape = (16,)
        if wavelet == 'db1':
            coeff_shapes = [(4,), (4,), (8,)]
        elif wavelet == 'sym2':
            if pad_mode == 'constant':
                coeff_shapes = [(6,), (6,), (9,)]
            elif pad_mode == 'pywt_periodic':
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
            if pad_mode == 'constant':
                coeff_shapes = [(6, 6), (6, 6), (9, 10)]
            elif pad_mode == 'pywt_periodic':
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
            if pad_mode == 'constant':
                coeff_shapes = [(6, 6, 6), (6, 6, 6), (9, 10, 10)]
            elif pad_mode == 'pywt_periodic':
                coeff_shapes = [(4, 5, 5), (4, 5, 5), (8, 9, 9)]
            else:
                raise RuntimeError
    else:
        raise RuntimeError

    return wavelet, pad_mode, nlevels, image_shape, coeff_shapes


def test_wavelet_transform(wave_impl, shape_setup, odl_floating_dtype, axes):
    # Verify that the operator works as expected
    dtype = odl_floating_dtype
    wavelet, pad_mode, nlevels, shape, _ = shape_setup
    ndim = len(shape)

    space = odl.uniform_discr([-1] * ndim, [1] * ndim, shape, dtype=dtype)
    image = noise_element(space)

    # TODO: check more error scenarios
    if wave_impl == 'pywt' and pad_mode == 'constant':
        with pytest.raises(ValueError):
            wave_trafo = odl.trafos.WaveletTransform(
                space, wavelet, nlevels, pad_mode, pad_const=1, impl=wave_impl,
                axes=axes)

    wave_trafo = odl.trafos.WaveletTransform(
        space, wavelet, nlevels, pad_mode, impl=wave_impl, axes=axes)

    assert wave_trafo.domain.dtype == dtype
    assert wave_trafo.range.dtype == dtype

    wave_trafo_inv = wave_trafo.inverse
    assert wave_trafo_inv.domain.dtype == dtype
    assert wave_trafo_inv.range.dtype == dtype
    assert wave_trafo_inv.nlevels == wave_trafo.nlevels
    assert wave_trafo_inv.wavelet == wave_trafo.wavelet
    assert wave_trafo_inv.pad_mode == wave_trafo.pad_mode
    assert wave_trafo_inv.pad_const == wave_trafo.pad_const
    assert wave_trafo_inv.pywt_pad_mode == wave_trafo.pywt_pad_mode

    coeffs = wave_trafo(image)
    reco_image = wave_trafo.inverse(coeffs)
    assert all_almost_equal(image.real, reco_image.real)
    assert all_almost_equal(image, reco_image)


if __name__ == '__main__':
    odl.util.test_file(__file__)
