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

import odl
from odl.util.testutils import (all_almost_equal, noise_element,
                                skip_if_no_pywavelets, simple_fixture)


# --- pytest fixtures --- #
wavelet = simple_fixture('wavelet', ['db1', 'sym2'])
pad_mode = simple_fixture('pad_mode', ['constant', 'pywt_periodic'])
ndim = simple_fixture('ndim', [1, 2, 3])
nlevels = simple_fixture('nlevels', [2, None])
wave_impl = simple_fixture('wave_impl', [skip_if_no_pywavelets('pywt')])


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


def test_wavelet_transform(wave_impl, shape_setup, floating_dtype):
    # Verify that the operator works as expected
    wavelet, pad_mode, nlevels, shape, _ = shape_setup
    ndim = len(shape)

    space = odl.uniform_discr([-1] * ndim, [1] * ndim, shape,
                              dtype=floating_dtype)
    image = noise_element(space)

    # TODO: check more error scenarios
    if wave_impl == 'pywt' and pad_mode == 'constant':
        with pytest.raises(ValueError):
            wave_trafo = odl.trafos.WaveletTransform(
                space, wavelet, nlevels, pad_mode, pad_const=1, impl=wave_impl)

    wave_trafo = odl.trafos.WaveletTransform(
        space, wavelet, nlevels, pad_mode, impl=wave_impl)

    assert wave_trafo.domain.dtype == floating_dtype
    assert wave_trafo.range.dtype == floating_dtype

    wave_trafo_inv = wave_trafo.inverse
    assert wave_trafo_inv.domain.dtype == floating_dtype
    assert wave_trafo_inv.range.dtype == floating_dtype
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
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
