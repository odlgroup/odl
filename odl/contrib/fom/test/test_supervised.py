# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for supervised FoMs."""

from __future__ import division

import numpy as np
import pytest
import scipy.misc
import scipy.signal

import odl
import odl.contrib.fom
from odl.contrib.fom.util import filter_image_sep2d
from odl.util.testutils import simple_fixture

fft_impl = simple_fixture('fft_impl',
                          [odl.util.testutils.never_skip('numpy'),
                           odl.util.testutils.skip_if_no_pyfftw('pyfftw')])
space = simple_fixture('space',
                       [odl.rn(3),
                        odl.uniform_discr(0, 1, 10)])


def filter_image(image, fh, fv):
    """Reference filtering function using ``scipy.signal.convolve``."""
    fh, fv = np.asarray(fh), np.asarray(fv)
    image = scipy.signal.convolve(image, fh[:, None], mode='same')
    return scipy.signal.convolve(image, fv[None, :], mode='same')


def test_filter_image_fft(fft_impl):
    """Test the FFT filtering function against the real-space variant."""
    image = np.random.rand(128, 128)
    image -= np.mean(image)

    for fh, fv in zip([[1, 1], [1, 1, 1], [1, 1, 1, 1]],
                      [[-1, 1], [-1, 1, 1], [-1, -1, 1, 1]]):

        conv_real = filter_image(image, fh, fv)
        conv_fft = filter_image_sep2d(image, fh, fv, impl=fft_impl)
        assert np.allclose(conv_real, conv_fft)


def test_mean_squared_error(space):
    true = odl.phantom.white_noise(space)
    data = odl.phantom.white_noise(space)

    result = odl.contrib.fom.mean_squared_error(data, true)
    expected = np.mean((true - data) ** 2)

    assert result == pytest.approx(expected)


def test_mean_absolute_error(space):
    true = odl.phantom.white_noise(space)
    data = odl.phantom.white_noise(space)

    result = odl.contrib.fom.mean_absolute_error(data, true)
    expected = np.mean(np.abs(true - data))

    assert result == pytest.approx(expected)


def test_psnr(space):
    true = odl.phantom.white_noise(space)
    data = odl.phantom.white_noise(space)

    result = odl.contrib.fom.psnr(data, true)

    mse = np.mean((true - data) ** 2)
    maxi = np.max(np.abs(true))
    expected = 10 * np.log10(maxi ** 2 / mse)

    assert result == pytest.approx(expected)


if __name__ == '__main__':
    odl.util.test_file(__file__)
