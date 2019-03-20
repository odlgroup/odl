# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the figures of merit (FOMs) that use a known ground truth."""

from __future__ import division

import numpy as np
import pytest
import scipy.misc
import scipy.signal

import odl
from odl.contrib import fom
from odl.util.testutils import noise_element, simple_fixture, skip_if_no_pyfftw


# --- pytest fixtures --- #


fft_impl = simple_fixture(
    'fft_impl',
    [pytest.param('numpy'), pytest.param('pyfftw', marks=skip_if_no_pyfftw)]
)

space = simple_fixture(
    'space',
    [odl.rn(3),
     odl.rn(3, dtype='float32'),
     odl.uniform_discr(0, 1, 10),
     odl.uniform_discr([0, 0], [1, 1], [5, 5])]
)

scalar_fom = simple_fixture(
    'scalar_fom',
    [fom.mean_squared_error,
     fom.mean_absolute_error,
     fom.mean_value_difference,
     fom.standard_deviation_difference,
     fom.range_difference,
     fom.blurring]
)


# --- Tests --- #


def test_general(space, scalar_fom):
    """Test general properties of FOMs."""
    for name, ground_truth in space.examples:
        ground_truth = np.abs(ground_truth)
        scale = np.max(ground_truth) - np.min(ground_truth)
        noise = scale * np.abs(noise_element(space))
        data = ground_truth + noise

        # Check that range is a real number
        assert np.isscalar(scalar_fom(data, ground_truth))

        # Check that FOM is minimal when ground truth is compared with itself
        assert (
            scalar_fom(ground_truth, ground_truth)
            <= scalar_fom(data, ground_truth)
        )

        # Check that FOM is monotonic wrt noise level
        # This does not work for the FOMS `standard_deviation_difference`
        # and `range_difference`.
        if scalar_fom not in [fom.standard_deviation_difference,
                              fom.range_difference]:
            assert (
                scalar_fom(ground_truth + noise, ground_truth)
                <= scalar_fom(ground_truth + 2 * noise, ground_truth)
            )

        # Check that supplying arrays works as well
        assert (
            scalar_fom(ground_truth.asarray(), ground_truth.asarray())
            <= scalar_fom(data.asarray(), ground_truth.asarray())
        )


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
        conv_fft = fom.util.filter_image_sep2d(image, fh, fv, impl=fft_impl)
        assert np.allclose(conv_real, conv_fft)


def test_mean_squared_error(space):
    true = odl.phantom.white_noise(space)
    data = odl.phantom.white_noise(space)

    result = fom.mean_squared_error(data, true)
    expected = np.mean((true - data) ** 2)

    assert result == pytest.approx(expected)


def test_mean_absolute_error(space):
    true = odl.phantom.white_noise(space)
    data = odl.phantom.white_noise(space)

    result = fom.mean_absolute_error(data, true)
    expected = np.mean(np.abs(true - data))

    assert result == pytest.approx(expected)


def test_psnr(space):
    """Test the ``psnr`` fom."""
    true = odl.phantom.white_noise(space)
    data = odl.phantom.white_noise(space)
    zero = space.zero()

    # Check the corner cases
    assert fom.psnr(true, true) == np.inf
    assert fom.psnr(zero, zero) == np.inf
    assert fom.psnr(data, zero) == -np.inf

    # Compute the true value
    mse = np.mean((true - data) ** 2)
    maxi = np.max(np.abs(true))
    expected = 10 * np.log10(maxi ** 2 / mse)

    # Test regular call
    result = fom.psnr(data, true)
    assert result == pytest.approx(expected, abs=1e-6)

    # Test with arrays as input
    result = fom.psnr(data.asarray(), true.asarray())
    assert result == pytest.approx(expected, abs=1e-6)

    # Test with force_lower_is_better giving negative of expected
    result = fom.psnr(data, true, force_lower_is_better=True)
    assert result == pytest.approx(-expected, abs=1e-6)

    # Test with Z-score that result is independent of affine transformation
    result = fom.psnr(data * 3.7 + 1.234, true, use_zscore=True)
    expected = fom.psnr(data, true, use_zscore=True)
    assert result == pytest.approx(expected, abs=1e-5)


def test_ssim(space):
    ground_truth = odl.phantom.white_noise(space)

    # SSIM of true image should be either
    # * 1 with force_lower_is_better == False,
    # * -1 with force_lower_is_better == True and normalized == False,
    # * 0 with force_lower_is_better == True and normalized == True.
    result = fom.ssim(ground_truth, ground_truth)
    assert result == pytest.approx(1)

    result_normalized = fom.ssim(ground_truth, ground_truth, normalized=True)
    assert result_normalized == pytest.approx(1)

    result_flib = fom.ssim(ground_truth, ground_truth,
                           force_lower_is_better=True)
    assert result_flib == pytest.approx(-1)

    result_nf = fom.ssim(ground_truth, ground_truth, normalized=True,
                         force_lower_is_better=True)
    assert result_nf == pytest.approx(0)

    # SSIM with ground truth zero should always give zero if not normalized
    # and 1/2 otherwise.
    data = odl.phantom.white_noise(space)

    result = fom.ssim(data, space.zero())
    assert result == pytest.approx(0)

    result_normalized = fom.ssim(data, space.zero(), normalized=True)
    assert result_normalized == pytest.approx(0.5)

    result_flib = fom.ssim(data, space.zero(), force_lower_is_better=True)
    assert result_flib == pytest.approx(0)

    result_nf = fom.ssim(data, space.zero(), normalized=True,
                         force_lower_is_better=True)
    assert result_nf == pytest.approx(0.5)

    # SSIM should be symmetric if the dynamic range is set explicitly.
    for nor in [True, False]:
        for flib in [True, False]:
            result1 = fom.ssim(data, ground_truth, dynamic_range=1,
                               normalized=nor, force_lower_is_better=flib)
            result2 = fom.ssim(ground_truth, data, dynamic_range=1,
                               normalized=nor, force_lower_is_better=flib)
            assert result1 == pytest.approx(result2)


def test_mean_value_difference_sign():
    space = odl.uniform_discr(0, 1, 10)
    I0 = space.one()
    assert np.abs(fom.mean_value_difference(I0, -I0)) == pytest.approx(2.0)


def test_mean_value_difference_range_value(space):
    I0 = odl.util.testutils.noise_element(space)
    I1 = odl.util.testutils.noise_element(space)
    max0 = np.max(I0)
    max1 = np.max(I1)
    min0 = np.min(I0)
    min1 = np.min(I1)

    assert fom.mean_value_difference(I0, I1) <= max(max0 - min1, max1 - min0)
    assert fom.mean_value_difference(I0, I0) == pytest.approx(0)
    assert fom.mean_value_difference(10 * I0, I0, normalized=True) <= 1.0


def test_standard_deviation_difference_range_value(space):
    I0 = odl.util.testutils.noise_element(space)
    value_shift = np.random.normal(0, 10)

    assert fom.standard_deviation_difference(I0, I0) == pytest.approx(0)
    assert (
        fom.standard_deviation_difference(10 * I0, I0, normalized=True)
        <= 1.0
    )
    assert (
        fom.standard_deviation_difference(I0, I0 + value_shift)
        == pytest.approx(0, abs=1e-5)
    )
    test_value = fom.standard_deviation_difference(space.one(), space.zero(),
                                                   normalized=True)
    assert test_value == pytest.approx(0, abs=1e-6)


def test_range_difference(space):
    I0 = space.element(np.random.normal(0, 1, size=space.shape))
    I1 = space.element(np.random.normal(0, 1, size=space.shape))
    const = np.random.normal(0, 10)

    assert fom.range_difference(I0, I0) == pytest.approx(0)
    assert fom.range_difference(I0 + const, I0) == pytest.approx(0, abs=1e-5)
    aconst = np.abs(const)
    eval0 = aconst * fom.range_difference(I0, I1)
    eval1 = fom.range_difference(aconst * I0, aconst * I1)
    assert eval0 == pytest.approx(eval1, abs=1e-5)


if __name__ == '__main__':
    odl.util.test_file(__file__)
