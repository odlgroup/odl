# Copyright 2014-2017 The ODL contributors
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
import scipy.signal
import scipy.misc
import odl
import odl.contrib.fom
from odl.contrib.fom.util import filter_image_sep2d
from odl.util.testutils import simple_fixture, noise_element

fft_impl = simple_fixture('fft_impl',
                          [odl.util.testutils.never_skip('numpy'),
                           odl.util.testutils.skip_if_no_pyfftw('pyfftw')])

space = simple_fixture('space',
                       [odl.rn(3),
                        odl.rn(3, dtype='float32'),
                        odl.uniform_discr(0, 1, 10),
                        odl.uniform_discr([0, 0], [1, 1], [5, 5])])

scalar_fom = simple_fixture('FOM',
                            [odl.contrib.fom.mean_squared_error,
                             odl.contrib.fom.mean_absolute_error,
                             odl.contrib.fom.mean_value_difference,
                             odl.contrib.fom.standard_deviation_difference,
                             odl.contrib.fom.range_difference,
                             odl.contrib.fom.blurring,
                             odl.contrib.fom.false_structures])


def test_general(space, scalar_fom):
    """Test general properties of FOMs."""
    for name, ground_truth in space.examples:
        ground_truth = np.abs(ground_truth)
        scale = np.max(ground_truth) - np.min(ground_truth)
        noise = scale * np.abs(noise_element(space))
        data = ground_truth + noise

        # Check that range is a real number
        assert np.isscalar(scalar_fom(data, ground_truth))

        # Check that FOM is minimal when ground truth is comared with itself
        assert (scalar_fom(ground_truth, ground_truth) <=
                scalar_fom(data, ground_truth))

        # Check that FOM is monotonic wrt noise level
        assert (scalar_fom(ground_truth + noise, ground_truth) <=
                scalar_fom(ground_truth + 2 * noise, ground_truth))


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
    """Test the ``psnr`` fom."""
    true = odl.phantom.white_noise(space)
    data = odl.phantom.white_noise(space)

    # Compute the true value
    mse = np.mean((true - data) ** 2)
    maxi = np.max(np.abs(true))
    expected = 10 * np.log10(maxi ** 2 / mse)

    # Test regular call
    result = odl.contrib.fom.psnr(data, true)
    assert result == pytest.approx(expected)

    # Test with force_lower_is_better giving negative of expected
    result = odl.contrib.fom.psnr(data, true,
                                  force_lower_is_better=True)
    assert result == pytest.approx(-expected)

    # Test with Z-score that result is independent of affine transformation
    result = odl.contrib.fom.psnr(data * 3.7 + 1.234, true,
                                  use_zscore=True)
    expected = odl.contrib.fom.psnr(data, true,
                                    use_zscore=True)
    assert result == pytest.approx(expected)



def test_mean_value_difference_sign():
    space = odl.uniform_discr(0, 1, 10)
    I0 = space.one()
    I1 = -I0.copy()
    assert np.abs(odl.contrib.fom.mean_value_difference(I0, I1)) > 0.1
    assert np.abs(odl.contrib.fom.mean_value_difference(I0, I1,
                  normalized=True)) > 0.1


def test_mean_value_difference_range_value(space):
    I0 = space.element(np.random.normal(0, 1, size=space.shape))
    I1 = space.element(np.random.normal(0, 1, size=space.shape))

    max0 = np.max(I0.asarray())
    max1 = np.max(I1.asarray())
    min0 = np.min(I0.asarray())
    min1 = np.min(I1.asarray())


    assert odl.contrib.fom.mean_value_difference(I0, I1) <= max(max0 - min1,
                                                max1 - min0)
    assert pytest.approx(odl.contrib.fom.mean_value_difference(I0, I0)) == 0
    assert odl.contrib.fom.mean_value_difference(10*I0, I0,
                                                 normalized=True) <= 1.0


def test_standard_deviation_difference_range_value(space):
    I0 = space.element(np.random.normal(0, 1, size=space.shape))
    const = np.random.normal(0, 10)

    assert pytest.approx(odl.contrib.fom.standard_deviation_difference(
            I0, I0)) == 0
    assert odl.contrib.fom.standard_deviation_difference(10*I0, I0,
            normalized=True) <= 1.0
    assert pytest.approx(odl.contrib.fom.standard_deviation_difference(
            I0, I0 + const)) == 0
    test_value = odl.contrib.fom.standard_deviation_difference(
            space.one(), space.zero(), normalized=True)
    assert pytest.approx(test_value) == 0

def test_range_difference(space):
    I0 = space.element(np.random.normal(0, 1, size=space.shape))
    I1 = space.element(np.random.normal(0, 1, size=space.shape))
    const = np.random.normal(0, 10)

    assert pytest.approx(odl.contrib.fom.range_difference(
            I0, I0)) == 0
    assert pytest.approx(odl.contrib.fom.range_difference(
            I0 + const, I0)) == 0
    aconst=np.abs(const)
    eval0=aconst * odl.contrib.fom.range_difference(I0, I1)
    eval1=odl.contrib.fom.range_difference(aconst*I0, aconst*I1)
    assert pytest.approx(eval0) == pytest.approx(eval1)




if __name__ == '__main__':
    odl.util.test_file(__file__)
