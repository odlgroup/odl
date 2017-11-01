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
                             odl.contrib.fom.blurring])


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
    zero = space.zero()

    # Test psnr of image with itself is infinity
    assert odl.contrib.fom.psnr(true, true) == np.inf

    # Test psnr with both constants is infinity
    assert odl.contrib.fom.psnr(zero, zero) == np.inf

    # Test psnr with ground truth constant is negative infinity
    assert odl.contrib.fom.psnr(data, zero) == -np.inf

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


def test_ssim(space):
    ground_truth = odl.phantom.white_noise(space)

    # SSIM of true image should be either
    # * 1 with force_lower_is_better == False,
    # * -1 with force_lower_is_better == True and normalized == False,
    # * 0 with force_lower_is_better == True and normalized == True.
    result = odl.contrib.fom.ssim(ground_truth, ground_truth)
    result_normalized = odl.contrib.fom.ssim(ground_truth, ground_truth,
                                             normalized=True)
    result_flib = odl.contrib.fom.ssim(ground_truth, ground_truth,
                                       force_lower_is_better=True)
    result_nf = odl.contrib.fom.ssim(ground_truth, ground_truth,
                                     normalized=True,
                                     force_lower_is_better=True)
    assert pytest.approx(result) == 1
    assert pytest.approx(result_normalized) == 1
    assert pytest.approx(result_flib) == -1
    assert pytest.approx(result_nf) == 0

    # SSIM with ground truth zero should always give zero if not normalized
    # and 1/2 otherwise.
    data = odl.phantom.white_noise(space)
    result = odl.contrib.fom.ssim(data, space.zero())
    result_normalized = odl.contrib.fom.ssim(data, space.zero(),
                                             normalized=True)
    result_flib = odl.contrib.fom.ssim(data, space.zero(),
                                       force_lower_is_better=True)
    result_nf = odl.contrib.fom.ssim(data, space.zero(),
                                     normalized=True,
                                     force_lower_is_better=True)
    assert pytest.approx(result) == 0
    assert pytest.approx(result_normalized) == 0.5
    assert pytest.approx(result_flib) == 0
    assert pytest.approx(result_nf) == 0.5

    # SSIM should be symmetric if the dynamic range is set explicitly.
    result1 = odl.contrib.fom.ssim(data, ground_truth, dynamic_range=1)
    result2 = odl.contrib.fom.ssim(ground_truth, data, dynamic_range=1)
    assert pytest.approx(result1) == result2


if __name__ == '__main__':
    odl.util.test_file(__file__)
