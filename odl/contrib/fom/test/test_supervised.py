import numpy as np
import pytest
import scipy.signal
import scipy.misc
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
    expected = 10 * np.log10(maxi**2 / mse)

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
    const=np.random.normal(0, 10)

    assert pytest.approx(odl.contrib.fom.standard_deviation_difference(I0, I0)) == 0
    assert odl.contrib.fom.standard_deviation_difference(10*I0, I0,
                                                 normalized=True) <= 1.0
    assert pytest.approx(odl.contrib.fom.standard_deviation_difference(I0, I0+ const)) == 0
    test_value = odl.contrib.fom.standard_deviation_difference(
            space.one(),space.zero(), normalized=True)
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
