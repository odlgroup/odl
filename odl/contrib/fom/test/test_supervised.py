import numpy as np
import pytest
import scipy.signal
import scipy.misc
import odl
from odl.contrib.fom.util import filter_image_sep2d
from odl.util.testutils import simple_fixture

fft_impl_params = [odl.util.testutils.never_skip('numpy'),
                   odl.util.testutils.skip_if_no_pyfftw('pyfftw')]
fft_impl = simple_fixture('fft_impl', fft_impl_params,
                          fmt=" {name} = '{value.args[1]}' ")


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


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
