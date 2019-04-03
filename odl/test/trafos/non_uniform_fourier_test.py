# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

import numpy as np

import odl
from odl.trafos.non_uniform_fourier import (
    NonUniformFourierTransform, NonUniformFourierTransformAdjoint)

# ---- Non-uniform Fourier Transform ---- #
def test_non_uniform_fourier_trafo_2d_call():
    im_size = 512
    shape = [im_size] * 2
    samples = np.array(
        np.where(np.random.normal(size=shape) >= 0),
        dtype=float,
    ).T
    image = np.random.normal(size=shape)
    nfft = NonUniformFourierTransform(shape=shape, samples=samples)
    nfft(image)


def test_non_uniform_fourier_adj_trafo_2d_call():
    im_size = 512
    shape = [im_size] * 2
    samples = np.array(
        np.where(np.random.normal(size=shape) >= 0),
        dtype=float,
    ).T
    nfft_coeffs = np.random.normal(size=(len(samples),))
    nfft_adj = NonUniformFourierTransformAdjoint(
        shape=shape,
        samples=samples,
    )
    nfft_adj(nfft_coeffs)


def test_non_uniform_fourier_trafo_1d_call():
    sig_size = 512
    shape = [sig_size]
    samples = np.array(
        np.where(np.random.normal(size=shape) >= 0),
        dtype=float,
    ).T
    image = np.random.normal(size=shape)
    nfft = NonUniformFourierTransform(shape=shape, samples=samples)
    nfft(image)


def test_non_uniform_fourier_adj_trafo_1d_call():
    sig_size = 512
    shape = [sig_size]
    samples = np.array(
        np.where(np.random.normal(size=shape) >= 0),
        dtype=float,
    ).T
    nfft_coeffs = np.random.normal(size=(len(samples),))
    nfft_adj = NonUniformFourierTransformAdjoint(
        shape=shape,
        samples=samples,
    )
    nfft_adj(nfft_coeffs)


def test_non_uniform_fourier_trafo_1d_res():
    sig_size = 512
    shape = [sig_size]
    samples = np.arange(sig_size)[:, None].astype(float)
    sig = np.random.normal(size=shape)
    nfft = NonUniformFourierTransform(
        shape=shape,
        samples=samples,
        max_frequencies=sig_size,
    )
    res_nfft = nfft(sig)
    res_np_fft = np.fft.fftshift(
        np.fft.fft(np.fft.fftshift(sig), norm="ortho"),
    )
    assert np.allclose(res_nfft, res_np_fft)


def test_non_uniform_fourier_trafo_2d_res():
    im_size = 512
    shape = [im_size] * 2
    coords = [np.arange(im_size)[:, None].astype(float)] * 2
    samples = np.hstack((np.meshgrid(*coords))).swapaxes(0,1).reshape(2,-1).T
    image = np.random.normal(size=shape)
    nfft = NonUniformFourierTransform(
        shape=shape,
        samples=samples,
        max_frequencies=im_size,
    )
    res_nfft = nfft(image)
    res_np_fft = np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(image), norm="ortho"),
    )
    assert np.allclose(res_nfft, res_np_fft.flatten())


if __name__ == '__main__':
    odl.util.test_file(__file__)
