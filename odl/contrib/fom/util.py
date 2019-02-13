# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for FOMs."""

import numpy as np

from odl.discr import uniform_discr
from odl.trafos.backends import PYFFTW_AVAILABLE

__all__ = ()


def filter_image_sep2d(image, fh, fv, impl='numpy', padding=None):
    """Filter an image with a separable filter.

    Parameters
    ----------
    image : 2D array-like
        The image to be filtered. It must have a real (vs. complex) dtype.
    fh, fv : 1D array-like
        Horizontal (axis 0) and vertical (axis 1) filters. Their sizes
        can be at most the image sizes in the respective axes.
    impl : {'numpy', 'pyfftw'}, optional
        FFT backend to use. The ``pyfftw`` backend requires the
        ``pyfftw`` package to be installed. It is usually significantly
        faster than the NumPy backend.
    padding : positive int, optional
        Amount of zeros added to the left and right of the image in all
        axes before FFT. This helps avoiding wraparound artifacts due to
        large boundary values.
        For ``None``, the padding is computed as ::

            padding = min(max(len(fh), len(fv)) - 1, 64)

        A padding of ``len(filt) - 1`` ensures that errors in FFT-based
        convolutions are small. At the same time, the padding should not
        be excessive to retain efficiency.

    Returns
    -------
    filtered : 2D `numpy.ndarray`
        The image filtered horizontally by ``fh`` and vertically by ``fv``.
        It has the same shape as ``image``, and its dtype is
        ``np.result_type(image, fh, fv)``.
    """
    # TODO: generalize for nD
    impl, impl_in = str(impl).lower(), impl
    if impl not in ('numpy', 'pyfftw'):
        raise ValueError('`impl` {!r} not understood'
                         ''.format(impl_in))

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError('`image` must be 2-dimensional, got image with '
                         'ndim={}'.format(image.ndim))
    if image.size == 0:
        raise ValueError('`image` cannot have size 0')
    if not np.issubsctype(image.dtype, np.floating):
        image = image.astype(float)

    fh = np.asarray(fh).astype(image.dtype)
    if fh.ndim != 1:
        raise ValueError('`fh` must be one-dimensional')
    elif fh.size == 0:
        raise ValueError('`fh` cannot have size 0')
    elif fh.size > image.shape[0]:
        raise ValueError('`fh` can be at most `image.shape[0]`, got '
                         '{} > {}'.format(fh.size, image.shape[0]))

    fv = np.asarray(fv).astype(image.dtype)
    if fv.ndim != 1:
        raise ValueError('`fv` must be one-dimensional')
    elif fv.size == 0:
        raise ValueError('`fv` cannot have size 0')
    elif fv.size > image.shape[0]:
        raise ValueError('`fv` can be at most `image.shape[1]`, got '
                         '{} > {}'.format(fv.size, image.shape[1]))

    # Pad image with zeros
    if padding is None:
        padding = min(max(len(fh), len(fv)) - 1, 64)

    if padding != 0:
        image_padded = np.pad(image, padding, mode='constant')
    else:
        image_padded = image.copy() if impl == 'pyfftw' else image

    # Prepare filters for the convolution
    def prepare_for_fft(filt, n_new):
        """Return padded and shifted filter ready for FFT.

        The filter is padded with zeros to the new size, and then shifted
        such that such that the middle element of old filter, i.e., the
        one at index ``(len(filt) - 1) // 2`` ends up at index 0.
        """
        mid = (len(filt) - 1) // 2
        padded = np.zeros(n_new, dtype=filt.dtype)
        padded[:len(filt) - mid] = filt[mid:]
        padded[len(padded) - mid:] = filt[:mid]
        return padded

    fh = prepare_for_fft(fh, image_padded.shape[0])
    fv = prepare_for_fft(fv, image_padded.shape[1])

    # Perform the multiplication in Fourier space and apply inverse FFT
    if impl == 'numpy':
        image_ft = np.fft.rfftn(image_padded)
        fh_ft = np.fft.fft(fh)
        fv_ft = np.fft.rfft(fv)

        image_ft *= fh_ft[:, None]
        image_ft *= fv_ft[None, :]
        # Important to specify the shape since `irfftn` cannot know the
        # original shape
        conv = np.fft.irfftn(image_ft, s=image_padded.shape)
        if conv.dtype != image.dtype:
            conv = conv.astype(image.dtype)

    elif impl == 'pyfftw':
        if not PYFFTW_AVAILABLE:
            raise ValueError(
                '`pyfftw` package is not available; you need to install it '
                'to use the pyfftw backend')

        import pyfftw
        import multiprocessing

        # Generate output arrays, for half-complex transform of image and
        # vertical filter, and full FT of the horizontal filter
        out_img_shape = (image_padded.shape[0], image_padded.shape[1] // 2 + 1)
        out_img_dtype = np.result_type(image_padded, 1j)
        out_img = np.empty(out_img_shape, out_img_dtype)

        out_fh_shape = out_img_shape[0]
        out_fh_dtype = np.result_type(fh, 1j)
        fh_c = fh.astype(out_fh_dtype)  # need to make this a C2C trafo
        out_fh = np.empty(out_fh_shape, out_fh_dtype)

        out_fv_shape = out_img_shape[1]
        out_fv_dtype = np.result_type(fv, 1j)
        out_fv = np.empty(out_fv_shape, out_fv_dtype)

        # Perform the forward transforms of image and filters. We use
        # the `FFTW_ESTIMATE` flag to not allow the planner to destroy
        # the input.
        plan = pyfftw.FFTW(image_padded, out_img, axes=(0, 1),
                           direction='FFTW_FORWARD',
                           flags=['FFTW_ESTIMATE'],
                           threads=multiprocessing.cpu_count())
        plan(image_padded, out_img)

        plan = pyfftw.FFTW(fh_c, out_fh, axes=(0,),
                           direction='FFTW_FORWARD',
                           flags=['FFTW_ESTIMATE'],
                           threads=multiprocessing.cpu_count())
        plan(fh_c, out_fh)

        plan = pyfftw.FFTW(fv, out_fv, axes=(0,),
                           direction='FFTW_FORWARD',
                           flags=['FFTW_ESTIMATE'],
                           threads=multiprocessing.cpu_count())
        plan(fv, out_fv)

        # Fourier space multiplication
        out_img *= out_fh[:, None]
        out_img *= out_fv[None, :]

        # Inverse trafo
        conv = image_padded  # Overwrite
        plan = pyfftw.FFTW(out_img.copy(), conv, axes=(0, 1),
                           direction='FFTW_BACKWARD',
                           flags=['FFTW_ESTIMATE'],
                           threads=multiprocessing.cpu_count())
        plan(out_img, conv)

    else:
        raise ValueError('unsupported `impl` {!r}'.format(impl_in))

    if padding:
        return conv[padding:-padding, padding:-padding]
    else:
        return conv


def haarpsi_similarity_map(img1, img2, axis, c, a):
    r"""Local similarity map for directional features along an axis.

    Parameters
    ----------
    img1, img2 : array-like
        The images to compare. They must have equal shape.
    axis : {0, 1}
        Direction in which to look for edge similarities.
    c : positive float
        Constant determining the score of maximally dissimilar values.
        Smaller constant means higher penalty for dissimilarity.
        See Notes for details.
    a : positive float
        Parameter in the logistic function. Larger value leads to a
        steeper curve, thus lowering the threshold for an input to
        be mapped to an output close to 1. See Notes for details.

    Returns
    -------
    local_sim : `numpy.ndarray`
        Pointwise similarity of directional edge features of ``img1`` and
        ``img2``, measured using two Haar wavelet detail levels.

    Notes
    -----
    For input images :math:`f_1, f_2` this function is defined as

    .. math::
        \mathrm{HS}_{f_1, f_2}^{(k)}(x) =
        l_a \left(
        \frac{1}{2} \sum_{j=1}^2
        S\left(\left|g_j^{(k)} \ast f_1 \right|(x),
        \left|g_j^{(k)} \ast f_2 \right|(x), c\right)
        \right),

    see `[Rei+2016] <https://arxiv.org/abs/1607.06140>`_ equation (10).
    Here, the superscript :math:`(k)` refers to the axis (0 or 1)
    in which edge features are compared, :math:`l_a` is the logistic
    function :math:`l_a(x) = (1 + \mathrm{e}^{-a x})^{-1}`, and :math:`S`
    is the pointwise similarity score

    .. math::
        S(x, y, c) = \frac{2xy + c^2}{x^2 + y^2 + c^2},

    Hence, :math:`c` is the :math:`y`-value at which the score
    drops to :math:`1 / 2` for :math:`x = 0`. In other words, the smaller
    :math:`c` is chosen, the more dissimilarity is penalized.

    The filters :math:`g_j^{(k)}` are high-pass Haar wavelet filters in the
    axis :math:`k` and low-pass Haar wavelet filters in the other axes.
    The index :math:`j` refers to the scaling level of the wavelet.
    In code, these filters can be computed as ::

        f_lo_level1 = [np.sqrt(2), np.sqrt(2)] # low-pass Haar filter
        f_hi_level1 = [-np.sqrt(2), np.sqrt(2)] # high-pass Haar filter
        f_lo_level2 = np.repeat(f_lo_level1, 2)
        f_hi_level2 = np.repeat(f_hi_level1, 2)
        f_lo_level3 = np.repeat(f_lo_level2, 2)
        f_hi_level3 = np.repeat(f_hi_level2, 2)
        ...

    The logistic function :math:`l_a` transforms values in
    :math:`[0, \infty)` to :math:`[1/2, 1)`, where the parameter
    :math:`a` determines how fast the curve attains values close
    to 1. Larger :math:`a` means that smaller :math:`x` will yield
    a value :math:`l_a(x)` close to 1 (and thus result in a higher
    score). In other words, the larger :math:`a`, the more forgiving
    the similarity measure.

    References
    ----------
    [Rei+2016] Reisenhofer, R, Bosse, S, Kutyniok, G, and Wiegand, T.
    *A Haar Wavelet-Based Perceptual Similarity Index for Image Quality
    Assessment*. arXiv:1607.06140 [cs], Jul. 2016.
    """
    # TODO: generalize for nD
    import scipy.special
    impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'

    # Haar wavelet filters for levels 1 and 2
    dec_lo_lvl1 = np.array([np.sqrt(2), np.sqrt(2)])
    dec_lo_lvl2 = np.repeat(dec_lo_lvl1, 2)
    dec_hi_lvl1 = np.array([-np.sqrt(2), np.sqrt(2)])
    dec_hi_lvl2 = np.repeat(dec_hi_lvl1, 2)

    if axis == 0:
        # High-pass in axis 0, low-pass in axis 1
        fh_lvl1 = dec_hi_lvl1
        fv_lvl1 = dec_lo_lvl1
        fh_lvl2 = dec_hi_lvl2
        fv_lvl2 = dec_lo_lvl2
    elif axis == 1:
        # Low-pass in axis 0, high-pass in axis 1
        fh_lvl1 = dec_lo_lvl1
        fv_lvl1 = dec_hi_lvl1
        fh_lvl2 = dec_lo_lvl2
        fv_lvl2 = dec_hi_lvl2
    else:
        raise ValueError('`axis` out of the valid range 0 -> 1')

    # Filter images with level 1 and 2 filters
    img1_lvl1 = filter_image_sep2d(img1, fh_lvl1, fv_lvl1, impl=impl)
    img1_lvl2 = filter_image_sep2d(img1, fh_lvl2, fv_lvl2, impl=impl)

    img2_lvl1 = filter_image_sep2d(img2, fh_lvl1, fv_lvl1, impl=impl)
    img2_lvl2 = filter_image_sep2d(img2, fh_lvl2, fv_lvl2, impl=impl)

    c = float(c)

    def S(x, y):
        """Return ``(2 * x * y + c ** 2) / (x ** 2 + y ** 2 + c ** 2)``."""
        num = 2 * x
        num *= y
        num += c ** 2
        denom = x ** 2
        denom += y ** 2
        denom += c ** 2
        frac = num
        frac /= denom
        return frac

    # Compute similarity scores for both levels
    np.abs(img1_lvl1, out=img1_lvl1)
    np.abs(img2_lvl1, out=img2_lvl1)
    np.abs(img1_lvl2, out=img1_lvl2)
    np.abs(img2_lvl2, out=img2_lvl2)

    sim_lvl1 = S(img1_lvl1, img2_lvl1)
    sim_lvl2 = S(img1_lvl2, img2_lvl2)

    # Return logistic of the mean value
    sim = sim_lvl1
    sim += sim_lvl2
    sim /= 2
    sim *= a
    return scipy.special.expit(sim)


def haarpsi_weight_map(img1, img2, axis):
    r"""Weighting map for directional features along an axis.

    Parameters
    ----------
    img1, img2 : array-like
        The images to compare. They must have equal shape.
    axis : {0, 1}
        Direction in which to look for edge similarities.

    Returns
    -------
    weight_map : `numpy.ndarray`
        The pointwise weight map. See Notes for details.

    Notes
    -----
    The pointwise weight map of associated with input images :math:`f_1, f_2`
    and axis :math:`k` is defined
    as

    .. math::
        \mathrm{W}_{f_1, f_2}^{(k)}(x) =
        \max \left\{
            \left|g_3^{(k)} \ast f_1 \right|(x),
            \left|g_3^{(k)} \ast f_2 \right|(x)
        \right\},

    see `[Rei+2016] <https://arxiv.org/abs/1607.06140>`_ equations (11)
    and (13).

    Here, :math:`g_3^{(k)}` is a Haar wavelet filter for scaling level 3
    that performs high-pass filtering in axis :math:`k` and low-pass
    filtering in the other axes. Such a filter can be computed as ::

        f_lo_level1 = [np.sqrt(2), np.sqrt(2)] # low-pass Haar filter
        f_hi_level1 = [-np.sqrt(2), np.sqrt(2)] # high-pass Haar filter
        f_lo_level3 = np.repeat(f_lo_level1, 4)
        f_hi_level3 = np.repeat(f_hi_level1, 4)

    References
    ----------
    [Rei+2016] Reisenhofer, R, Bosse, S, Kutyniok, G, and Wiegand, T.
    *A Haar Wavelet-Based Perceptual Similarity Index for Image Quality
    Assessment*. arXiv:1607.06140 [cs], Jul. 2016.
    """
    # TODO: generalize for nD
    impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'

    # Haar wavelet filters for level 3
    dec_lo_lvl3 = np.repeat([np.sqrt(2), np.sqrt(2)], 4)
    dec_hi_lvl3 = np.repeat([-np.sqrt(2), np.sqrt(2)], 4)

    if axis == 0:
        fh_lvl3 = dec_hi_lvl3
        fv_lvl3 = dec_lo_lvl3
    elif axis == 1:
        fh_lvl3 = dec_lo_lvl3
        fv_lvl3 = dec_hi_lvl3
    else:
        raise ValueError('`axis` out of the valid range 0 -> 1')

    # Filter with level 3 wavelet filter
    img1_lvl3 = filter_image_sep2d(img1, fh_lvl3, fv_lvl3, impl=impl)
    img2_lvl3 = filter_image_sep2d(img2, fh_lvl3, fv_lvl3, impl=impl)

    # Return the pointwise maximum of the filtered images
    np.abs(img1_lvl3, out=img1_lvl3)
    np.abs(img2_lvl3, out=img2_lvl3)

    return np.maximum(img1_lvl3, img2_lvl3)


def spherical_sum(image, binning_factor=1.0):
    """Sum image values over concentric annuli.

    Parameters
    ----------
    image : `DiscreteLp` element
        Input data whose radial sum should be computed.
    binning_factor : positive float, optional
        Reduce the number of output bins by this factor. Increasing this
        number can help reducing fluctuations due to the variance of points
        that fall in a particular annulus.
        A binning factor of ``1`` corresponds to a bin size equal to
        image pixel size for images with square pixels, otherwise ::

            max(norm2(c)) / norm2(shape)

        where the maximum is taken over all corners of the image domain.

    Returns
    -------
    spherical_sum : 1D `DiscreteLp` element
        The spherical sum of ``image``. Its space is one-dimensional with
        domain ``[0, rmax]``, where ``rmax`` is the radius of the smallest
        ball containing ``image.space.domain``. Its shape is ``(N,)`` with ::

            N = int(sqrt(sum(n ** 2 for n in image.shape)) / binning_factor)
    """
    r = np.sqrt(sum(xi ** 2 for xi in image.space.meshgrid))
    rmax = max(np.linalg.norm(c) for c in image.space.domain.corners())
    n_bins = int(np.sqrt(sum(n ** 2 for n in image.shape)) / binning_factor)
    rad_sum, _ = np.histogram(r, weights=image, bins=n_bins, range=(0, rmax))

    out_spc = uniform_discr(min_pt=0, max_pt=rmax, shape=n_bins,
                            impl=image.space.impl, dtype=image.space.dtype,
                            interp="linear", axis_labels=["$r$"])

    return out_spc.element(rad_sum)
