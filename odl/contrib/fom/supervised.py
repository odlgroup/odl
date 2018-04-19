# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Figures of merit (FOMs) for comparison against a known ground truth."""

from __future__ import division

import numpy as np

import odl

__all__ = ('mean_squared_error', 'mean_absolute_error',
           'mean_value_difference', 'standard_deviation_difference',
           'range_difference', 'blurring', 'false_structures', 'ssim', 'psnr',
           'haarpsi', 'noise_power_spectrum')


def mean_squared_error(data, ground_truth, mask=None, normalized=False):
    """Return mean squared L2 distance between ``data`` and ``ground_truth``.

    See also `this Wikipedia article
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_.

    Parameters
    ----------
    data : `Tensor`
        Input data to compare to the ground truth.
    ground_truth : `Tensor`
        Reference to compare ``data`` to.
    mask : `array-like`, optional
        If given, ``data * mask`` is compared to ``ground_truth * mask``.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    mse : float
        FOM value, where a lower value means a better match.

    Notes
    -----
    The FOM evaluates

    .. math::
        \mathrm{MSE}(f, g) = \\frac{\| f - g \|_2^2}{\| 1 \|_2^2},

    where :math:`\| 1 \|^2_2` is the volume of the domain of definition
    of the functions. For :math:`\mathbb{R}^n` type spaces, this is equal
    to the number of elements :math:`n`.

    The normalized form is

    .. math::
        \mathrm{MSE_N} = \\frac{\| f - g \|_2^2}{(\| f \|_2 + \| g \|_2)^2}.

    The normalized variant takes values in :math:`[0, 1]`.
    """
    l2norm = odl.solvers.L2Norm(data.space)
    l2norm_squared = odl.solvers.L2NormSquared(data.space)

    if mask is not None:
        data = data * mask
        ground_truth = ground_truth * mask

    diff = data - ground_truth
    fom = l2norm_squared(diff)

    if normalized:
        fom /= (l2norm(data) + l2norm(ground_truth)) ** 2
    else:
        fom /= l2norm_squared(data.space.one())

    return fom


def mean_absolute_error(data, ground_truth, mask=None, normalized=False):
    """Return L1-distance between ``data`` and ``ground_truth``.

    See also `this Wikipedia article
    <https://en.wikipedia.org/wiki/Mean_absolute_error>`_.

    Parameters
    ----------
    data : `Tensor`
        Input data to compare to the ground truth.
    ground_truth : `Tensor`
        Reference to compare ``data`` to.
    mask : `array-like`, optional
        If given, ``data * mask`` is compared to ``ground_truth * mask``.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    mae : float
        FOM value, where a lower value means a better match.

    Notes
    -----
    The FOM evaluates

    .. math::
        \mathrm{MAE}(f, g) = \\frac{\| f - g \|_1}{\| 1 \|_1},

    where :math:`\| 1 \|_1` is the volume of the domain of definition
    of the functions. For :math:`\mathbb{R}^n` type spaces, this is equal
    to the number of elements :math:`n`.

    The normalized form is

    .. math::
        \mathrm{MAE_N}(f, g) = \\frac{\| f - g \|_1}{\| f \|_1 + \| g \|_1}.

    The normalized variant takes values in :math:`[0, 1]`.
    """
    l1_norm = odl.solvers.L1Norm(data.space)
    if mask is not None:
        data = data * mask
        ground_truth = ground_truth * mask
    diff = data - ground_truth
    fom = l1_norm(diff)

    if normalized:
        fom /= (l1_norm(data) + l1_norm(ground_truth))
    else:
        fom /= l1_norm(data.space.one())

    return fom


def mean_value_difference(data, ground_truth, mask=None, normalized=False):
    """Return difference in mean value between ``data`` and ``ground_truth``.

    Parameters
    ----------
    data : `Tensor`
        Input data to compare to the ground truth.
    ground_truth : `Tensor`
        Reference to compare ``data`` to.
    mask : `array-like`, optional
        If given, ``data * mask`` is compared to ``ground_truth * mask``.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    mvd : float
        FOM value, where a lower value means a better match.

    Notes
    -----
    The FOM evaluates

    .. math::
         \mathrm{MVD}(f, g) =
         \\Big| |\\overline{f}| - |\\overline{g}| \\Big|,

    or, in normalized form

    .. math::
         \mathrm{MVD_N}(f, g) =
         \\frac{\\Big| |\\overline{f}| - |\\overline{g}| \\Big|}
               {\\Big| |\\overline{f}| + |\\overline{g}| \\Big|}

    where :math:`\\overline{f}` is the mean value of :math:`f`,

    .. math::
        \\overline{f} = \\frac{\\langle f, 1\\rangle}{\|1|_1}.

    The normalized variant takes values in :math:`[0, 1]`.
    """
    l1_norm = odl.solvers.L1Norm(data.space)
    if mask is not None:
        data = data * mask
        ground_truth = ground_truth * mask

    # Volume of space
    vol = l1_norm(data.space.one())

    data_mean = data.inner(data.space.one()) / vol
    ground_truth_mean = ground_truth.inner(ground_truth.space.one()) / vol

    fom = np.abs(np.abs(data_mean) - np.abs(ground_truth_mean))

    if normalized:
        fom /= (np.abs(data_mean) + np.abs(ground_truth_mean))

    return fom


def standard_deviation_difference(data, ground_truth, mask=None,
                                  normalized=False):
    """Return absolute difference in std between ``data`` and ``ground_truth``.

    Parameters
    ----------
    data : `Tensor`
        Input data to compare to the ground truth.
    ground_truth : `Tensor`
        Reference to compare ``data`` to.
    mask : `array-like`, optional
        If given, ``data * mask`` is compared to ``ground_truth * mask``.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    sdd : float
        FOM value, where a lower value means a better match.

    Notes
    -----
    The FOM evaluates

    .. math::
        \mathrm{SDD}(f, g) =
         \Big| \| f - \\overline{f} \|_2 - \| g - \\overline{g} \|_2 \Big|,

    or, in normalized form

    .. math::
        \mathrm{SDD_N}(f, g) =
         \\frac{\Big| \| f - \\overline{f} \|_2 -
                      \| g - \\overline{g} \|_2 \Big|}
               {\Big| \| f - \\overline{f} \|_2 +
                      \| g - \\overline{g} \|_2 \Big|},

    where :math:`\\overline{f}` is the mean value of :math:`f`,

    .. math::
        \\overline{f} = \\frac{\\langle f, 1\\rangle}{\|1|_1}.

    The normalized variant takes values in :math:`[0, 1]`.
    """
    l1_norm = odl.solvers.L1Norm(data.space)
    l2_norm = odl.solvers.L2Norm(data.space)

    if mask is not None:
        data = data * mask
        ground_truth = ground_truth * mask

    # Volume of space
    vol = l1_norm(data.space.one())

    data_mean = data.inner(data.space.one()) / vol
    ground_truth_mean = ground_truth.inner(ground_truth.space.one()) / vol

    fom = np.abs((l2_norm(data - data_mean) -
                  l2_norm(ground_truth - ground_truth_mean)))

    if normalized:
        fom /= (l2_norm(data - data_mean) +
                l2_norm(ground_truth - ground_truth_mean))

    return fom


def range_difference(data, ground_truth, mask=None, normalized=False):
    """Return dynamic range difference between ``data`` and ``ground_truth``.

    Evaluates difference in range between input (``data``) and reference
    data (``ground_truth``). Allows for normalization (``normalized``) and a
    masking of the two spaces (``mask``).

    Parameters
    ----------
    data : `Tensor`
        Input data to compare to the ground truth.
    ground_truth : `Tensor`
        Reference to compare ``data`` to.
    mask : `array-like`, optional
        Binary mask or index array to define ROI in which FOM evaluation
        is performed.
    normalized  : bool, optional
        If ``True``, normalize the FOM to lie in [0, 1].

    Returns
    -------
    rd : float
        FOM value, where a lower value means a better match.

    Notes
    -----
    The FOM evaluates

    .. math::
        \mathrm{RD}(f, g) = \Big|
            \\big(\\max(f) - \\min(f) \\big) -
            \\big(\\max(g) - \\min(g) \\big)
            \Big|

    or, in normalized form

    .. math::
        \mathrm{RD_N}(f, g) = \\frac{
            \Big|
            \\big(\\max(f) - \\min(f) \\big) -
            \\big(\\max(g) - \\min(g) \\big)
            \Big|}{
            \Big|
            \\big(\\max(f) - \\min(f) \\big) +
            \\big(\\max(g) - \\min(g) \\big)
            \Big|}

    The normalized variant takes values in :math:`[0, 1]`.
    """
    if mask is None:
        data_range = np.max(data) - np.min(data)
        ground_truth_range = np.max(ground_truth) - np.min(ground_truth)
    else:
        mask = np.asarray(mask)
        data_range = (np.max(data.asarray()[mask]) -
                      np.min(data.asarray()[mask]))
        ground_truth_range = (np.max(ground_truth.asarray()[mask]) -
                              np.min(ground_truth.asarray()[mask]))

    fom = np.abs(data_range - ground_truth_range)

    if normalized:
        fom /= np.abs(data_range + ground_truth_range)

    return fom


def blurring(data, ground_truth, mask=None, normalized=False,
             smoothness_factor=None):
    """Return weighted L2 distance, emphasizing regions defined by ``mask``.

    .. note::
        If the mask argument is omitted, this FOM is equivalent to the
        mean squared error.

    Parameters
    ----------
    data : `Tensor`
        Input data to compare to the ground truth.
    ground_truth : `Tensor`
        Reference to compare ``data`` to.
    mask : `array-like`, optional
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.
    smoothness_factor : float, optional
        Positive real number. Higher value gives smoother weighting.

    Returns
    -------
    blur : float
        FOM value, where a lower value means a better match.

    See Also
    --------
    false_structures
    mean_squared_error

    Notes
    -----
    The FOM evaluates

    .. math::
        \mathrm{BLUR}(f, g) = \|\\alpha (f - g) \|_2^2,

    or, in normalized form

    .. math::
        \mathrm{BLUR_N}(f, g) =
            \\frac{\|\\alpha(f - g)\|^2_2}
                  {\|\\alpha f\|^2_2 + \|\\alpha g\|^2_2}.

    The weighting function :math:`\\alpha` is given as

    .. math::
        \\alpha(x) = e^{-\\frac{1}{k} \\beta_m(x)},

    where :math:`\\beta_m(x)` is the Euclidian distance transform of a
    given binary mask :math:`m`, and :math:`k` positive real number that
    controls the smoothness of the weighting function :math:`\\alpha`.
    The weighting gives higher values to structures in the region of
    interest defined by the mask.

    The normalized variant takes values in :math:`[0, 1]`.
    """
    from scipy.ndimage.morphology import distance_transform_edt

    if smoothness_factor is None:
        smoothness_factor = np.mean(data.space.shape) / 10

    if mask is not None:
        mask = distance_transform_edt(1 - mask)
        mask = np.exp(-mask / smoothness_factor)

    return mean_squared_error(data, ground_truth, mask, normalized)


def false_structures(data, ground_truth, mask=None, normalized=False,
                     smoothness_factor=None):
    """Return weighted L2 distance, de-emphasizing regions defined by ``mask``.

    .. note::
        If the mask argument is omitted, this FOM is equivalent to the
        mean squared error.

    Parameters
    ----------
    data : `Tensor`
        Input data to compare to the ground truth.
    ground_truth : `Tensor`
        Reference to compare ``data`` to.
    mask : `array-like`, optional
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.
    smoothness_factor : float, optional
        Positive real number. Higher value gives smoother weighting.

    Returns
    -------
    fs : float
        FOM value, where a lower value means a better match.

    Notes
    -----
    The FOM evaluates

    .. math::
        \mathrm{FS} = \\big \| \\alpha^{-1} (f - g) \\big \|^2_2,

    or, in normalized form

    .. math::
        \mathrm{FS_N} = \\frac{\\big \| \\alpha^{-1}(f - g)\\big \|^2_2}
                              {\\big \| \\alpha^{-1} f \\big \|^2_2 +
                               \\big \| \\alpha^{-1} g \\big \|^2_2}.

    The weighting function :math:`\\alpha` is given as

    .. math::
        \\alpha(x) = e^{-\\frac{1}{k} \\beta_m(x)},

    where :math:`\\beta_m(x)` is the Euclidian distance transform of a
    given binary mask :math:`m`, and :math:`k` positive real number that
    controls the smoothness of the weighting function :math:`\\alpha`.
    The weighting gives higher values to structures outside the region
    of interest defined by the mask.
    """
    from scipy.ndimage.morphology import distance_transform_edt

    if smoothness_factor is None:
        smoothness_factor = np.mean(data.space.shape) / 10

    if mask is not None:
        mask = distance_transform_edt(1 - mask)
        mask = np.exp(mask / smoothness_factor)

    return mean_squared_error(data, ground_truth, mask, normalized)


def ssim(data, ground_truth,
         size=11, sigma=1.5, K1=0.01, K2=0.03, dynamic_range=None):
    """Structural SIMilarity between ``data`` and ``ground_truth``.

    The SSIM takes value -1 for maximum dissimilarity and +1 for maximum
    similarity.

    See also `this Wikipedia article
    <https://en.wikipedia.org/wiki/Structural_similarity>`_.

    Parameters
    ----------
    data : `Tensor`
        Input data to compare to the ground truth.
    ground_truth : `Tensor`
        Reference to compare ``data`` to.
    size : odd int
        Size in elements per axis of the Gaussian window that is used
        for all smoothing operations.
    sigma : positive float
        Width of the Gaussian function used for smoothing.
    K1, K2 : positive float
        TODO
    dynamic_range : nonnegative float
        TODO

    Returns
    -------
    ssim : float
        FOM value, where a higher value means a better match.
    """
    from scipy.signal import fftconvolve

    data = np.asarray(data)
    ground_truth = np.asarray(ground_truth)

    # Compute gaussian on a `size`-sized grid in each axis
    coords = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    grid = np.meshgrid(*([coords] * data.ndim), sparse=True)

    window = np.exp(-(sum(xi ** 2 for xi in grid) / (2.0 * sigma ** 2)))
    window /= np.sum(window)

    def smoothen(img):
        return fftconvolve(window, img, mode='valid')

    if dynamic_range is None:
        dynamic_range = np.max(ground_truth) - np.min(ground_truth)

    C1 = (K1 * dynamic_range) ** 2
    C2 = (K2 * dynamic_range) ** 2
    mu1 = smoothen(data)
    mu2 = smoothen(ground_truth)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = smoothen(data * data) - mu1_sq
    sigma2_sq = smoothen(ground_truth * ground_truth) - mu2_sq
    sigma12 = smoothen(data * ground_truth) - mu1_mu2

    nom = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    pointwise_ssim = nom / denom

    return np.mean(pointwise_ssim)


def psnr(data, ground_truth, normalized=False):
    """Return the Peak Signal-to-Noise Ratio of ``data`` wrt ``ground_truth``.

    See also `this Wikipedia article
    <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.

    Parameters
    ----------
    data : `Tensor`
        Input data to compare to the ground truth.
    ground_truth : `Tensor`
        Reference to compare ``data`` to.
    normalized : bool
        If true, normalize ``data`` and ``ground_truth`` to have the
        same mean and variance before comparison.

    Returns
    -------
    psnr : float
        FOM value, where a higher value means a better match.

    Examples
    --------
    Compute the PSNR for two vectors:

    >>> spc = odl.rn(5)
    >>> data = spc.element([1, 1, 1, 1, 1])
    >>> ground_truth = spc.element([1, 1, 1, 1, 2])
    >>> result = psnr(data, ground_truth)
    >>> print('{:.3f}'.format(result))
    13.010

    If data == ground_truth, the result is positive infinity:

    >>> psnr(ground_truth, ground_truth)
    inf

    With ``normalized=True``, scaling differences and constant offsets
    are ignored:

    >>> (psnr(data, ground_truth, normalized=True) ==
    ...  psnr(data, 3 + 4 * ground_truth, normalized=True))
    True
    """
    if normalized:
        data = odl.util.zscore(data)
        ground_truth = odl.util.zscore(ground_truth)

    mse = mean_squared_error(data, ground_truth)
    max_true = np.max(np.abs(ground_truth))

    if mse == 0:
        return np.inf
    elif max_true == 0:
        return -np.inf
    else:
        return 20 * np.log10(max_true) - 10 * np.log10(mse)


def haarpsi(data, ground_truth, a=4.2, c=None):
    """Haar-Wavelet based perceptual similarity index FOM.

    This function evaluates the structural similarity between two images
    based on edge features along the coordinate axes, analyzed with two
    wavelet filter levels. See
    `[Rei+2016] <https://arxiv.org/abs/1607.06140>`_ and the Notes section
    for further details.

    Parameters
    ----------
    data : 2D array-like
        The image to compare to the ground truth.
    ground_truth : 2D array-like
        The true image with which to compare ``data``. It must have the
        same shape as ``data``.
    a : positive float, optional
        Parameter in the logistic function. Larger value leads to a
        steeper curve, thus lowering the threshold for an input to
        be mapped to an output close to 1. See Notes for details.
        The default value 4.2 is taken from the referenced paper.
    c : positive float, optional
        Constant determining the score of maximally dissimilar values.
        Smaller constant means higher penalty for dissimilarity.
        See `haarpsi_similarity_map` for details.
        For ``None``, the value is chosen as
        ``3 * sqrt(max(abs(ground_truth)))``.

    Returns
    -------
    haarpsi : float between 0 and 1
        The similarity score, where a higher score means a better match.
        See Notes for details.

    See Also
    --------
    haarpsi_similarity_map
    haarpsi_weight_map

    Notes
    -----
    For input images :math:`f_1, f_2`, the HaarPSI score is defined as

    .. math::
        \mathrm{HaarPSI}_{f_1, f_2} =
        l_a^{-1} \\left(
        \\frac{
        \sum_x \sum_{k=1}^2 \mathrm{HS}_{f_1, f_2}^{(k)}(x) \cdot
        \mathrm{W}_{f_1, f_2}^{(k)}(x)}{
        \sum_x \sum_{k=1}^2 \mathrm{W}_{f_1, f_2}^{(k)}(x)}
        \\right)^2

    see `[Rei+2016] <https://arxiv.org/abs/1607.06140>`_ equation (12).

    For the definitions of the constituting functions, see

        - `haarpsi_similarity_map` for :math:`\mathrm{HS}_{f_1, f_2}^{(k)}`,
        - `haarpsi_weight_map` for :math:`\mathrm{W}_{f_1, f_2}^{(k)}`.

    References
    ----------
    [Rei+2016] Reisenhofer, R, Bosse, S, Kutyniok, G, and Wiegand, T.
    *A Haar Wavelet-Based Perceptual Similarity Index for Image Quality
    Assessment*. arXiv:1607.06140 [cs], Jul. 2016.
    """
    import scipy.special
    from odl.contrib.fom.util import haarpsi_similarity_map, haarpsi_weight_map

    if c is None:
        c = 3 * np.sqrt(np.max(np.abs(ground_truth)))

    lsim_horiz = haarpsi_similarity_map(data, ground_truth, axis=0, c=c, a=a)
    lsim_vert = haarpsi_similarity_map(data, ground_truth, axis=1, c=c, a=a)

    wmap_horiz = haarpsi_weight_map(data, ground_truth, axis=0)
    wmap_vert = haarpsi_weight_map(data, ground_truth, axis=1)

    numer = np.sum(lsim_horiz * wmap_horiz + lsim_vert * wmap_vert)
    denom = np.sum(wmap_horiz + wmap_vert)

    return (scipy.special.logit(numer / denom) / a) ** 2


def noise_power_spectrum(data, ground_truth, radial=False):
    """Return the Noise Power Spectrum (NPS).

    The NPS is given by the squared magnitude of the Fourier transform of the
    noise.

    Parameters
    ----------
    data : `DiscreteLp` element
        Input data or reconstruction.
    ground_truth : `DiscreteLp` element
        Reference to compare ``data`` to.
    radial : bool
        If true, compute the radial NPS.

    Returns
    -------
    noise_power_spectrum : `DiscreteLp`-element
        The space is the Fourier space corresponding to ``data.space``, and
        hence the axes indicate frequency.
        If ``radial`` is ``True``, this is an element in a one-dimensional
        space of the same type as ``data.space``.
    """
    fftop = odl.trafos.FourierTransform(data.space, halfcomplex=False)
    f1 = fftop(data - ground_truth)
    nps = np.abs(f1).real ** 2

    if radial:
        r = np.sqrt(sum(xi ** 2 for xi in nps.space.meshgrid))
        n_bins = max(*nps.shape)
        radial_nps, rad = np.histogram(r, weights=nps, bins=n_bins)

        out_spc = odl.uniform_discr(0, np.max(r), n_bins,
                                    impl=f1.space.impl,
                                    dtype=f1.space.real_dtype,
                                    axis_labels=['$\\^{r}$'])

        return out_spc.element(radial_nps)
    else:
        return nps


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
