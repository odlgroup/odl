# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Figures of merit (FOMs) for comparison against a known ground truth."""

import odl
import numpy as np

__all__ = ('mean_squared_error', 'mean_absolute_error',
           'mean_value_difference', 'standard_deviation_difference',
           'range_difference', 'blurring', 'false_structures', 'ssim', 'psnr',
           'haarpsi')


def mean_squared_error(data, ground_truth, mask=None, normalized=False):
    """Return L2-distance between ``data`` and ``ground_truth``.

    Evaluates `mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between
    input (``data``) and reference (``ground_truth``) Allows for normalization
    (``normalized``) and a masking of the two spaces (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
        \| f - g \|^2_2,

    or, in normalized form

    .. math::
        \\frac{\| f - g \|^2_2}{\| f \|^2_2 + \| g \|^2_2}.

    The normalized FOM takes values in [0, 1].

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`, optional
        Mask to define ROI in which FOM evaluation is performed. The mask is
        allowed to be weighted (i.e. non-binary), see ``blurring`` and
        ``false_structures.``
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : float
        Scalar (float) indicating mean squared error between ``data`` and
        ``ground_truth``. In normalized form the FOM takes values in
        [0, 1], with higher correspondance at lower FOM value.
    """
    l2_normSquared = odl.solvers.L2NormSquared(data.space)

    if mask is not None:
        data = data * mask
        ground_truth = ground_truth * mask

    diff = data - ground_truth
    fom = l2_normSquared(diff)

    # Volume of space
    vol = l2_normSquared(data.space.one())
    fom /= vol

    if normalized:
            fom /= (l2_normSquared(data) + l2_normSquared(ground_truth))

    return fom


def mean_absolute_error(data, ground_truth, mask=None, normalized=False):
    """Return L1-distance between ``data`` and ``ground_truth``.

    Evaluates `mean absolute error
    <https://en.wikipedia.org/wiki/Mean_absolute_error>`_ between
    input (``data``) and reference (``ground_truth``). Allows for normalization
    (``normalized``) and a masking of the two spaces (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
        \| f - g \|_1,

    or, in normalized form

    .. math::
        \\frac{\| f - g \|_1}{\| f \|_1 + \| g \|_1}.

    The normalized FOM takes values in [0, 1].

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`, optional
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : float
        Scalar (float) indicating mean absolute error between ``data`` and
        ``ground_truth``. In normalized form the FOM takes values in
        [0, 1], with higher correspondance at lower FOM value.
    """
    l1_norm = odl.solvers.L1Norm(data.space)
    if mask:
        data = data * mask
        ground_truth = ground_truth * mask
    diff = data - ground_truth
    fom = l1_norm(diff)

    # Volume of space
    vol = l1_norm(data.space.one())
    fom /= vol

    if normalized:
        fom /= (l1_norm(data) + l1_norm(ground_truth))

    return fom


def mean_value_difference(data, ground_truth, mask=None, normalized=False):
    """Return difference in mean value between ``data`` and ``ground_truth``.

    Evaluates difference in `mean value
    <https://en.wikipedia.org/wiki/Mean_of_a_function>`_ between input
    (``data``) and reference (``ground_truth``). Allows for normalization
    (``normalized``) and a masking of the two spaces (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
         \\bigg \\lvert \\lvert  \\overline{f} \\rvert -
                \\lvert \\overline{g} \\rvert \\bigg \\rvert,

    or, in normalized form

    .. math::
         \\bigg \\lvert \\frac{\\lvert \\overline{f} \\rvert -
                               \\lvert \\overline{g} \\rvert}
                              {\\lvert \\overline{f} \\rvert +
                               \\lvert \\overline{g} \\rvert} \\bigg \\rvert

    where

    .. math::
        \\overline{f} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega f dx,

    and

    .. math::
        \\overline{g} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega g dx.

    The normalized FOM takes values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`, optional
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : float
        Scalar (float) indicating difference in mean value between
        ``data`` and ``ground_truth``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """
    l1_norm = odl.solvers.L1Norm(data.space)
    if mask:
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

    Evaluates difference in standard deviation (std) between input (``data``)
    and reference (``ground_truth``). Allows for normalization (``normalized``)
    and a masking of the two spaces (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
         \\lvert \| f - \\overline{f} \|_2 -
                 \| g - \\overline{g} \|_2 \\rvert,

    or, in normalized form

    .. math::
        \\bigg \\lvert \\frac{\| f - \\overline{f} \|_2 -
                              \| g - \\overline{g} \|_2}
                             {\| f - \\overline{f} \|_2 +
                              \| g - \\overline{g} \|_2 } \\bigg \\rvert,

    where

    .. math::
        \\overline{f} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega f dx,

    and

    .. math::
        \\overline{g} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega g dx.

    The normalized FOM takes values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`, optional
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : float
        Scalar (float) indicating absolute difference in standard deviation
        between ``data`` and ``ground_truth``. In normalized form the FOM
        takes values in [0, 1], with higher correspondance at lower FOM value.
    """
    l1_norm = odl.solvers.L1Norm(data.space)
    l2_norm = odl.solvers.L2Norm(data.space)

    if mask:
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
    """Return difference in range between ``data`` and ``ground_truth``.

    Evaluates difference in range between input (``data``) and reference
    data (``ground_truth``). Allows for normalization (``normalized``) and a
    masking of the two spaces (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
        \\lvert \\left(\\max(f) - \\min(f) \\right) -
                \\left(\\max(g) - \\min(g) \\right) \\rvert

    or, in normalized form

    .. math::
        \\bigg \\lvert \\frac{\\left(\\max(f) - \\min(f) \\right) -
                              \\left(\\max(g) - \\min(g)\\right)}
                             {\\left(\\max(f) - \\min(f)\\right) +
                              \\left(\\max(g) - \\min(g)\\right)}
        \\bigg \\rvert

    The normalized FOM takes values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`, optional
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized
        FOM.alse_structures.

    Returns
    -------
    fom : float
        Scalar (float) indicating absolute difference in range between
        ``data`` and ``ground_truth``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """
    if mask:
        indices = np.where(mask is True)
        data_range = (np.max(data.asarray()[indices]) -
                      np.min(data.asarray()[indices]))
        ground_truth_range = (np.max(ground_truth.asarray()[indices]) -
                              np.min(ground_truth.asarray()[indices]))
    else:
        data_range = np.max(data) - np.min(data)
        ground_truth_range = np.max(ground_truth) - np.min(ground_truth)

    fom = np.abs(data_range - ground_truth_range)

    if normalized:
        fom /= np.abs(data_range + ground_truth_range)

    return fom


def blurring(data, ground_truth, mask=None, normalized=False,
             smoothness_factor=None):
    """Return weighted L2-distance, emphasizing regions defined by ``mask``.

    Evaluates `mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between input
    (``data``) and reference data (``ground_truth``) using an added binary
    mask (``mask``), such that the error is weighted with higher importance
    given to the defined structure-of-interest. Allows for normalization
    (``normalized``).

    .. note:: If omitting the mask argument, the blurring FOM is equivalent
              to the mean squared error FOM.

    Notes
    ----------
    The FOM evaluates

    .. math::
        \|\\alpha (f - g) \|^2_2,

    or, in normalized form

    .. math::
        \\frac{\| \\alpha (f - g) \|^2_2}{\| \\alpha f \|^2_2 +
                                  \| \\alpha  g \|^2_2},

    where :math:`\\alpha` is a weighting function with higher values near a
    structure of interest defined by ``mask``. The weighting function is given
    as

    .. math::
        \\alpha = e^{-\\frac{1}{k} \\beta},

    where :math:`\\beta(x)` is the Euclidian distance from :math:`x` to the
    complement of the structure of interest, and :math:`k`
    (``smoothness_factor``) is a positive real number that controls the
    'smoothness' of the weighting function :math:`\\alpha`.

    The normalized FOM takes values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`, optional
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.
    smoothness_factor : float, optional
        Positive real number. Higher values gives smoother weighting.

    Returns
    -------
    fom : float
        Scalar (float) indicating weighted mean squared error between
        ``data`` and ``ground_truth``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """
    import scipy.ndimage.morphology as scimorph

    if smoothness_factor is None:
        smoothness_factor = np.mean(data.space.shape) / 10

    if mask is not None:
        mask = scimorph.distance_transform_edt(1 - mask)
        mask = np.exp(-mask / smoothness_factor)

    fom = mean_squared_error(data,
                             ground_truth,
                             mask=mask,
                             normalized=normalized)

    return fom


def false_structures(data, ground_truth, mask=None, normalized=False,
                     smoothness_factor=None):
    """Return weighted L2-distance, de-emphasizing regions defined by ``mask``.

    Evaluates `mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between input
    (``data``) and reference data (``ground_truth``) using an added binary
    mask (``mask``), such that the error is weighted with lower importance
    given to the defined structure-of-interest. Allows for normalization
    (``normalized``).

    .. note:: If omitting the mask argument, the false structures FOM is
              equivalent to the mean squared error FOM.

    Notes
    ----------
    The FOM evaluates

    .. math::
        \\bigg \| \\frac{1}{\\alpha} (f - g) \\bigg \|^2_2,

    or, in normalized form

    .. math::
        \\frac{\\bigg \| \\frac{1}{\\alpha} (f - g) \\bigg \|^2_2}
              {\\bigg \| \\frac{1}{\\alpha} f \\bigg \|^2_2 +
               \\bigg \| \\frac{1}{\\alpha} g \\bigg \|^2_2},

    where :math:`\\alpha` is a weighting function with higher values near a
    structure of interest defined by ``mask``. The weighting function is given
    as

    .. math::
        \\alpha = e^{-\\frac{1}{k} \\beta},

    where :math:`\\beta(x)` is the Euclidian distance from :math:`x` to the
    complement of the structure of interest, and :math:`k`
    (``smoothness_factor``) is a positive real number that controls the
    'smoothness' of the weighting function :math:`\\alpha`.

    The normalized FOM takes values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare 'data' to.
    mask : `FnBaseVector`, optional
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.
    smoothness_factor : float, optional
        Positive real number. Higher values gives smoother weighting.

    Returns
    -------
    fom : float
        Scalar (float) indicating weighted mean squared error between
        ``data`` and ``ground_truth``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """
    import scipy.ndimage.morphology as scimorph

    if smoothness_factor is None:
        smoothness_factor = np.mean(data.space.shape) / 10

    if mask is not None:
        mask = scimorph.distance_transform_edt(1 - mask)
        mask = np.exp(mask / smoothness_factor)

    fom = mean_squared_error(data,
                             ground_truth,
                             mask=mask,
                             normalized=normalized)

    return fom


def ssim(data, ground_truth,
         size=11, sigma=1.5, K1=0.01, K2=0.03, dynamic_range=None,
         normalized=False):
    """Structural SIMilarity between ``data`` and ``ground_truth``.

    Evaluates `structural similarity
    <https://en.wikipedia.org/wiki/Structural_similarity>`_ between
    input (``data``) and reference (``ground_truth``).

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    normalized : bool
        TEXT

    Returns
    -------
    fom : float
        Scalar (float) indicating structural similarity between ``data`` and
        ``ground_truth``. Takes values in [-1, 1], with -1 indicating full
        dis-similarity and 1 full similarity. Uncorrelated images will have
        similarity index 0. In normalized form the FOM values are rescaled to
        [0, 1], with higher correspondance at lower FOM value.
    """
    from scipy.signal import fftconvolve

    data = np.asarray(data)
    ground_truth = np.asarray(ground_truth)
    ndim = data.ndim

    # Compute gaussian
    coords = np.meshgrid(*(ndim * (np.linspace(-(size - 1) / 2,
                                               (size - 1) / 2, size),)))

    window = np.exp(-(sum(xi**2 for xi in coords) / (2.0 * sigma**2)))
    window /= np.sum(window)

    def smoothen(img):
        return fftconvolve(window, img, mode='valid')

    if dynamic_range is None:
        dynamic_range = np.max(ground_truth) - np.min(ground_truth)

    C1 = (K1 * dynamic_range)**2
    C2 = (K2 * dynamic_range)**2
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

    ssim = np.mean(pointwise_ssim)

    if normalized:
        return 0.5 - ssim / 2
    else:
        return ssim


def psnr(data, ground_truth, normalize=False):
    """Return the Peak Signal-to-Noise Ratio.

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    normalize : bool
        If true, normalizes ``data`` and ``ground_truth`` to have the same mean
        and variance before comparison.

    Returns
    -------
    psnr : float

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

    With ``normalize=True``, internal scaling and constant offsets are ignored:

    >>> (psnr(data, ground_truth, normalize=True) ==
    ...  psnr(data, 3 + 4 * ground_truth, normalize=True))
    True
    """
    if normalize:
        data = odl.util.zscore(data)
        ground_truth = odl.util.zscore(ground_truth)

    mse_result = mean_squared_error(data, ground_truth)
    max_true = np.max(np.abs(ground_truth))

    if mse_result == 0:
        return np.inf
    elif max_true == 0:
        return -np.inf
    else:
        return 20 * np.log10(max_true) - 10 * np.log10(mse_result)


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
    score : float between 0 and 1
        The similarity score. See Notes for details.

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


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
