# coding: utf-8
# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.discr import ResizingOperator
from odl.trafos import FourierTransform, PYFFTW_AVAILABLE


__all__ = ('fbp_op', 'fbp_filter_op', 'tam_danielson_window',
           'parker_weighting')


def _axis_in_detector(geometry):
    """A vector in the detector plane that points along the rotation axis."""
    du, dv = geometry.det_axes_init
    axis = geometry.axis
    c = np.array([np.vdot(axis, du), np.vdot(axis, dv)])
    cnorm = np.linalg.norm(c)

    # Check for numerical errors
    assert cnorm != 0

    return c / cnorm


def _rotation_direction_in_detector(geometry):
    """A vector in the detector plane that points in the rotation direction."""
    du, dv = geometry.det_axes_init
    axis = geometry.axis
    det_normal = np.cross(dv, du)
    rot_dir = np.cross(axis, det_normal)
    c = np.array([np.vdot(rot_dir, du), np.vdot(rot_dir, dv)])
    cnorm = np.linalg.norm(c)

    # Check for numerical errors
    assert cnorm != 0

    return c / cnorm


def _fbp_filter(norm_freq, filter_type, frequency_scaling):
    """Create a smoothing filter for FBP.

    Parameters
    ----------
    norm_freq : `array-like`
        Frequencies normalized to lie in the interval [0, 1].
    filter_type : {'Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann',
                   callable}
        The type of filter to be used.
        If a string is given, use one of the standard filters with that name.
        A callable should take an array of values in [0, 1] and return the
        filter for these frequencies.
    frequency_scaling : float
        Scaling of the frequencies for the filter. All frequencies are scaled
        by this number, any relative frequency above ``frequency_scaling`` is
        set to 0.

    Returns
    -------
    smoothing_filter : `numpy.ndarray`

    Examples
    --------
    Create an FBP filter

    >>> norm_freq = np.linspace(0, 1, 10)
    >>> filt = _fbp_filter(norm_freq,
    ...                    filter_type='Hann',
    ...                    frequency_scaling=0.8)
    """
    filter_type, filter_type_in = str(filter_type).lower(), filter_type
    if callable(filter_type):
        filt = filter_type(norm_freq)
    elif filter_type == 'ram-lak':
        filt = np.copy(norm_freq)
    elif filter_type == 'shepp-logan':
        filt = norm_freq * np.sinc(norm_freq / (2 * frequency_scaling))
    elif filter_type == 'cosine':
        filt = norm_freq * np.cos(norm_freq * np.pi / (2 * frequency_scaling))
    elif filter_type == 'hamming':
        filt = norm_freq * (
            0.54 + 0.46 * np.cos(norm_freq * np.pi / (frequency_scaling)))
    elif filter_type == 'hann':
        filt = norm_freq * (
            np.cos(norm_freq * np.pi / (2 * frequency_scaling)) ** 2)
    else:
        raise ValueError('unknown `filter_type` ({})'
                         ''.format(filter_type_in))

    indicator = (norm_freq <= frequency_scaling)
    filt *= indicator
    return filt


def tam_danielson_window(ray_trafo, smoothing_width=0.05, n_pi=1):
    """Create Tam-Danielson window from a `RayTransform`.

    The Tam-Danielson window is an indicator function on the minimal set of
    data needed to reconstruct a volume from given data. It is useful in
    analytic reconstruction methods such as FBP to give a more accurate
    reconstruction.

    See [TAM1998] for more informationon the window.
    See [PKGT2000] for information on the ``n_pi`` parameter.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform for which to compute the window.
    smoothing_width : positive float, optional
        Width of the smoothing applied to the window's edges given as a
        fraction of the width of the full window.
    n_pi : odd int, optional
        Total number of half rotations to include in the window. Values larger
        than 1 should be used if the pitch is much smaller than the detector
        height.

    Returns
    -------
    tam_danielson_window : ``ray_trafo.range`` element

    See Also
    --------
    fbp_op : Filtered back-projection operator from `RayTransform`
    tam_danielson_window : Weighting for short scan data
    odl.tomo.geometry.conebeam.ConeFlatGeometry :
        Primary use case for this window function.

    References
    ----------
    [TSS1998] Tam, K C, Samarasekera, S and Sauer, F.
    *Exact cone beam CT with a spiral scan*.
    Physics in Medicine & Biology 4 (1998), p 1015.
    https://dx.doi.org/10.1088/0031-9155/43/4/028

    [PKGT2000] Proksa R, Köhler T, Grass M, Timmer J.
    *The n-PI-method for helical cone-beam CT*
    IEEE Trans Med Imaging. 2000 Sep;19(9):848-63.
    https://www.ncbi.nlm.nih.gov/pubmed/11127600
    """
    # Extract parameters
    src_radius = ray_trafo.geometry.src_radius
    det_radius = ray_trafo.geometry.det_radius
    pitch = ray_trafo.geometry.pitch

    if pitch == 0:
        raise ValueError('Tam-Danielson window is only defined with '
                         '`pitch!=0`')

    smoothing_width = float(smoothing_width)
    if smoothing_width < 0:
        raise ValueError('`smoothing_width` should be a positive float')

    if n_pi % 2 != 1:
        raise ValueError('`n_pi` must be odd, got {}'.format(n_pi))

    # Find projection of axis on detector
    axis_proj = _axis_in_detector(ray_trafo.geometry)
    rot_dir = _rotation_direction_in_detector(ray_trafo.geometry)

    # Find distance from projection of rotation axis for each pixel
    dx = (rot_dir[0] * ray_trafo.range.meshgrid[1]
          + rot_dir[1] * ray_trafo.range.meshgrid[2])

    dx_axis = dx * src_radius / (src_radius + det_radius)

    def Vn(u):
        return (pitch / (2 * np.pi)
                * (1 + (u / src_radius) ** 2)
                * (n_pi * np.pi / 2.0 - np.arctan(u / src_radius)))

    lower_proj_axis = -Vn(dx_axis)
    upper_proj_axis = Vn(-dx_axis)

    lower_proj = lower_proj_axis * (src_radius + det_radius) / src_radius
    upper_proj = upper_proj_axis * (src_radius + det_radius) / src_radius

    # Compute a smoothed width
    interval = (upper_proj - lower_proj)
    width = interval * smoothing_width / np.sqrt(2)

    # Create window function
    def window_fcn(x):
        # Lazy import to improve `import odl` time
        import scipy.special

        x_along_axis = axis_proj[0] * x[1] + axis_proj[1] * x[2]
        if smoothing_width != 0:
            lower_wndw = 0.5 * (
                1 + scipy.special.erf((x_along_axis - lower_proj) / width))
            upper_wndw = 0.5 * (
                1 + scipy.special.erf((upper_proj - x_along_axis) / width))
        else:
            lower_wndw = (x_along_axis >= lower_proj)
            upper_wndw = (x_along_axis <= upper_proj)

        return lower_wndw * upper_wndw

    return ray_trafo.range.element(window_fcn) / n_pi


def parker_weighting(ray_trafo, q=0.25):
    """Create parker weighting for a `RayTransform`.

    Parker weighting is a weighting function that ensures that oversampled
    fan/cone beam data are weighted such that each line has unit weight. It is
    useful in analytic reconstruction methods such as FBP to give a more
    accurate result and can improve convergence rates for iterative methods.

    See the article `Parker weights revisited`_ for more information.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform for which to compute the weights.
    q : float, optional
        Parameter controlling the speed of the roll-off at the edges of the
        weighting. 1.0 gives the classical Parker weighting, while smaller
        values in general lead to lower noise but stronger discretization
        artifacts.

    Returns
    -------
    parker_weighting : ``ray_trafo.range`` element

    See Also
    --------
    fbp_op : Filtered back-projection operator from `RayTransform`
    tam_danielson_window : Indicator function for helical data
    odl.tomo.geometry.conebeam.FanBeamGeometry : Use case in 2d
    odl.tomo.geometry.conebeam.ConeFlatGeometry : Use case in 3d (for pitch 0)

    References
    ----------
    .. _Parker weights revisited: https://www.ncbi.nlm.nih.gov/pubmed/11929021
    """
    # Note: Parameter names taken from WES2002

    # Extract parameters
    src_radius = ray_trafo.geometry.src_radius
    det_radius = ray_trafo.geometry.det_radius
    ndim = ray_trafo.geometry.ndim
    angles = ray_trafo.range.meshgrid[0]
    min_rot_angle = ray_trafo.geometry.motion_partition.min_pt
    alen = ray_trafo.geometry.motion_params.length

    # Parker weightings are not defined for helical geometries
    if ray_trafo.geometry.ndim != 2:
        pitch = ray_trafo.geometry.pitch
        if pitch != 0:
            raise ValueError('Parker weighting window is only defined with '
                             '`pitch==0`')

    # Find distance from projection of rotation axis for each pixel
    if ndim == 2:
        dx = ray_trafo.range.meshgrid[1]
    elif ndim == 3:
        # Find projection of axis on detector
        rot_dir = _rotation_direction_in_detector(ray_trafo.geometry)
        # If axis is aligned to a coordinate axis, save some memory and time by
        # using broadcasting
        if rot_dir[0] == 0:
            dx = rot_dir[1] * ray_trafo.range.meshgrid[2]
        elif rot_dir[1] == 0:
            dx = rot_dir[0] * ray_trafo.range.meshgrid[1]
        else:
            dx = (rot_dir[0] * ray_trafo.range.meshgrid[1]
                  + rot_dir[1] * ray_trafo.range.meshgrid[2])

    # Compute parameters
    dx_abs_max = np.max(np.abs(dx))
    max_fan_angle = 2 * np.arctan2(dx_abs_max, src_radius + det_radius)
    delta = max_fan_angle / 2
    epsilon = alen - np.pi - max_fan_angle

    if epsilon < 0:
        raise Exception('data not sufficiently sampled for parker weighting')

    # Define utility functions
    def S(betap):
        return (0.5 * (1.0 + np.sin(np.pi * betap)) * (np.abs(betap) < 0.5)
                + (betap >= 0.5))

    def b(alpha):
        return q * (2 * delta - 2 * alpha + epsilon)

    # Create weighting function
    beta = np.asarray(angles - min_rot_angle,
                      dtype=ray_trafo.range.dtype)  # rotation angle
    alpha = np.asarray(np.arctan2(dx, src_radius + det_radius),
                       dtype=ray_trafo.range.dtype)

    # Compute sum in place to save memory
    S_sum = S(beta / b(alpha) - 0.5)
    S_sum += S((beta - 2 * delta + 2 * alpha - epsilon) / b(alpha) + 0.5)
    S_sum -= S((beta - np.pi + 2 * alpha) / b(-alpha) - 0.5)
    S_sum -= S((beta - np.pi - 2 * delta - epsilon) / b(-alpha) + 0.5)

    scale = 0.5 * alen / np.pi
    return ray_trafo.range.element(
        np.broadcast_to(S_sum * scale, ray_trafo.range.shape))


def fbp_filter_op(ray_trafo, padding=True, filter_type='Ram-Lak',
                  frequency_scaling=1.0):
    """Create a filter operator for FBP from a `RayTransform`.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform (forward operator) whose approximate inverse should
        be computed. Its geometry has to be any of the following

        `Parallel2dGeometry` : Exact reconstruction

        `Parallel3dAxisGeometry` : Exact reconstruction

        `FanBeamGeometry` : Approximate reconstruction, correct in limit of
        fan angle = 0.
        Only flat detectors are supported (det_curvature_radius is None).

        `ConeFlatGeometry`, pitch = 0 (circular) : Approximate reconstruction,
        correct in the limit of fan angle = 0 and cone angle = 0.

        `ConeFlatGeometry`, pitch > 0 (helical) : Very approximate unless a
        `tam_danielson_window` is used. Accurate with the window.

        Other geometries: Not supported

    padding : bool, optional
        If the data space should be zero padded. Without padding, the data may
        be corrupted due to the circular convolution used. Using padding makes
        the algorithm slower.
    filter_type : optional
        The type of filter to be used.
        The predefined options are, in approximate order from most noise
        senstive to least noise sensitive:
        ``'Ram-Lak'``, ``'Shepp-Logan'``, ``'Cosine'``, ``'Hamming'`` and
        ``'Hann'``.
        A callable can also be provided. It must take an array of values in
        [0, 1] and return the filter for these frequencies.
    frequency_scaling : float, optional
        Relative cutoff frequency for the filter.
        The normalized frequencies are rescaled so that they fit into the range
        [0, frequency_scaling]. Any frequency above ``frequency_scaling`` is
        set to zero.

    Returns
    -------
    filter_op : `Operator`
        Filtering operator for FBP based on ``ray_trafo``.

    See Also
    --------
    tam_danielson_window : Windowing for helical data
    """
    impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'
    alen = ray_trafo.geometry.motion_params.length

    if ray_trafo.domain.ndim == 2:
        # Define ramp filter
        def fourier_filter(x):
            abs_freq = np.abs(x[1])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(norm_freq, filter_type, frequency_scaling)
            scaling = 1 / (2 * alen)
            return filt * np.max(abs_freq) * scaling

        # Define (padded) fourier transform
        if padding:
            # Define padding operator
            ran_shp = (ray_trafo.range.shape[0],
                       ray_trafo.range.shape[1] * 2 - 1)
            resizing = ResizingOperator(ray_trafo.range, ran_shp=ran_shp)

            fourier = FourierTransform(resizing.range, axes=1, impl=impl)
            fourier = fourier * resizing
        else:
            fourier = FourierTransform(ray_trafo.range, axes=1, impl=impl)

    elif ray_trafo.domain.ndim == 3:
        # Find the direction that the filter should be taken in
        rot_dir = _rotation_direction_in_detector(ray_trafo.geometry)

        # Find what axes should be used in the fourier transform
        used_axes = (rot_dir != 0)
        if used_axes[0] and not used_axes[1]:
            axes = [1]
        elif not used_axes[0] and used_axes[1]:
            axes = [2]
        else:
            axes = [1, 2]

        # Add scaling for cone-beam case
        if hasattr(ray_trafo.geometry, 'src_radius'):
            scale = (ray_trafo.geometry.src_radius
                     / (ray_trafo.geometry.src_radius
                        + ray_trafo.geometry.det_radius))

            if ray_trafo.geometry.pitch != 0:
                # In helical geometry the whole volume is not in each
                # projection and we need to use another weighting.
                # Ideally each point in the volume effects only
                # the projections in a half rotation, so we assume that that
                # is the case.
                scale *= alen / (np.pi)
        else:
            scale = 1.0

        # Define ramp filter
        def fourier_filter(x):
            # If axis is aligned to a coordinate axis, save some memory and
            # time by using broadcasting
            if not used_axes[0]:
                abs_freq = np.abs(rot_dir[1] * x[2])
            elif not used_axes[1]:
                abs_freq = np.abs(rot_dir[0] * x[1])
            else:
                abs_freq = np.abs(rot_dir[0] * x[1] + rot_dir[1] * x[2])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(norm_freq, filter_type, frequency_scaling)
            scaling = scale * np.max(abs_freq) / (2 * alen)
            return filt * scaling

        # Define (padded) fourier transform
        if padding:
            # Define padding operator
            if used_axes[0]:
                padded_shape_u = ray_trafo.range.shape[1] * 2 - 1
            else:
                padded_shape_u = ray_trafo.range.shape[1]

            if used_axes[1]:
                padded_shape_v = ray_trafo.range.shape[2] * 2 - 1
            else:
                padded_shape_v = ray_trafo.range.shape[2]

            ran_shp = (ray_trafo.range.shape[0],
                       padded_shape_u,
                       padded_shape_v)
            resizing = ResizingOperator(ray_trafo.range, ran_shp=ran_shp)

            fourier = FourierTransform(resizing.range, axes=axes, impl=impl)
            fourier = fourier * resizing
        else:
            fourier = FourierTransform(ray_trafo.range, axes=axes, impl=impl)
    else:
        raise NotImplementedError('FBP only implemented in 2d and 3d')

    # Create ramp in the detector direction
    ramp_function = fourier.range.element(fourier_filter)

    weight = 1
    if not ray_trafo.range.is_weighted:
        # Compensate for potentially unweighted range of the ray transform
        weight *= ray_trafo.range.cell_volume

    if not ray_trafo.domain.is_weighted:
        # Compensate for potentially unweighted domain of the ray transform
        weight /= ray_trafo.domain.cell_volume

    ramp_function *= weight

    # Create ramp filter via the convolution formula with fourier transforms
    return fourier.inverse * ramp_function * fourier


def fbp_op(ray_trafo, padding=True, filter_type='Ram-Lak',
           frequency_scaling=1.0):
    """Create filtered back-projection operator from a `RayTransform`.

    The filtered back-projection is an approximate inverse to the ray
    transform.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform (forward operator) whose approximate inverse should
        be computed. Its geometry has to be any of the following

        `Parallel2dGeometry` : Exact reconstruction

        `Parallel3dAxisGeometry` : Exact reconstruction

        `FanBeamGeometry` : Approximate reconstruction, correct in limit of fan
        angle = 0.
        Only flat detectors are supported (det_curvature_radius is None).

        `ConeFlatGeometry`, pitch = 0 (circular) : Approximate reconstruction,
        correct in the limit of fan angle = 0 and cone angle = 0.

        `ConeFlatGeometry`, pitch > 0 (helical) : Very approximate unless a
        `tam_danielson_window` is used. Accurate with the window.

        Other geometries: Not supported

    padding : bool, optional
        If the data space should be zero padded. Without padding, the data may
        be corrupted due to the circular convolution used. Using padding makes
        the algorithm slower.
    filter_type : optional
        The type of filter to be used.
        The predefined options are, in approximate order from most noise
        senstive to least noise sensitive:
        ``'Ram-Lak'``, ``'Shepp-Logan'``, ``'Cosine'``, ``'Hamming'`` and
        ``'Hann'``.
        A callable can also be provided. It must take an array of values in
        [0, 1] and return the filter for these frequencies.
    frequency_scaling : float, optional
        Relative cutoff frequency for the filter.
        The normalized frequencies are rescaled so that they fit into the range
        [0, frequency_scaling]. Any frequency above ``frequency_scaling`` is
        set to zero.

    Returns
    -------
    fbp_op : `Operator`
        Approximate inverse operator of ``ray_trafo``.

    See Also
    --------
    tam_danielson_window : Windowing for helical data.
    parker_weighting : Windowing for overcomplete fan-beam data.
    """
    return ray_trafo.adjoint * fbp_filter_op(ray_trafo, padding, filter_type,
                                             frequency_scaling)


if __name__ == '__main__':
    import odl
    import matplotlib.pyplot as plt
    from odl.util.testutils import run_doctests

    # Display the various filters
    x = np.linspace(0, 1, 100)
    cutoff = 0.7

    plt.figure('fbp filter')
    for filter_name in ['Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann',
                        np.sqrt]:
        plt.plot(x, _fbp_filter(x, filter_name, cutoff), label=filter_name)

    plt.title('Filters with frequency scaling = {}'.format(cutoff))
    plt.legend(loc=2)

    # Show the Tam-Danielson window

    # Create Ray Transform in helical geometry
    reco_space = odl.uniform_discr(
        min_pt=[-20, -20, 0], max_pt=[20, 20, 40], shape=[300, 300, 300])
    angle_partition = odl.uniform_partition(0, 8 * 2 * np.pi, 2000)
    detector_partition = odl.uniform_partition([-40, -4], [40, 4], [500, 500])
    geometry = odl.tomo.ConeFlatGeometry(
        angle_partition, detector_partition, src_radius=100, det_radius=100,
        pitch=5.0)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

    # Crete and show TD window
    td_window = tam_danielson_window(ray_trafo, smoothing_width=0)
    td_window.show('Tam-Danielson window', coords=[0, None, None])

    # Show the Parker weighting

    # Create Ray Transform in fan beam geometry
    geometry = odl.tomo.cone_beam_geometry(reco_space,
                                           src_radius=40, det_radius=80)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

    # Crete and show parker weighting
    parker_weighting = parker_weighting(ray_trafo)
    parker_weighting.show('Parker weighting')

    # Also run the doctests
    run_doctests()
