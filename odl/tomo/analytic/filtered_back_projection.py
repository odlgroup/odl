# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import scipy as sp
from odl.discr import ResizingOperator
from odl.trafos import FourierTransform, PYFFTW_AVAILABLE
from odl.tomo.util.utility import perpendicular_vector


__all__ = ('fbp_op', 'tam_danielson_window')


def _axis_in_detector(geometry):
    """A vector in the detector plane that points along the rotation axis."""
    du = geometry.det_init_axes[0]
    dv = geometry.det_init_axes[1]
    axis = geometry.axis
    c = np.array([np.vdot(axis, du), np.vdot(axis, dv)])
    cnorm = np.linalg.norm(c)

    # Check for numerical errors
    assert cnorm != 0

    return c / cnorm


def _rotation_direction_in_detector(geometry):
    """A vector in the detector plane that points in the rotation direction."""
    du = geometry.det_init_axes[0]
    dv = geometry.det_init_axes[1]
    axis = geometry.axis
    det_normal = np.cross(du, dv)
    rot_dir = np.cross(axis, det_normal)
    c = np.array([np.vdot(rot_dir, du), np.vdot(rot_dir, dv)])
    cnorm = np.linalg.norm(c)

    # Check for numerical errors
    assert cnorm != 0

    return c / cnorm


def _fbp_filter(norm_freq, filter_type, filter_cutoff):
    """Create a smoothing filter for FBP.

    Parameters
    ----------
    norm_freq : `array-like`
        Normalized frequencies.
    filter_type : {'Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann'}
        The type of filter to be used.
    filter_cutoff : float
        Relative cutoff frequency for the filter.

    Returns
    -------
    smoothing_filter : `numpy.ndarray`

    References
    ----------
    http://oftankonyv.reak.bme.hu/tiki-index.php?page=Reconstruction
    """
    if not (0 < filter_cutoff <= 1):
        raise ValueError('`filter_cutoff` ({}) not in the interval (0, 1]'
                         ''.format(filter_cutoff))

    indicator = (norm_freq <= filter_cutoff)

    if filter_type == 'Ram-Lak':
        filt = indicator
    elif filter_type == 'Shepp-Logan':
        filt = indicator * np.sinc(norm_freq / (2 * filter_cutoff))
    elif filter_type == 'Cosine':
        filt = indicator * np.cos(norm_freq * np.pi / (2 * filter_cutoff))
    elif filter_type == 'Hamming':
        filt = indicator * (0.54 +
                            0.46 * np.cos(norm_freq * np.pi / (filter_cutoff)))
    elif filter_type == 'Hann':
        filt = indicator * np.cos(norm_freq * np.pi / (2 * filter_cutoff)) ** 2
    else:
        raise ValueError('unknown `filter_type` ({})'
                         ''.format(filter_type))

    return filt


def tam_danielson_window(ray_trafo, smoothing_width=0.05):
    """Create Tam-Danielson window from a `RayTransform`.

    The Tam-Danielson window is an indicator function on the minimal set of
    data needed to reconstruct a given data. It is useful in analytic
    reconstruction methods such as FBP to give a more accurate reconstruction.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform that the window should be computed for.
    smoothing_width : float
        Relative width of the smoothing applied to the windows edges.

    Returns
    -------
    tam_danielson_window : ``ray_trafo.range`` element

    See Also
    --------
    fbp_op : Filtered back-projection from `RayTransform`
    HelicalConeFlatGeometry : The geometry this is most useful for.
    """

    # Extract parameters
    src_radius = ray_trafo.geometry.src_radius
    det_radius = ray_trafo.geometry.det_radius
    pitch = ray_trafo.geometry.pitch
    dx = ray_trafo.range.meshgrid[1].ravel()

    # Find the direction that the filter should be taken in
    axis_proj = _axis_in_detector(ray_trafo.geometry)

    # Compute angles
    phi = np.arctan(dx / (src_radius + det_radius))
    theta = phi * 2

    # Compute lower and upper bound
    source_to_line_distance = src_radius + src_radius * np.cos(theta)
    source_to_line_lower = pitch * (theta - np.pi) / (2 * np.pi)
    source_to_line_upper = pitch * (theta + np.pi) / (2 * np.pi)

    scale = (src_radius + det_radius) / source_to_line_distance

    lower_proj = source_to_line_lower * scale
    upper_proj = source_to_line_upper * scale

    # Compute a smoothed width
    interval = (upper_proj - lower_proj)
    width = interval * smoothing_width / np.sqrt(2)

    # Append axes
    interval = interval[None, :, None]
    lower_proj = lower_proj[None, :, None]
    upper_proj = upper_proj[None, :, None]
    width = width[None, :, None]

    # Create window function
    def window_fcn(x):
        x_along_axis = axis_proj[0] * x[1] + axis_proj[1] * x[2]
        lower_wndw = 0.5 * (
            1 + sp.special.erf((x_along_axis - lower_proj) / width))
        upper_wndw = 0.5 * (
            1 + sp.special.erf((upper_proj - x_along_axis) / width))

        return lower_wndw * upper_wndw

    return ray_trafo.range.element(window_fcn)


def fbp_op(ray_trafo, padding=True, filter_type='Ram-Lak', filter_cutoff=1.0):
    """Create filtered back-projection from a `RayTransform`.

    The filtered back-projection is an approximate inverse to the ray
    transform.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform (forward operator) whose approximate inverse should
        be computed. Its geometry has to be any of the following

        `Parallel2DGeometry` : Exact reconstruction

        `Parallel3dAxisGeometry` : Exact reconstruction

        `FanFlatGeometry` : Approximate reconstruction, correct in limit of fan
        angle = 0.

        `CircularConeFlatGeometry` : Approximate reconstruction, correct in
        limit of fan angle = 0 and cone angle = 0.

        `HelicalConeFlatGeometry` : Very approximate unless a
        `tam_danielson_window` is used. Accurate with the window.

        Other geometries: Not supported

    padding : bool, optional
        If the data space should be zero padded. Without padding, the data may
        be corrupted due to the circular convolution used. Using padding makes
        the algorithm slower.
    filter_type : string, optional
        The type of filter to be used. The options are, approximate order from
        most noise senstive to least noise sensitive: 'Ram-Lak', 'Shepp-Logan',
        'Cosine', 'Hamming' and 'Hann'.
    filter_cutoff : float, optional
        Relative cutoff frequency for the filter, a scalar in the range (0, 1].
        The normalized frequencies are rescaled so that they fit into the range
        [0, filter_cutoff].

    Returns
    -------
    fbp_op : `Operator`
        Approximate inverse operator of ``ray_trafo``.

    See Also
    --------
    tam_danielson_window : Windowing for helical data
    """
    impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'
    alen = ray_trafo.geometry.motion_params.length

    if ray_trafo.domain.ndim == 2:
        # Define ramp filter
        def fft_filter(x):
            abs_freq = np.abs(x[1])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(norm_freq, filter_type, filter_cutoff)
            scaling = 1 / (2 * alen)
            return filt * abs_freq * scaling

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
            scale = (ray_trafo.geometry.src_radius /
                     (ray_trafo.geometry.src_radius +
                      ray_trafo.geometry.det_radius))

            if ray_trafo.geometry.pitch != 0:
                # In helical each projection hits the detector less than once.
                scale *= alen / (np.pi)
        else:
            scale = 1.0

        # Define ramp filter
        def fft_filter(x):
            abs_freq = np.abs(rot_dir[0] * x[1] + rot_dir[1] * x[2])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(norm_freq, filter_type, filter_cutoff)
            scaling = scale / (2 * alen)
            return filt * abs_freq * scaling

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
    ramp_function = fourier.range.element(fft_filter)

    # Create ramp filter via the convolution formula with fourier transforms
    ramp_filter = fourier.inverse * ramp_function * fourier

    # Create filtered backprojection by composing the backprojection
    # (adjoint) with the ramp filter.
    return ray_trafo.adjoint * ramp_filter


if __name__ == '__main__':
    # Display the various filters
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 100)
    cutoff = 0.7

    for filter_name in ['Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann']:
        plt.plot(x, x * _fbp_filter(x, filter_name, cutoff), label=filter_name)

    plt.title('Filters with cutoff = {}'.format(cutoff))
    plt.legend(loc=2)

    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
