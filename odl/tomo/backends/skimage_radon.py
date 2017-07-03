# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Radon transform (ray transform) in 2d using skimage.transform."""

from odl.discr import uniform_discr_frompartition, uniform_partition
import numpy as np
try:
    from skimage.transform import radon, iradon
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

__all__ = ('skimage_radon_forward', 'skimage_radon_back_projector',
           'SKIMAGE_AVAILABLE')


def skimage_theta(geometry):
    """Calculate angles in degrees with ODL skimage conventions."""
    return geometry.angles * 180.0 / np.pi


def skimage_sinogram_space(geometry, volume_space, sinogram_space):
    """Create a range adapted to the skimage radon geometry."""
    padded_size = int(np.ceil(volume_space.shape[0] * np.sqrt(2)))
    det_width = volume_space.domain.extent[0] * np.sqrt(2)
    skimage_detector_part = uniform_partition(-det_width / 2.0,
                                              det_width / 2.0,
                                              padded_size)

    skimage_range_part = geometry.motion_partition.insert(
        1, skimage_detector_part)

    skimage_range = uniform_discr_frompartition(skimage_range_part,
                                                interp=sinogram_space.interp,
                                                dtype=sinogram_space.dtype)

    return skimage_range


def clamped_interpolation(skimage_range, sinogram):
    """Interpolate in a possibly smaller space.

    Sets all points that would be outside the domain to match the
    boundary values.
    """
    min_x = skimage_range.domain.min()[1]
    max_x = skimage_range.domain.max()[1]

    def interpolation_wrapper(x):
        x = (x[0], np.maximum(min_x, np.minimum(max_x, x[1])))

        return sinogram.interpolation(x)
    return interpolation_wrapper


def skimage_radon_forward(volume, geometry, range, out=None):
    """Calculate forward projection using skimage.

    Parameters
    ----------
    volume : `DiscreteLpElement`
        The volume to project
    geometry : `Geometry`
        The projection geometry to use
    range : `DiscreteLp`
        range of this projection (sinogram space)
    out : ``range`` element, optional
        An element in range that the result should be written to

    Returns
    -------
    sinogram : ``range`` element
        Sinogram given by the projection.
    """
    # Check basic requirements. Fully checking should be in wrapper
    assert volume.shape[0] == volume.shape[1]

    theta = skimage_theta(geometry)
    skimage_range = skimage_sinogram_space(geometry, volume.space, range)

    # Rotate volume from (x, y) to (rows, cols)
    sino_arr = radon(np.rot90(volume.asarray(), 1),
                     theta=theta, circle=False)
    sinogram = skimage_range.element(sino_arr.T)

    if out is None:
        out = range.element()

    out.sampling(clamped_interpolation(skimage_range, sinogram))

    scale = volume.space.cell_sides[0]

    out *= scale

    return out


def skimage_radon_back_projector(sinogram, geometry, range, out=None):
    """Calculate forward projection using skimage.

    Parameters
    ----------
    sinogram : `DiscreteLpElement`
        Sinogram (projections) to backproject.
    geometry : `Geometry`
        The projection geometry to use.
    range : `DiscreteLp`
        range of this projection (volume space).
    out : ``range`` element, optional
        An element in range that the result should be written to.

    Returns
    -------
    sinogram : ``range`` element
        Sinogram given by the projection.
    """
    theta = skimage_theta(geometry)
    skimage_range = skimage_sinogram_space(geometry, range, sinogram.space)

    skimage_sinogram = skimage_range.element()
    skimage_sinogram.sampling(clamped_interpolation(range, sinogram))

    if out is None:
        out = range.element()
    else:
        # Only do asserts here since these are backend functions
        assert out in range

    # Rotate back from (rows, cols) to (x, y)
    backproj = iradon(skimage_sinogram.asarray().T, theta,
                      output_size=range.shape[0], filter=None, circle=False)
    out[:] = np.rot90(backproj, -1)

    # Empirically determined value, gives correct scaling
    scaling_factor = 4.0 * float(geometry.motion_params.length) / (2 * np.pi)

    # Correct in case of non-weighted spaces
    proj_extent = float(sinogram.space.partition.extent.prod())
    proj_size = float(sinogram.space.partition.size)
    proj_weighting = proj_extent / proj_size

    scaling_factor *= (sinogram.space.weighting.const /
                       proj_weighting)
    scaling_factor /= (range.weighting.const /
                       range.cell_volume)

    # Correctly scale the output
    out *= scaling_factor

    return out
