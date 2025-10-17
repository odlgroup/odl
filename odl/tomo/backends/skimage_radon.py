# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Radon transform (ray transform) in 2d using skimage.transform."""

from __future__ import division

import warnings

import numpy as np

from odl.core.discr import (
    DiscretizedSpace, uniform_discr_frompartition, uniform_partition)
from odl.core.discr.discr_utils import linear_interpolator, point_collocation
from odl.tomo.backends.util import _add_default_complex_impl
from odl.tomo.geometry import Geometry, Parallel2dGeometry
from odl.core.util.utility import writable_array

try:
    import skimage

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

__all__ = (
    'SKIMAGE_AVAILABLE',
    'skimage_radon_forward_projector',
    'skimage_radon_back_projector',
)


def skimage_proj_space(geometry, volume_space, proj_space):
    """Create a projection space adapted to the skimage radon geometry."""
    padded_size = int(np.ceil(volume_space.shape[0] * np.sqrt(2)))
    det_width = volume_space.domain.extent[0] * np.sqrt(2)
    det_part = uniform_partition(-det_width / 2, det_width / 2, padded_size)

    part = geometry.motion_partition.insert(1, det_part)
    space = uniform_discr_frompartition(part, dtype=proj_space.dtype)
    return space


def clamped_interpolation(skimage_range, sinogram):
    """Return interpolator that clamps points to min/max of the space."""
    min_x = skimage_range.domain.min()[1]
    max_x = skimage_range.domain.max()[1]

    def _interpolator(x, out=None):
        x = (x[0], np.clip(x[1], min_x, max_x))
        interpolator = linear_interpolator(
            sinogram, skimage_range.grid.coord_vectors
        )
        return interpolator(x, out=out)

    return _interpolator


def skimage_radon_forward_projector(volume, geometry, proj_space, out=None):
    """Calculate forward projection using skimage.

    Parameters
    ----------
    volume : `DiscretizedSpaceElement`
        The volume to project.
    geometry : `Geometry`
        The projection geometry to use.
    proj_space : `DiscretizedSpace`
        Space in which the projections (sinograms) live.
    out : ``proj_space`` element, optional
        Element to which the result should be written.

    Returns
    -------
    sinogram : ``proj_space`` element
        Result of the forward projection. If ``out`` was given, the returned
        object is a reference to it.
    """
    # Lazy import due to significant import time
    from skimage.transform import radon

    # Check basic requirements. Fully checking should be in wrapper
    assert volume.shape[0] == volume.shape[1]

    theta = np.degrees(geometry.angles)
    skimage_range = skimage_proj_space(geometry, volume.space, proj_space)

    # Rotate volume from (x, y) to (rows, cols), then project
    sino_arr = radon(
        np.rot90(volume.asarray(), 1), theta=theta, circle=False
    )
    sinogram = skimage_range.element(sino_arr.T)

    if out is None:
        out = proj_space.element()

    with writable_array(out) as out_arr:
        point_collocation(
            clamped_interpolation(skimage_range, sinogram),
            proj_space.grid.meshgrid,
            out=out_arr,
        )

    scale = volume.space.cell_sides[0]
    out *= scale

    return out


def skimage_radon_back_projector(sinogram, geometry, vol_space, out=None):
    """Calculate forward projection using skimage.

    Parameters
    ----------
    sinogram : `DiscretizedSpaceElement`
        Sinogram (projections) to backproject.
    geometry : `Geometry`
        The projection geometry to use.
    vol_space : `DiscretizedSpace`
        Space in which reconstructed volumes live.
    out : ``vol_space`` element, optional
        An element to which the result should be written.

    Returns
    -------
    volume : ``vol_space`` element
        Result of the back-projection. If ``out`` was given, the returned
        object is a reference to it.
    """
    # Lazy import due to significant import time
    from skimage.transform import iradon

    theta = np.degrees(geometry.angles)
    skimage_range = skimage_proj_space(geometry, vol_space, sinogram.space)

    skimage_sinogram = skimage_range.element()
    with writable_array(skimage_sinogram) as sino_arr:
        point_collocation(
            clamped_interpolation(sinogram.space, sinogram),
            skimage_range.grid.meshgrid,
            out=sino_arr,
        )

    if out is None:
        out = vol_space.element()
    else:
        # Only do asserts here since these are backend functions
        assert out in vol_space

    # scikit-image changed the name of this parameter in version 0.17
    if (skimage.__version__ < '0.17'):
        filter_disable = {"filter": None}
    else:
        filter_disable = {"filter_name": None}

    # Rotate back from (rows, cols) to (x, y), then back-project (no filter)
    backproj = iradon(
        skimage_sinogram.asarray().T,
        theta,
        output_size=vol_space.shape[0],
        circle=False,
        **filter_disable
    )
    out[:] = np.rot90(backproj, -1)

    # Empirically determined value, gives correct scaling
    scaling_factor = 4 * geometry.motion_params.length / (2 * np.pi)

    # Correct in case of non-weighted spaces
    proj_volume = np.prod(sinogram.space.partition.extent)
    proj_size = sinogram.space.partition.size
    proj_weighting = proj_volume / proj_size

    scaling_factor *= sinogram.space.weighting.const / proj_weighting
    scaling_factor /= vol_space.weighting.const / vol_space.cell_volume

    # Correctly scale the output
    out *= scaling_factor

    return out


class SkImageImpl:
    """Scikit-image backend of the `RayTransform` operator."""

    def __init__(self, geometry, vol_space, proj_space):
        """Initialize a new instance.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup.
        vol_space : `DiscretizedSpace`
            Reconstruction space, the space of the images to be forward
            projected.
        proj_space : `DiscretizedSpace`
            Projection space, the space of the result.
        """
        if not isinstance(geometry, Geometry):
            raise TypeError(
                '`geometry` must be a `Geometry` instance, got {!r}'
                ''.format(geometry)
            )
        if not isinstance(vol_space, DiscretizedSpace):
            raise TypeError(
                '`vol_space` must be a `DiscretizedSpace` instance, got {!r}'
                ''.format(vol_space)
            )
        if vol_space.impl != 'numpy':
            raise TypeError(
                '`vol_space` implementation must be `numpy`, got {!r}'
                ''.format(vol_space.impl)
            )
        if not isinstance(proj_space, DiscretizedSpace):
            raise TypeError(
                '`proj_space` must be a `DiscretizedSpace` instance, got {!r}'
                ''.format(proj_space)
            )
        if proj_space.impl != 'numpy':
            raise TypeError(
                '`proj_space` implementation must be `numpy`, got {!r}'
                ''.format(proj_space.impl)
            )
        if not isinstance(geometry, Parallel2dGeometry):
            raise TypeError(
                "{!r} backend only supports 2d parallel geometries"
                ''.format(self.__class__.__name__)
            )
        mid_pt = vol_space.domain.mid_pt
        if not np.allclose(mid_pt, [0, 0]):
            raise ValueError(
                'reconstruction space must be centered at (0, 0), '
                'got midpoint {}'.format(mid_pt)
            )
        shape = vol_space.shape
        if shape[0] != shape[1]:
            raise ValueError(
                '`vol_space.shape` must have equal entries, got {}'
                ''.format(shape)
            )
        extent = vol_space.domain.extent
        if extent[0] != extent[1]:
            raise ValueError(
                '`vol_space.extent` must have equal entries, got {}'
                ''.format(extent)
            )

        if vol_space.size >= 256 ** 2:
            warnings.warn(
                "The 'skimage' backend may be too slow for volumes of this "
                "size. Consider using 'astra_cpu', or 'astra_cuda' if your "
                "machine has an Nvidia GPU.",
                RuntimeWarning,
            )

        self.geometry = geometry
        self._vol_space = vol_space
        self._proj_space = proj_space

    @property
    def vol_space(self):
        return self._vol_space

    @property
    def proj_space(self):
        return self._proj_space

    @_add_default_complex_impl
    def call_forward(self, x, out, **kwargs):
        return skimage_radon_forward_projector(
            x, self.geometry, self.proj_space.real_space, out
        )

    @_add_default_complex_impl
    def call_backward(self, x, out, **kwargs):
        return skimage_radon_back_projector(
            x, self.geometry, self.vol_space.real_space, out
        )


if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests

    run_doctests()
