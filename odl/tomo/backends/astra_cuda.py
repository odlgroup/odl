# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Backend for ASTRA using CUDA."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from pkg_resources import parse_version
try:
    import astra
    ASTRA_CUDA_AVAILABLE = astra.astra.use_cuda()
except ImportError:
    ASTRA_CUDA_AVAILABLE = False

from odl.discr import DiscreteLp
from odl.tomo.backends.astra_setup import (
    ASTRA_VERSION,
    astra_projection_geometry, astra_volume_geometry, astra_projector,
    astra_data, astra_algorithm)
from odl.tomo.geometry import (
    Geometry, Parallel2dGeometry, FanFlatGeometry, Parallel3dAxisGeometry,
    HelicalConeFlatGeometry)


__all__ = ('ASTRA_CUDA_AVAILABLE',
           'AstraCudaProjectorImpl', 'AstraCudaBackProjectorImpl')


class AstraCudaProjectorImpl(object):

    """Thin wrapper around ASTRA."""

    def __init__(self, geometry, reco_space, proj_space):
        """Initialize a new instance.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup.
        reco_space : `DiscreteLp`
            Reconstruction space, the space of the images to be forward
            projected.
        proj_space : ``DiscreteLp``
            Projection space, the space of the result.
        """
        assert isinstance(geometry, Geometry)
        assert isinstance(reco_space, DiscreteLp)
        assert isinstance(proj_space, DiscreteLp)

        self.geometry = geometry
        self.reco_space = reco_space
        self.proj_space = proj_space

        self.create_ids()

    def call_forward(self, vol_data, out=None):
        """Run an ASTRA forward projection on the given data using the GPU.

        Parameters
        ----------
        vol_data : `reco_space` element
            Volume data to which the projector is applied.
        out : `proj_space` element, optional
            Element of the projection space to which the result is written. If
            ``None``, an element in `proj_space` is created.

        Returns
        -------
        out : ``proj_space`` element
            Projection data resulting from the application of the projector.
            If ``out`` was provided, the returned object is a reference to it.
        """
        assert vol_data in self.reco_space
        if out is not None:
            assert out in self.proj_space
        else:
            out = self.proj_space.element()

        # Copy data to GPU memory
        if self.geometry.ndim == 2:
            astra.data2d.store(self.vol_id, vol_data.asarray())
        elif self.geometry.ndim == 3:
            astra.data3d.store(self.vol_id, vol_data.asarray())
        else:
            raise RuntimeError('unknown ndim')

        # Run algorithm
        astra.algorithm.run(self.algo_id)

        # Copy result to host
        if self.geometry.ndim == 2:
            out[:] = self.out_array
        elif self.geometry.ndim == 3:
            out[:] = np.rollaxis(self.out_array, 0, 3).reshape(
                self.proj_space.shape)

        # Fix scaling to weight by pixel size
        if isinstance(self.geometry, Parallel2dGeometry):
            # parallel2d scales with pixel stride
            out *= 1 / float(self.geometry.det_partition.cell_volume)

        return out

    def create_ids(self):
        """Create ASTRA objects."""
        # Create input and output arrays
        if self.geometry.motion_partition.ndim == 1:
            motion_shape = self.geometry.motion_partition.shape
        else:
            # Need to flatten 2- or 3-dimensional angles into one axis
            motion_shape = (np.prod(self.geometry.motion_partition.shape),)

        proj_shape = motion_shape + self.geometry.det_partition.shape
        proj_ndim = len(proj_shape)

        if proj_ndim == 2:
            astra_proj_shape = proj_shape
            astra_vol_shape = self.reco_space.shape
        elif proj_ndim == 3:
            astra_proj_shape = (proj_shape[2], proj_shape[0], proj_shape[1])
            astra_vol_shape = self.reco_space.shape

        self.in_array = np.empty(astra_vol_shape,
                                 dtype='float32', order='C')
        self.out_array = np.empty(astra_proj_shape,
                                  dtype='float32', order='C')

        # Create ASTRA data structures
        vol_geom = astra_volume_geometry(self.reco_space)
        proj_geom = astra_projection_geometry(self.geometry)
        self.vol_id = astra_data(vol_geom,
                                 datatype='volume',
                                 ndim=self.reco_space.ndim,
                                 data=self.in_array,
                                 allow_copy=False)

        self.proj_id = astra_projector('nearest', vol_geom, proj_geom,
                                       ndim=proj_ndim, impl='cuda')

        self.sino_id = astra_data(proj_geom,
                                  datatype='projection',
                                  ndim=proj_ndim,
                                  data=self.out_array,
                                  allow_copy=False)

        # Create algorithm
        self.algo_id = astra_algorithm(
            'forward', proj_ndim, self.vol_id, self.sino_id,
            proj_id=self.proj_id, impl='cuda')

    def __del__(self):
        """Delete ASTRA objects."""
        if self.geometry.ndim == 2:
            adata, aproj = astra.data2d, astra.projector
        else:
            adata, aproj = astra.data3d, astra.projector3d

        if self.algo_id is not None:
            astra.algorithm.delete(self.algo_id)
            self.algo_id = None
        if self.vol_id is not None:
            adata.delete(self.vol_id)
            self.vol_id = None
        if self.sino_id is not None:
            adata.delete(self.sino_id)
            self.sino_id = None
        if self.proj_id is not None:
            aproj.delete(self.proj_id)
            self.proj_id = None

        self.in_array = None
        self.out_array = None


class AstraCudaBackProjectorImpl(object):

    """Thin wrapper around ASTRA."""

    def __init__(self, geometry, reco_space, proj_space):
        """Initialize a new instance.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup.
        reco_space : `DiscreteLp`
            Reconstruction space, the space to which the backprojection maps.
        proj_space : ``DiscreteLp``
            Projection space, the space from which the backprojection maps.
        """
        assert isinstance(geometry, Geometry)
        assert isinstance(reco_space, DiscreteLp)
        assert isinstance(proj_space, DiscreteLp)

        self.geometry = geometry
        self.reco_space = reco_space
        self.proj_space = proj_space
        self.create_ids()

    def call_backward(self, proj_data, out=None):
        """Run an ASTRA back-projection on the given data using the GPU.

        Parameters
        ----------
        proj_data : `proj_space` element
            Projection data to which the back-projector is applied.
        out : `reco_space` element, optional
            Element of the reconstruction space to which the result is written.
            If ``None``, an element in ``reco_space`` is created.

        Returns
        -------
        out : ``reco_space`` element
            Reconstruction data resulting from the application of the
            back-projector. If ``out`` was provided, the returned object is a
            reference to it.
        """
        assert proj_data in self.proj_space
        if out is not None:
            assert out in self.reco_space
        else:
            out = self.reco_space.element()

        # Copy data to GPU memory
        if self.geometry.ndim == 2:
            astra.data2d.store(self.sino_id, proj_data.asarray())
        elif self.geometry.ndim == 3:
            shape = (-1,) + self.geometry.det_partition.shape
            reshaped_proj_data = proj_data.asarray().reshape(shape)
            swapped_proj_data = np.rollaxis(reshaped_proj_data, 2, 0)
            astra.data3d.store(self.sino_id, swapped_proj_data)

        # Run algorithm
        astra.algorithm.run(self.algo_id)

        # Copy result to CPU memory
        out[:] = self.out_array

        # Fix scaling to weight by pixel/voxel size
        out *= astra_cuda_bp_scaling_factor(
            self.proj_space, self.reco_space, self.geometry)

        return out

    def create_ids(self):
        """Create ASTRA objects."""
        if self.geometry.motion_partition.ndim == 1:
            motion_shape = self.geometry.motion_partition.shape
        else:
            # Need to flatten 2- or 3-dimensional angles into one axis
            motion_shape = (np.prod(self.geometry.motion_partition.shape),)

        proj_shape = motion_shape + self.geometry.det_partition.shape
        proj_ndim = len(proj_shape)

        if proj_ndim == 2:
            astra_proj_shape = proj_shape
            astra_vol_shape = self.reco_space.shape
        elif proj_ndim == 3:
            astra_proj_shape = (proj_shape[2], proj_shape[0], proj_shape[1])
            astra_vol_shape = self.reco_space.shape

        self.in_array = np.empty(astra_proj_shape,
                                 dtype='float32', order='C')
        self.out_array = np.empty(astra_vol_shape,
                                  dtype='float32', order='C')

        # Create ASTRA data structures
        vol_geom = astra_volume_geometry(self.reco_space)
        proj_geom = astra_projection_geometry(self.geometry)
        self.sino_id = astra_data(proj_geom,
                                  datatype='projection',
                                  data=self.in_array,
                                  ndim=proj_ndim,
                                  allow_copy=False)

        self.proj_id = astra_projector('nearest', vol_geom, proj_geom,
                                       proj_ndim, impl='cuda')

        self.vol_id = astra_data(vol_geom,
                                 datatype='volume',
                                 data=self.out_array,
                                 ndim=self.reco_space.ndim,
                                 allow_copy=False)

        # Create algorithm
        self.algo_id = astra_algorithm(
            'backward', proj_ndim, self.vol_id, self.sino_id,
            proj_id=self.proj_id, impl='cuda')

    def __del__(self):
        """Delete ASTRA objects."""
        if self.geometry.ndim == 2:
            adata, aproj = astra.data2d, astra.projector
        else:
            adata, aproj = astra.data3d, astra.projector3d

        if self.algo_id is not None:
            astra.algorithm.delete(self.algo_id)
            self.algo_id = None
        if self.vol_id is not None:
            adata.delete(self.vol_id)
            self.vol_id = None
        if self.sino_id is not None:
            adata.delete(self.sino_id)
            self.sino_id = None
        if self.proj_id is not None:
            aproj.delete(self.proj_id)
            self.proj_id = None


def astra_cuda_bp_scaling_factor(proj_space, reco_space, geometry):
    """Volume scaling accounting for differing adjoint definitions.

    ASTRA defines the adjoint operator in terms of a fully discrete
    setting (transposed "projection matrix") without any relation to
    physical dimensions, which makes a re-scaling necessary to
    translate it to spaces with physical dimensions.

    Behavior of ASTRA changes slightly between versions, so we keep
    track of it and adapt the scaling accordingly.
    """
    # Angular integration weighting factor
    # angle interval weight by approximate cell volume
    angle_extent = geometry.motion_partition.extent
    num_angles = geometry.motion_partition.shape
    # TODO: this gives the wrong factor for Parallel3dEulerGeometry with
    # 2 angles
    scaling_factor = (angle_extent / num_angles).prod()

    # Correct in case of non-weighted spaces
    proj_extent = float(proj_space.partition.extent.prod())
    proj_size = float(proj_space.partition.size)
    proj_weighting = proj_extent / proj_size

    scaling_factor *= (proj_space.weighting.const /
                       proj_weighting)
    scaling_factor /= (reco_space.weighting.const /
                       reco_space.cell_volume)

    if parse_version(ASTRA_VERSION) < parse_version('1.8rc1'):
        if isinstance(geometry, Parallel2dGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
        elif isinstance(geometry, FanFlatGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
            # Additional magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius)
        elif isinstance(geometry, Parallel3dAxisGeometry):
            # Scales with voxel stride
            # In 1.7, only cubic voxels are supported
            voxel_stride = reco_space.cell_sides[0]
            scaling_factor /= float(voxel_stride)
        elif isinstance(geometry, HelicalConeFlatGeometry):
            # Scales with 1 / cell_volume
            # In 1.7, only cubic voxels are supported
            voxel_stride = reco_space.cell_sides[0]
            scaling_factor /= float(voxel_stride)
            # Magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2

    else:
        if isinstance(geometry, Parallel2dGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
        elif isinstance(geometry, FanFlatGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
            # Magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius)
        elif isinstance(geometry, Parallel3dAxisGeometry):
            # Scales with cell volume
            # currently only square voxels are supported
            scaling_factor /= reco_space.cell_volume
        elif isinstance(geometry, HelicalConeFlatGeometry):
            # Scales with cell volume
            scaling_factor /= reco_space.cell_volume
            # Magnification correction (scaling = 1 / magnification ** 2)
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2

            # Correction for scaled 1/r^2 factor in ASTRA's density weighting.
            # This compensates for scaled voxels and pixels, as well as a
            # missing factor src_radius ** 2 in the ASTRA BP with
            # density weighting.
            det_px_area = geometry.det_partition.cell_volume
            scaling_factor *= (src_radius ** 2 * det_px_area ** 2 /
                               reco_space.cell_volume ** 2)

        # TODO: add case with new ASTRA release

    return scaling_factor


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
