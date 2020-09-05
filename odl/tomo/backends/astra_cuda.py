# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Backend for ASTRA using CUDA."""

from __future__ import absolute_import, division, print_function

import warnings
from multiprocessing import Lock

import numpy as np
from packaging.version import parse as parse_version

from odl.discr import DiscretizedSpace
from odl.tomo.backends.astra_setup import (
    ASTRA_VERSION, astra_algorithm, astra_data, astra_projection_geometry,
    astra_projector, astra_supports, astra_versions_supporting,
    astra_volume_geometry)
from odl.tomo.backends.util import _add_default_complex_impl
from odl.tomo.geometry import (
    ConeBeamGeometry, FanBeamGeometry, Geometry, Parallel2dGeometry,
    Parallel3dAxisGeometry)

try:
    import astra

    ASTRA_CUDA_AVAILABLE = astra.astra.use_cuda()
except ImportError:
    ASTRA_CUDA_AVAILABLE = False

__all__ = (
    'ASTRA_CUDA_AVAILABLE',
)


class AstraCudaImpl:
    """`RayTransform` implementation for CUDA algorithms in ASTRA."""

    algo_forward_id = None
    algo_backward_id = None
    vol_id = None
    sino_id = None
    proj_id = None

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
        if not isinstance(proj_space, DiscretizedSpace):
            raise TypeError(
                '`proj_space` must be a `DiscretizedSpace` instance, got {!r}'
                ''.format(proj_space)
            )

        # Print a warning if the detector midpoint normal vector at any
        # angle is perpendicular to the geometry axis in parallel 3d
        # single-axis geometry -- this is broken in some ASTRA versions
        if (
            isinstance(geometry, Parallel3dAxisGeometry)
            and not astra_supports('par3d_det_mid_pt_perp_to_axis')
        ):
            req_ver = astra_versions_supporting(
                'par3d_det_mid_pt_perp_to_axis'
            )
            axis = geometry.axis
            mid_pt = geometry.det_params.mid_pt
            for i, angle in enumerate(geometry.angles):
                if abs(
                    np.dot(axis, geometry.det_to_src(angle, mid_pt))
                ) < 1e-4:
                    warnings.warn(
                        'angle {}: detector midpoint normal {} is '
                        'perpendicular to the geometry axis {} in '
                        '`Parallel3dAxisGeometry`; this is broken in '
                        'ASTRA {}, please upgrade to ASTRA {}'
                        ''.format(i, geometry.det_to_src(angle, mid_pt),
                                  axis, ASTRA_VERSION, req_ver),
                        RuntimeWarning)
                    break

        self.geometry = geometry
        self._vol_space = vol_space
        self._proj_space = proj_space

        self.create_ids()

        # ASTRA projectors are not thread-safe, thus we need to lock manually
        self._mutex = Lock()

    @property
    def vol_space(self):
        return self._vol_space

    @property
    def proj_space(self):
        return self._proj_space

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
            astra_vol_shape = self.vol_space.shape
        elif proj_ndim == 3:
            # The `u` and `v` axes of the projection data are swapped,
            # see explanation in `astra_*_3d_geom_to_vec`.
            astra_proj_shape = (proj_shape[1], proj_shape[0], proj_shape[2])
            astra_vol_shape = self.vol_space.shape

        self.vol_array = np.empty(astra_vol_shape, dtype='float32', order='C')
        self.proj_array = np.empty(astra_proj_shape, dtype='float32',
                                   order='C')

        # Create ASTRA data structures
        vol_geom = astra_volume_geometry(self.vol_space)
        proj_geom = astra_projection_geometry(self.geometry)
        self.vol_id = astra_data(
            vol_geom,
            datatype='volume',
            ndim=self.vol_space.ndim,
            data=self.vol_array,
            allow_copy=False,
        )

        proj_type = 'cuda' if proj_ndim == 2 else 'cuda3d'
        self.proj_id = astra_projector(
            proj_type, vol_geom, proj_geom, proj_ndim
        )

        self.sino_id = astra_data(
            proj_geom,
            datatype='projection',
            ndim=proj_ndim,
            data=self.proj_array,
            allow_copy=False,
        )

        # Create algorithm
        self.algo_forward_id = astra_algorithm(
            'forward',
            proj_ndim,
            self.vol_id,
            self.sino_id,
            self.proj_id,
            impl='cuda',
        )

        # Create algorithm
        self.algo_backward_id = astra_algorithm(
            'backward',
            proj_ndim,
            self.vol_id,
            self.sino_id,
            self.proj_id,
            impl='cuda',
        )

    @_add_default_complex_impl
    def call_forward(self, x, out=None, **kwargs):
        return self._call_forward_real(x, out, **kwargs)

    def _call_forward_real(self, vol_data, out=None, **kwargs):
        """Run an ASTRA forward projection on the given data using the GPU.

        Parameters
        ----------
        vol_data : ``vol_space.real_space`` element
            Volume data to which the projector is applied. Although
            ``vol_space`` may be complex, this element needs to be real.
        out : ``proj_space`` element, optional
            Element of the projection space to which the result is written. If
            ``None``, an element in `proj_space` is created.

        Returns
        -------
        out : ``proj_space`` element
            Projection data resulting from the application of the projector.
            If ``out`` was provided, the returned object is a reference to it.
        """
        with self._mutex:
            assert vol_data in self.vol_space.real_space

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
            astra.algorithm.run(self.algo_forward_id)

            # Copy result to host
            if self.geometry.ndim == 2:
                out[:] = self.proj_array
            elif self.geometry.ndim == 3:
                out[:] = np.swapaxes(self.proj_array, 0, 1).reshape(
                    self.proj_space.shape)

            # Fix scaling to weight by pixel size
            if (
                isinstance(self.geometry, Parallel2dGeometry)
                and parse_version(ASTRA_VERSION) < parse_version('1.9.9.dev')
            ):
                # parallel2d scales with pixel stride
                out *= 1 / float(self.geometry.det_partition.cell_volume)

            return out

    @_add_default_complex_impl
    def call_backward(self, x, out=None, **kwargs):
        return self._call_backward_real(x, out, **kwargs)

    def _call_backward_real(self, proj_data, out=None, **kwargs):
        """Run an ASTRA back-projection on the given data using the GPU.

        Parameters
        ----------
        proj_data : ``proj_space.real_space`` element
            Projection data to which the back-projector is applied. Although
            ``proj_space`` may be complex, this element needs to be real.
        out : ``vol_space`` element, optional
            Element of the reconstruction space to which the result is written.
            If ``None``, an element in ``vol_space`` is created.

        Returns
        -------
        out : ``vol_space`` element
            Reconstruction data resulting from the application of the
            back-projector. If ``out`` was provided, the returned object is a
            reference to it.
        """
        with self._mutex:
            assert proj_data in self.proj_space.real_space

            if out is not None:
                assert out in self.vol_space
            else:
                out = self.vol_space.element()

            # Copy data to GPU memory
            if self.geometry.ndim == 2:
                astra.data2d.store(self.sino_id, proj_data.asarray())
            elif self.geometry.ndim == 3:
                shape = (-1,) + self.geometry.det_partition.shape
                reshaped_proj_data = proj_data.asarray().reshape(shape)
                swapped_proj_data = np.ascontiguousarray(
                    np.swapaxes(reshaped_proj_data, 0, 1)
                )
                astra.data3d.store(self.sino_id, swapped_proj_data)

            # Run algorithm
            astra.algorithm.run(self.algo_backward_id)

            # Copy result to CPU memory
            out[:] = self.vol_array

            # Fix scaling to weight by pixel/voxel size
            out *= astra_cuda_bp_scaling_factor(
                self.proj_space, self.vol_space, self.geometry
            )

            return out

    def __del__(self):
        """Delete ASTRA objects."""
        if self.geometry.ndim == 2:
            adata, aproj = astra.data2d, astra.projector
        else:
            adata, aproj = astra.data3d, astra.projector3d

        if self.algo_forward_id is not None:
            astra.algorithm.delete(self.algo_forward_id)
            self.algo_forward_id = None
        if self.algo_backward_id is not None:
            astra.algorithm.delete(self.algo_backward_id)
            self.algo_backward_id = None
        if self.vol_id is not None:
            adata.delete(self.vol_id)
            self.vol_id = None
        if self.sino_id is not None:
            adata.delete(self.sino_id)
            self.sino_id = None
        if self.proj_id is not None:
            aproj.delete(self.proj_id)
            self.proj_id = None


def astra_cuda_bp_scaling_factor(proj_space, vol_space, geometry):
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

    scaling_factor *= (
        proj_space.weighting.const / proj_weighting
    )
    scaling_factor /= (
        vol_space.weighting.const / vol_space.cell_volume
    )

    if parse_version(ASTRA_VERSION) < parse_version('1.8rc1'):
        # Scaling for the old, pre-1.8 behaviour
        if isinstance(geometry, Parallel2dGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(vol_space.cell_volume)
        elif (isinstance(geometry, FanBeamGeometry)
              and geometry.det_curvature_radius is None):
            # Scales with 1 / cell_volume
            scaling_factor *= float(vol_space.cell_volume)
            # Additional magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius)
        elif isinstance(geometry, Parallel3dAxisGeometry):
            # Scales with voxel stride
            # In 1.7, only cubic voxels are supported
            voxel_stride = vol_space.cell_sides[0]
            scaling_factor /= float(voxel_stride)
        elif (isinstance(geometry, ConeBeamGeometry)
              and geometry.det_curvature_radius is None):
            # Scales with 1 / cell_volume
            # In 1.7, only cubic voxels are supported
            voxel_stride = vol_space.cell_sides[0]
            scaling_factor /= float(voxel_stride)
            # Magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2
    elif parse_version(ASTRA_VERSION) < parse_version('1.9.0dev'):
        # Scaling for the 1.8.x releases
        if isinstance(geometry, Parallel2dGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(vol_space.cell_volume)
        elif (isinstance(geometry, FanBeamGeometry)
              and geometry.det_curvature_radius is None):
            # Scales with 1 / cell_volume
            scaling_factor *= float(vol_space.cell_volume)
            # Magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius)
        elif isinstance(geometry, Parallel3dAxisGeometry):
            # Scales with cell volume
            # currently only square voxels are supported
            scaling_factor /= vol_space.cell_volume
        elif (isinstance(geometry, ConeBeamGeometry)
              and geometry.det_curvature_radius is None):
            # Scales with cell volume
            scaling_factor /= vol_space.cell_volume
            # Magnification correction (scaling = 1 / magnification ** 2)
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2

            # Correction for scaled 1/r^2 factor in ASTRA's density weighting.
            # This compensates for scaled voxels and pixels, as well as a
            # missing factor src_radius ** 2 in the ASTRA BP with
            # density weighting.
            det_px_area = geometry.det_partition.cell_volume
            scaling_factor *= (
                src_radius ** 2 * det_px_area ** 2 / vol_space.cell_volume ** 2
            )
    elif parse_version(ASTRA_VERSION) < parse_version('1.9.9.dev'):
        # Scaling for intermediate dev releases between 1.8.3 and 1.9.9.dev
        if isinstance(geometry, Parallel2dGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(vol_space.cell_volume)
        elif (isinstance(geometry, FanBeamGeometry)
              and geometry.det_curvature_radius is None):
            # Scales with 1 / cell_volume
            scaling_factor *= float(vol_space.cell_volume)
            # Magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius)
        elif isinstance(geometry, Parallel3dAxisGeometry):
            # Scales with cell volume
            # currently only square voxels are supported
            scaling_factor /= vol_space.cell_volume
        elif (isinstance(geometry, ConeBeamGeometry)
              and geometry.det_curvature_radius is None):
            # Scales with cell volume
            scaling_factor /= vol_space.cell_volume
            # Magnification correction (scaling = 1 / magnification ** 2)
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2

            # Correction for scaled 1/r^2 factor in ASTRA's density weighting.
            # This compensates for scaled voxels and pixels, as well as a
            # missing factor src_radius ** 2 in the ASTRA BP with
            # density weighting.
            det_px_area = geometry.det_partition.cell_volume
            scaling_factor *= (src_radius ** 2 * det_px_area ** 2)
    else:
        # Scaling for versions since 1.9.9.dev
        scaling_factor /= float(vol_space.cell_volume)
        scaling_factor *= float(geometry.det_partition.cell_volume)

    return scaling_factor


if __name__ == '__main__':
    from odl.util.testutils import run_doctests

    run_doctests()
