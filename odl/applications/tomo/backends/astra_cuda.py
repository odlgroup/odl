# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Backend for ASTRA using CUDA."""


import warnings
from multiprocessing import Lock

import numpy as np
from packaging.version import parse as parse_version

from odl.core.discr import DiscretizedSpace
from odl.applications.tomo.backends.astra_setup import (
    ASTRA_VERSION, astra_projection_geometry,
    astra_projector, astra_supports, astra_versions_supporting,
    astra_volume_geometry)
from odl.applications.tomo.backends.util import _add_default_complex_impl
from odl.applications.tomo.geometry import (
    ConeBeamGeometry, FanBeamGeometry, Geometry, Parallel2dGeometry,
    Parallel3dAxisGeometry)
from odl.core.discr.discr_space import DiscretizedSpaceElement
from odl.core.array_API_support import empty, get_array_and_backend

try:
    import astra
    # This is important, although not use explicitely. 
    # If not imported, astra.experimental is not "visible"
    import astra.experimental     
    ASTRA_CUDA_AVAILABLE = astra.astra.use_cuda()
except ImportError:
    ASTRA_CUDA_AVAILABLE = False

__all__ = (
    'ASTRA_CUDA_AVAILABLE',
)
   

def index_of_cuda_device(device: "torch.device"):
    if device == 'cpu':
        return None
    else:
        return int(str(device).split(':')[-1])

class AstraCudaImpl:
    """`RayTransform` implementation for CUDA algorithms in ASTRA."""
    projector_id = None

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
            raise TypeError(f"`geometry` must be a `Geometry` instance, got {geometry}")
        if not isinstance(vol_space, DiscretizedSpace):
            raise TypeError(
                f"`vol_space` must be a `DiscretizedSpace` instance, got {vol_space}"
            )
        if not isinstance(proj_space, DiscretizedSpace):
            raise TypeError(
                f"`proj_space` must be a `DiscretizedSpace` instance, got {proj_space}"
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
                        f"angle {i}: detector midpoint normal {geometry.det_to_src(angle, mid_pt)}"
                        + f" is perpendicular to the geometry axis {axis} in `Parallel3dAxisGeometry`;"
                        + f" this is broken in ASTRA {ASTRA_VERSION}, please upgrade to ASTRA {req_ver}",
                        RuntimeWarning,
                    )
                    break

        self.geometry = geometry
        self._vol_space = vol_space
        self._proj_space = proj_space

        self.create_ids()

        # ASTRA projectors are not thread-safe, thus we need to lock manually
        self._mutex = Lock()
        assert (
            vol_space.impl == proj_space.impl
        ), f"Volume space ({vol_space.impl}) != Projection space ({proj_space.impl})"

        if self.geometry.ndim == 3:
            if vol_space.impl == 'numpy':
                self.transpose_tuple = (1,0,2)
            elif vol_space.impl == 'pytorch':
                self.transpose_tuple = (1,0)
            else:
                raise NotImplementedError('Not implemented for another backend')
            
        self.fp_scaling_factor = astra_cuda_fp_scaling_factor(
            self.geometry
        )
        self.bp_scaling_factor = astra_cuda_bp_scaling_factor(
                self.proj_space, self.vol_space, self.geometry
            )

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
        self.proj_ndim = len(proj_shape)

        # Create ASTRA data structures
        self.vol_geom  = astra_volume_geometry(self.vol_space, 'cuda')
        
        self.proj_geom = astra_projection_geometry(self.geometry, 'cuda')

        self.projector_id = astra_projector(
            astra_proj_type = 'cuda3d', 
            astra_vol_geom  = self.vol_geom, 
            astra_proj_geom = self.proj_geom, 
            ndim = 3, 
            override_2D = bool(self.geometry.ndim == 2)
        )        

    @_add_default_complex_impl
    def call_forward(self, x, out=None, **kwargs):
        return self._call_forward_real(x, out, **kwargs)

    def _call_forward_real(self, vol_data:DiscretizedSpaceElement, out=None, **kwargs):
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
                assert out in self.proj_space.real_space, f"The out argument provided is a {type(out)}, which is not an element of the projection space {self.proj_space.real_space}"
                if self.vol_space.impl == 'pytorch':
                    warnings.warn("You requested an out-of-place transform with PyTorch. This will require cloning the data and will allocate extra memory", RuntimeWarning)
                proj_data = out.data[None] if self.proj_ndim==2 else out.data
                if self.geometry.ndim == 3:
                    proj_data = proj_data.transpose(*self.transpose_tuple)                    

            else:
                proj_data = empty(
                    impl   = self.proj_space.impl,
                    shape  = astra.geom_size(self.proj_geom),
                    dtype  = self.proj_space.dtype,
                    device = self.proj_space.device
                )
                    
            if self.proj_ndim == 2:
                volume_data = vol_data.data[None]
            elif self.proj_ndim == 3:
                volume_data = vol_data.data
            else:
                raise NotImplementedError

            volume_data, vol_backend = get_array_and_backend(volume_data, must_be_contiguous=True)
            proj_data, proj_backend  = get_array_and_backend(proj_data, must_be_contiguous=True)

            if self.proj_space.impl == 'pytorch':
                device_index = index_of_cuda_device(
                                  self.proj_space.tspace.device) #type:ignore
                if device_index is not None:
                    astra.set_gpu_index(device_index)
                    
            astra.experimental.direct_FP3D( #type:ignore
                self.projector_id,
                volume_data,
                proj_data
            )
            
            proj_data *= self.fp_scaling_factor
            proj_data = proj_data[0] if self.geometry.ndim == 2 else proj_data.transpose(*self.transpose_tuple)

            if out is not None:
                out.data[:] = proj_data if self.proj_space.impl == 'numpy' else proj_data.clone()
            else:
                return self.proj_space.element(proj_data)

    @_add_default_complex_impl
    def call_backward(self, x, out=None, **kwargs):
        return self._call_backward_real(x, out, **kwargs)

    def _call_backward_real(self, proj_data:DiscretizedSpaceElement, out=None, **kwargs):
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
                assert out in self.vol_space.real_space, f"The out argument provided is a {type(out)}, which is not an element of the projection space {self.vol_space.real_space}"
                if self.vol_space.impl == 'pytorch':
                    warnings.warn(
                        "You requested an out-of-place transform with PyTorch. \
                        This will require cloning the data and will allocate extra memory", 
                        RuntimeWarning)
                volume_data = out.data[None] if self.geometry.ndim==2 else out.data
            else:
                volume_data = empty(
                    self.vol_space.impl,
                    astra.geom_size(self.vol_geom),
                    dtype  = self.vol_space.dtype,
                    device = self.vol_space.device
                )

            ### Transpose projection tensor            
            if self.proj_ndim == 2:
                proj_data = proj_data.data[None]
            elif self.proj_ndim == 3:                
                proj_data = proj_data.data.transpose(*self.transpose_tuple)
            else:
                raise NotImplementedError
                
            # Ensure data is contiguous otherwise astra will throw an error
            volume_data, vol_backend = get_array_and_backend(volume_data, must_be_contiguous=True)
            proj_data, proj_backend  = get_array_and_backend(proj_data, must_be_contiguous=True)
            
            if self.vol_space.tspace.impl == 'pytorch':
                device_index = index_of_cuda_device(self.vol_space.tspace.device) #type:ignore
                if device_index is not None:
                    astra.set_gpu_index(device_index)

            ### Call the backprojection
            astra.experimental.direct_BP3D( #type:ignore
                self.projector_id,
                volume_data,
                proj_data                
            )
            volume_data *= self.bp_scaling_factor
            volume_data = volume_data[0] if self.geometry.ndim == 2 else volume_data

            if out is not None:
                out[:] = volume_data if self.vol_space.impl == 'numpy' else volume_data.clone()
                return out
            else:
                return self.vol_space.element(volume_data)


def astra_cuda_fp_scaling_factor(geometry):
    """Volume scaling accounting for differing adjoint definitions.

    ASTRA defines the adjoint operator in terms of a fully discrete
    setting (transposed "projection matrix") without any relation to
    physical dimensions, which makes a re-scaling necessary to
    translate it to spaces with physical dimensions.

    Behavior of ASTRA changes slightly between versions, so we keep
    track of it and adapt the scaling accordingly.
    """
    if (
        isinstance(geometry, Parallel2dGeometry)
        and parse_version(ASTRA_VERSION) < parse_version('1.9.9.dev')
    ):
        # parallel2d scales with pixel stride
        return 1 / float(geometry.det_partition.cell_volume)
    
    else:
        return 1

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
    from odl.core.util.testutils import run_doctests

    run_doctests()
