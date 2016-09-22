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

"""Backend for ASTRA using CUDA."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
try:
    import astra
    ASTRA_CUDA_AVAILABLE = astra.astra.use_cuda()
except ImportError:
    ASTRA_CUDA_AVAILABLE = False

from odl.discr import DiscreteLp, DiscreteLpElement
from odl.tomo.backends.astra_setup import (
    astra_projection_geometry, astra_volume_geometry, astra_projector,
    astra_data, astra_algorithm)
from odl.tomo.geometry import (
    Geometry, Parallel2dGeometry, FanFlatGeometry, Parallel3dAxisGeometry,
    HelicalConeFlatGeometry)
from odl.util.utility import writable_array


__all__ = ('astra_cuda_forward_projector', 'astra_cuda_back_projector',
           'ASTRA_CUDA_AVAILABLE')


# TODO: use context manager when creating data structures
# TODO: is magnification scaling at the right place?

def astra_cuda_forward_projector(vol_data, geometry, proj_space, out=None):
    """Run an ASTRA forward projection on the given data using the GPU.

    Parameters
    ----------
    vol_data : `DiscreteLpElement`
        Volume data to which the projector is applied
    geometry : `Geometry`
        Geometry defining the tomographic setup
    proj_space : `DiscreteLp`
        Space to which the calling operator maps
    out : ``proj_space`` element, optional
        Element of the projection space to which the result is written. If
        ``None``, an element in ``proj_space`` is created.

    Returns
    -------
    out : ``proj_space`` element
        Projection data resulting from the application of the projector.
        If ``out`` was provided, the returned object is a reference to it.
    """
    if not isinstance(vol_data, DiscreteLpElement):
        raise TypeError('volume data {!r} is not a DiscreteLpElement '
                        'instance'.format(vol_data))
    if not isinstance(geometry, Geometry):
        raise TypeError('geometry  {!r} is not a Geometry instance'
                        ''.format(geometry))
    if vol_data.ndim != geometry.ndim:
        raise ValueError('dimensions {} of volume data and {} of geometry '
                         'do not match'
                         ''.format(vol_data.ndim, geometry.ndim))
    if not isinstance(proj_space, DiscreteLp):
        raise TypeError('projection space {!r} is not a DiscreteLp '
                        'instance'.format(proj_space))
    if out is not None:
        if not isinstance(out, DiscreteLpElement):
            raise TypeError('`out` {} is neither None nor a '
                            'DiscreteLpElement instance'.format(out))

    ndim = vol_data.ndim

    # Create astra geometries
    vol_geom = astra_volume_geometry(vol_data.space)
    proj_geom = astra_projection_geometry(geometry)

    # Create ASTRA data structures

    # In the case dim == 3, we need to swap axes, so can't perform the FP
    # in-place
    vol_id = astra_data(vol_geom, datatype='volume', data=vol_data,
                        allow_copy=True)

    # Create projector
    proj_id = astra_projector('nearest', vol_geom, proj_geom, ndim,
                              impl='cuda')

    if ndim == 2:
        if out is None:
            out = proj_space.element()

        # Wrap the array in correct dtype etc if needed
        with writable_array(out, dtype='float32', order='C') as arr:
            sino_id = astra_data(proj_geom, datatype='projection', data=arr)

            # Create algorithm
            algo_id = astra_algorithm('forward', ndim, vol_id, sino_id,
                                      proj_id=proj_id, impl='cuda')

            # Run algorithm
            astra.algorithm.run(algo_id)
    elif ndim == 3:
        sino_id = astra_data(proj_geom, datatype='projection',
                             ndim=proj_space.ndim)

        # Create algorithm
        algo_id = astra_algorithm('forward', ndim, vol_id, sino_id,
                                  proj_id=proj_id, impl='cuda')

        # Run algorithm
        astra.algorithm.run(algo_id)

        if out is None:
            out = proj_space.element(np.rollaxis(astra.data3d.get(sino_id),
                                                 0, 3))
        else:
            out[:] = np.rollaxis(astra.data3d.get(sino_id), 0, 3)
    else:
        raise RuntimeError('unknown ndim')

    # Fix inconsistent scaling
    if isinstance(geometry, Parallel2dGeometry):
        # parallel2d scales with pixel stride
        out *= 1 / float(geometry.det_partition.cell_sides[0])

    # Delete ASTRA objects
    astra.algorithm.delete(algo_id)
    if ndim == 2:
        astra.data2d.delete((vol_id, sino_id))
        astra.projector.delete(proj_id)
    else:
        astra.data3d.delete((vol_id, sino_id))
        astra.projector3d.delete(proj_id)

    return out


def astra_cuda_back_projector(proj_data, geometry, reco_space, out=None):
    """Run an ASTRA backward projection on the given data using the GPU.

    Parameters
    ----------
    proj_data : `DiscreteLp` element
        Projection data to which the backward projector is applied
    geometry : `Geometry`
        Geometry defining the tomographic setup
    reco_space : `DiscreteLp`
        Space to which the calling operator maps
    out : ``reco_space`` element, optional
        Element of the reconstruction space to which the result is written.
        If ``None``, an element in ``reco_space`` is created.

    Returns
    -------
    out : ``reco_space`` element
        Reconstruction data resulting from the application of the backward
        projector. If ``out`` was provided, the returned object is a
        reference to it.
        """
    if not isinstance(proj_data, DiscreteLpElement):
        raise TypeError('projection data {!r} is not a DiscreteLpElement '
                        'instance'.format(proj_data))
    if not isinstance(geometry, Geometry):
        raise TypeError('geometry  {!r} is not a Geometry instance'
                        ''.format(geometry))
    if not isinstance(reco_space, DiscreteLp):
        raise TypeError('reconstruction space {!r} is not a DiscreteLp '
                        'instance'.format(reco_space))
    if reco_space.ndim != geometry.ndim:
        raise ValueError('dimensions {} of reconstruction space and {} of '
                         'geometry do not match'.format(reco_space.ndim,
                                                        geometry.ndim))
    if out is not None:
        if not isinstance(out, DiscreteLpElement):
            raise TypeError('`out` {} is neither None nor a '
                            'DiscreteLpElement instance'.format(out))

    ndim = proj_data.ndim

    # Create geometries
    vol_geom = astra_volume_geometry(reco_space)
    proj_geom = astra_projection_geometry(geometry)

    if ndim == 2:
        swapped_proj_data = proj_data
    else:
        swapped_proj_data = np.ascontiguousarray(
            np.rollaxis(proj_data.asarray(), 2, 0))

    sino_id = astra_data(proj_geom, datatype='projection',
                         data=swapped_proj_data, allow_copy=True)

    # Create projector
    proj_id = astra_projector('nearest', vol_geom, proj_geom, ndim,
                              impl='cuda')

    # Reconstruction volume
    if out is None:
        out = reco_space.element()

    # Wrap the array in correct dtype etc if needed
    with writable_array(out, dtype='float32', order='C') as arr:
        vol_id = astra_data(vol_geom, datatype='volume', data=arr,
                            ndim=reco_space.ndim)
        # Create algorithm
        algo_id = astra_algorithm('backward', ndim, vol_id, sino_id,
                                  proj_id=proj_id, impl='cuda')

        # Run algorithm
        astra.algorithm.run(algo_id)

    # Angular integration weighting factor
    # angle interval weight by approximate cell volume
    extent = float(geometry.motion_partition.extent())
    size = float(geometry.motion_partition.size)
    scaling_factor = extent / size

    # Fix inconsistent scaling
    if isinstance(geometry, Parallel2dGeometry):
        # scales with 1 / (cell volume)
        scaling_factor *= float(reco_space.cell_volume)
    elif isinstance(geometry, FanFlatGeometry):
        # scales with 1 / (cell volume)
        scaling_factor *= float(reco_space.cell_volume)
        # magnification correction
        src_radius = geometry.src_radius
        det_radius = geometry.det_radius
        scaling_factor *= ((src_radius + det_radius) / src_radius)
    elif isinstance(geometry, Parallel3dAxisGeometry):
        # scales with voxel stride
        # currently only square voxels are supported
        extent = reco_space.partition.extent()
        shape = np.array(reco_space.shape, dtype=float)
        scaling_factor /= float(extent[0] / shape[0])
    elif isinstance(geometry, HelicalConeFlatGeometry):
        # cone scales with voxel stride
        # currently only square voxels are supported
        extent = reco_space.partition.extent()
        shape = np.array(reco_space.shape, dtype=float)
        scaling_factor /= float(extent[0] / shape[0])
        # magnification correction
        src_radius = geometry.src_radius
        det_radius = geometry.det_radius
        scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2
    out *= scaling_factor

    # Delete ASTRA objects
    astra.algorithm.delete(algo_id)
    if ndim == 2:
        astra.data2d.delete((vol_id, sino_id))
        astra.projector.delete(proj_id)
    else:
        astra.data3d.delete((vol_id, sino_id))
        astra.projector3d.delete(proj_id)

    return out


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
