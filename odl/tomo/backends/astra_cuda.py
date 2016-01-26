# Copyright 2014, 2015 The ODL development group
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

"""Backend for ASTRA using CUDA"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np

try:
    import astra
    if astra.astra.use_cuda():
        ASTRA_CUDA_AVAILABLE = True
    else:
        ASTRA_CUDA_AVAILABLE = False
except ImportError:
    ASTRA_CUDA_AVAILABLE = False

# Internal
from odl.discr.lp_discr import DiscreteLp, DiscreteLpVector
from odl.tomo.backends.astra_setup import (astra_projection_geometry,
                                           astra_volume_geometry,
                                           astra_projector, astra_data,
                                           astra_algorithm)
from odl.tomo.geometry import Geometry
from odl.tomo.geometry import (Parallel2dGeometry, FanFlatGeometry,
                               Parallel3dGeometry, HelicalConeFlatGeometry)

__all__ = ('astra_cuda_forward_projector_call',
           'astra_cuda_backward_projector_call',
           'ASTRA_CUDA_AVAILABLE')


# TODO: use context manager when creating data structures

def astra_cuda_forward_projector_call(vol_data, geometry, proj_space,
                                      out=None):
    """Run an ASTRA forward projection on the given data using the GPU.

    Parameters
    ----------
    vol_data : `DiscreteLpVector`
        Volume data to which the projector is applied
    geometry : `Geometry`
        Geometry defining the tomographic setup
    proj_space : `DiscreteLp`
        Space to which the calling operator maps
    out : `DiscreteLpVector`, optional
        Vector in the projection space to which the result is written. If
        `None` creates an element in the projection space ``proj_space``

    Returns
    -------
    out : ``proj_space`` element
        Projection data resulting from the application of the projector
    """
    if not isinstance(vol_data, DiscreteLpVector):
        raise TypeError('volume data {!r} is not a `DiscreteLp.Vector` '
                        'instance.'.format(vol_data))
    if not isinstance(geometry, Geometry):
        raise TypeError('geometry  {!r} is not a `Geometry` instance.'
                        ''.format(geometry))
    if vol_data.ndim != geometry.ndim:
        raise ValueError('dimensions {} of volume data and {} of geometry '
                         'do not match.'
                         ''.format(vol_data.ndim, geometry.ndim))
    if not isinstance(proj_space, DiscreteLp):
        raise TypeError('projection space {!r} is not a `DiscreteLp` '
                        'instance.'.format(proj_space))
    if out is not None:
        if not isinstance(out, DiscreteLpVector):
            raise TypeError('out {} is neither `None` nor a '
                            '`DiscreteLpVector` instance'.format(out))

    ndim = vol_data.ndim

    # Create astra geometries
    vol_geom = astra_volume_geometry(vol_data.space)
    proj_geom = astra_projection_geometry(geometry)

    # Create ASTRA data structures

    # In the case dim == 3, we need to swap axes, so can't perform the FP
    # in-place
    if out is None and ndim == 2:
        out = proj_space.element()

    vol_id = astra_data(vol_geom, datatype='volume', data=vol_data)
    sino_id = astra_data(proj_geom, datatype='projection', data=out,
                         ndim=proj_space.ndim)

    # Create projector
    proj_id = astra_projector('nearest', vol_geom, proj_geom, ndim,
                              impl='cuda')

    # Create algorithm
    algo_id = astra_algorithm('forward', ndim, vol_id, sino_id,
                              proj_id=proj_id, impl='cuda')

    # Run algorithm
    astra.algorithm.run(algo_id)

    # Wrap data
    if ndim == 3:
        tmp = proj_space.element(np.rollaxis(astra.data3d.get(sino_id), 0, 3))
        if out is None:
            out = tmp
        else:
            out.assign(tmp)

    # Fix scaling issue
    if isinstance(geometry, Parallel2dGeometry):
        # cuda parallel2d scales linearly with linear pixel stride
        out *= 1/float(geometry.det_grid.stride[0])

    # Delete ASTRA objects
    astra.algorithm.delete(algo_id)
    if ndim == 2:
        astra.data2d.delete((vol_id, sino_id))
        astra.projector.delete(proj_id)
    else:
        astra.data3d.delete((vol_id, sino_id))
        astra.projector3d.delete(proj_id)

    return out


def astra_cuda_backward_projector_call(proj_data, geometry, reco_space,
                                       out=None):
    """Run an ASTRA backward projection on the given data using the GPU.

        Parameters
        ----------
        proj_data : `DiscreteLp` element
            Projection data to which the backward projector is applied
        geometry : `Geometry`
            Geometry defining the tomographic setup
        reco_space : `DiscreteLp`
            Space to which the calling operator maps
        out : `DiscreteLpVector`, optional
            Vector in the reconstruction space to which the result is written.
            If `None` creates an element in the reconstruction space
            ``reco_space``

        Returns
        -------
        out : ``reco_space`` element
            Reconstruction data resulting from the application of the backward
            projector
        """
    if not isinstance(proj_data, DiscreteLpVector):
        raise TypeError('projection data {!r} is not a `DiscreteLpVector` '
                        'instance.'.format(proj_data))
    if not isinstance(geometry, Geometry):
        raise TypeError('geometry  {!r} is not a `Geometry` instance.'
                        ''.format(geometry))
    if not isinstance(reco_space, DiscreteLp):
        raise TypeError('reconstruction space {!r} is not a `DiscreteLp` '
                        'instance.'.format(reco_space))
    if reco_space.ndim != geometry.ndim:
        raise ValueError('dimensions {} of reconstruction space and {} of '
                         'geometry do not match.'.format(reco_space.ndim,
                                                         geometry.ndim))
    if out is not None:
        if not isinstance(out, DiscreteLpVector):
            raise TypeError('out {} is neither `None` nor a '
                            '`DiscreteLpVector` instance'.format(out))

    ndim = proj_data.ndim

    # Create geometries
    vol_geom = astra_volume_geometry(reco_space)
    proj_geom = astra_projection_geometry(geometry)

    # Create data structures
    if out is None:
        out = reco_space.element()

    vol_id = astra_data(vol_geom, datatype='volume', data=out,
                        ndim=reco_space.ndim)

    if ndim == 2:
        swapped_proj_data = proj_data
    else:
        swapped_proj_data = np.ascontiguousarray(
            np.rollaxis(proj_data.asarray(), 2, 0))

    sino_id = astra_data(proj_geom, datatype='projection',
                         data=swapped_proj_data)

    # Create projector
    proj_id = astra_projector('nearest', vol_geom, proj_geom, ndim,
                              impl='cuda')

    # Create algorithm
    algo_id = astra_algorithm('backward', ndim, vol_id, sino_id,
                              proj_id=proj_id, impl='cuda')

    # Run algorithm
    astra.algorithm.run(algo_id)

    # Fix scaling issues
    if isinstance(geometry, (FanFlatGeometry, Parallel2dGeometry)):
        # cuda fanflat and parallel2d scale linearly with cell volume
        out *= float(reco_space.cell_volume)
    elif isinstance(geometry, (HelicalConeFlatGeometry, Parallel3dGeometry)):
        # cuda cone and parallel3d scale linearly with linear voxel size
        out /= float(reco_space.cell_size[0])


    # Delete ASTRA objects
    astra.algorithm.delete(algo_id)
    if ndim == 2:
        astra.data2d.delete((vol_id, sino_id))
        astra.projector.delete(proj_id)
    else:
        astra.data3d.delete((vol_id, sino_id))
        astra.projector3d.delete(proj_id)

    return out
