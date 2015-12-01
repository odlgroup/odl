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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import astra
if not astra.astra.use_cuda():
    raise ImportError

# Internal
from odl.discr.lp_discr import DiscreteLp, DiscreteLpVector
from odl.tomo.backends.astra_setup import (astra_projection_geometry,
    astra_volume_geometry, astra_data, astra_algorithm, astra_cleanup)
from odl.tomo.geometry.geometry import Geometry


__all__ = ('astra_gpu_forward_projector_call',
           'astra_gpu_forward_projector_apply',
           'astra_gpu_backward_projector_call',
           'astra_gpu_backward_projector_apply',)


# TODO: rename gpu to cuda?
def astra_gpu_forward_projector_call(vol_data, geometry, proj_space):
    """Run an ASTRA forward projection on the given data using the GPU.

    Parameters
    ----------
    vol_data : `DiscreteLpVector`
        Volume data to which the projector is applied
    geometry : `Geometry`
        Geometry defining the tomographic setup
    proj_space : `DiscreteLp`
        Space to which the calling operator maps

    Returns
    -------
    projection : proj_space element
        Projection data resulting from the application of the projector
    """
    if not isinstance(vol_data, DiscreteLpVector):
        raise TypeError('volume data {!r} is not a `DiscreteLp.Vector` '
                        'instance.'.format(vol_data))
    if not isinstance(geometry, Geometry):
        raise TypeError('geometry  {!r} is not a `Geometry` instance.'
                        ''.format(geometry))
    if not isinstance(proj_space, DiscreteLp):
        raise TypeError('projection space {!r} is not a `DiscreteLp` '
                        'instance.'.format(proj_space))

    if vol_data.space.grid.ndim != geometry.ndim:
        raise ValueError('dimensions {} of volume data and {} of geometry '
                         'do not match.'
                         ''.format(vol_data.space.grid.ndim, geometry.ndim))

    ndim = vol_data.space.grid.ndim

    # Create astra geometries
    vol_geom = astra_volume_geometry(vol_data.space)
    proj_geom = astra_projection_geometry(geometry, vol_data.space)
    # print(proj_geom)

    # Create ASTRA data structures
    vol_id = astra_data(vol_geom, datatype='volume', data=vol_data)
    sino_id = astra_data(proj_geom, datatype='projection', data=None,
                         ndim=proj_space.grid.ndim)

    # Create algorithm
    algo_id = astra_algorithm('forward', ndim, vol_id, sino_id, proj_id=None,
                              impl='cuda')

    # Run algorithm and delete it
    astra.algorithm.run(algo_id)

    # TODO: check if axes have to be swapped
    # Wrap data
    if ndim == 2:
        elem = proj_space.element(astra.data2d.get_shared(sino_id))
    else:  # ndim = 3
        elem = proj_space.element(
            astra.data3d.get_shared(sino_id).swapaxes(0,1))

    # Delete ASTRA objects
    astra_cleanup()

    return elem


def astra_gpu_forward_projector_apply(vol_data, geometry, proj_data):
    """Run an ASTRA forward projection on the given data using the GPU.

    Parameters
    ----------
    vol_data : `DiscreteLpVector`
        Volume data to which the projector is applied
    geometry : `Geometry`
        Geometry defining the tomographic setup
    proj_data : `DiscreteLpVector`
        Projection space element to which the projection data is written
    """

def astra_gpu_backward_projector_call(proj_data, geometry, reco_space):
    """Run an ASTRA backward projection on the given data using the GPU.

        Parameters
        ----------
        proj_data : `DiscreteLp` element
            Projection data to which the backward projector is applied
        geometry : `Geometry`
            Geometry defining the tomographic setup
        reco_space : `DiscreteLp`
            Space to which the calling operator maps

        Returns
        -------
        reconstruction : reco_space element
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

    if reco_space.grid.ndim != geometry.ndim:
        raise ValueError('dimensions {} of reconstruction space and {} of '
                         'geometry do not match.'.format(reco_space.grid.ndim,
                                                         geometry.ndim))

    ndim = proj_data.space.grid.ndim

    # Create astra geometries
    vol_geom = astra_volume_geometry(reco_space)
    proj_geom = astra_projection_geometry(geometry, reco_space)

    # Create ASTRA data structures
    vol_id = astra_data(vol_geom, datatype='volume', data=None,
                        ndim=reco_space.grid.ndim)
    sino_id = astra_data(proj_geom, datatype='projection', data=proj_data)

    # Create projector
    vol_interp = proj_data.space.interp

    # Create algorithm
    algo_id = astra_algorithm('backward', ndim, vol_id, sino_id, proj_id=None,
                              impl='cuda')

    # Run algorithm and delete it
    astra.algorithm.run(algo_id)

    # Wrap data
    if ndim == 2:
        get_data = astra.data2d.get_shared
    else:  # ndim = 3
        get_data = astra.data3d.get_shared

    # TODO: check if axes have to be swapped
    elem = reco_space.element(get_data(vol_id))

    # Delete ASTRA objects
    astra_cleanup()

    return elem


def astra_gpu_backward_projector_apply(proj_data, geometry, reco_data):
    """Run an ASTRA backward projection on the given data using the GPU.

        Parameters
        ----------
        proj_data : `DiscreteLpVector`
            Projection data to which the backward projector is applied
        geometry : `Geometry`
            Geometry defining the tomographic setup
        reco_data : `DiscreteLpVector`
            Space to which the calling operator maps
        """
