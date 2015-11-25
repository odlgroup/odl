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

from odl import DiscreteLp
# Internal
from odltomo.backends.astra_setup import (astra_projection_geometry,
    astra_volume_geometry, astra_data, astra_algorithm, astra_cleanup)
from odltomo.geometry.geometry import Geometry


__all__ = ('astra_gpu_forward_projector_call',
           'astra_gpu_forward_projector_apply',
           'astra_gpu_backward_projector_call',
           'astra_gpu_backward_projector_apply',)


# TODO: rename gpu to cuda?
def astra_gpu_forward_projector_call(vol_data, geometry, proj_space):
    """Run an ASTRA forward projection on the given data using the GPU.

    Parameters
    ----------
    vol_data : `odl.DiscreteLp` element
        Volume data to which the projector is applied
    geometry : `Geometry`
        Geometry defining the tomographic setup
    proj_space : `odl.DiscreteLp`
        Space to which the calling operator maps

    Returns
    -------
    projection : proj_space element
        Projection data resulting from the application of the projector
    """
    if not isinstance(vol_data, DiscreteLp.Vector):
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
    proj_geom = astra_projection_geometry(geometry)

    # Create ASTRA data structures
    vol_id = astra_data(vol_geom, datatype='volume', data=vol_data)
    sino_id = astra_data(proj_geom, datatype='projection', data=None,
                         ndim=proj_space.grid.ndim)

    # Create algorithm
    algo_id = astra_algorithm('forward', ndim, vol_id, sino_id, proj_id=None,
                              impl='cuda')

    # Run algorithm and delete it
    astra.algorithm.run(algo_id)

    # Wrap data
    if ndim == 2:
        get_data = astra.data2d.get_shared
    else:  # ndim = 3
        get_data = astra.data3d.get_shared

    # TODO: check if axes have to be swapped
    elem = proj_space.element(get_data(sino_id))

    # Delete ASTRA objects
    astra_cleanup()

    return elem


def astra_gpu_forward_projector_apply(vol_data, geometry, proj_data, direction):
    """Run an ASTRA forward projection on the given data using the GPU.

    Parameters
    ----------
    vol_data : `odl.DiscreteLp` element
        Volume data to which the projector is applied
    geometry : `Geometry`
        Geometry defining the tomographic setup
    proj_data : `odl.DiscreteLp.Vector`
        Projection space element to which the projection data is written
    direction : {'forward', 'backward'}
        'forward' : apply forward projection

        'backward' : apply backprojection

    Returns
    -------
    None
    """

def astra_gpu_backward_projector_call():
    pass


def astra_gpu_backward_projector_apply():
    pass
