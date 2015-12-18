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

"""Backend for ASTRA using CPU"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
try:
    import astra
except ImportError:
    pass

# Internal
from odl.discr import DiscreteLp, DiscreteLpVector
from odl.space.ntuples import Ntuples
from odl.tomo.backends.astra_setup import (astra_projection_geometry,
                                           astra_volume_geometry, astra_data,
                                           astra_projector, astra_algorithm,
                                           astra_cleanup)
from odl.tomo.geometry.geometry import Geometry
from odl.tomo.geometry.fanbeam import FanFlatGeometry

__all__ = ('astra_cpu_forward_projector_call',
           'astra_cpu_backward_projector_call')


# TODO: Fix inconsistent scaling of ASTRA projector with pixel size

def astra_cpu_forward_projector_call(vol_data, geometry, proj_space, out=None):
    """Run an ASTRA forward projection on the given data using the CPU.

    Parameters
    ----------
    vol_data : `DiscreteLpVector`
        Volume data to which the forward projector is applied
    geometry : `Geometry`
        Geometry defining the tomographic setup
    proj_space : `DiscreteLp`
        Space to which the calling operator maps
    out : `DiscreteLpVector`, optional
        Vector in the projection space to which the result is written. If
        `None` creates an element in the projection space ``proj_space`

    Returns
    -------
    out : ``proj_space`` element
        Projection data resulting from the application of the projector
    """
    if not isinstance(vol_data, DiscreteLpVector):
        raise TypeError('volume data {!r} is not a `DiscreteLpVector` '
                        'instance.'.format(vol_data))
    if not isinstance(geometry, Geometry):
        raise TypeError('geometry  {!r} is not a `Geometry` instance.'
                        ''.format(geometry))
    if not isinstance(proj_space, DiscreteLp):
        raise TypeError('projection space {!r} is not a `DiscreteLp` '
                        'instance.'.format(proj_space))
    if not isinstance(proj_space.dspace, Ntuples):
        raise TypeError('data type {!r} of the reconstruction space is not an '
                        'instance of `Ntuples`'.format(proj_space.dspace))
    if not isinstance(vol_data.space.dspace, Ntuples):
        raise TypeError('')

    if out is None:
        out = proj_space.element()
    else:
        if not isinstance(out, DiscreteLpVector):
            raise TypeError('out {} is neither `None` nor a '
                            '`DiscreteLpVector` instance'.format(out))
    if vol_data.ndim != geometry.ndim:
        raise ValueError('dimensions {} of volume data and {} of geometry '
                         'do not match.'
                         ''.format(vol_data.ndim, geometry.ndim))

    ndim = vol_data.ndim

    # Create astra geometries
    vol_geom = astra_volume_geometry(vol_data.space)
    proj_geom = astra_projection_geometry(geometry)

    # Create ASTRA data structures
    vol_id = astra_data(vol_geom, datatype='volume', data=vol_data)
    sino_id = astra_data(proj_geom, datatype='projection', data=out,
                         ndim=proj_space.ndim)

    # Create projector
    vol_interp = vol_data.space.interp
    proj_id = astra_projector(vol_interp, vol_geom, proj_geom, ndim,
                              impl='cpu')

    # Create algorithm
    algo_id = astra_algorithm('forward', ndim, vol_id, sino_id, proj_id,
                              impl='cpu')

    # Run algorithm and delete it
    astra.algorithm.run(algo_id)

    # Flip detector pixels for fanflat
    if isinstance(geometry, FanFlatGeometry):
        out[:] = out.asarray()[::-1, ::-1]

    # Delete ASTRA objects
    astra.algorithm.delete(algo_id)
    astra.data2d.delete((vol_id, sino_id))
    astra.projector.delete(proj_id)

    return out


def astra_cpu_backward_projector_call(proj_data, geometry, reco_space,
                                      out=None):
    """Run an ASTRA backward projection on the given data using the CPU.

    Parameters
    ----------
    proj_data : `DiscreteLpVector`
        Projection data to which the backward projector is applied
    geometry : `Geometry`
        Geometry defining the tomographic setup
    reco_space : `DiscreteLp`
        Space to which the calling operator maps
    out : `DiscreteLpVector` or `None`, optional
        Vector in the reconstruction space to which the result is written. If
        `None` creates an element in the reconstruction space ``reco_space``

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
    if not isinstance(reco_space.dspace, Ntuples):
        raise TypeError('data type {!r} of the reconstruction space is not an '
                        'instance of `Ntuples`'.format(reco_space.dspace))
    if reco_space.ndim != geometry.ndim:
        raise ValueError('dimensions {} of reconstruction space and {} of '
                         'geometry do not match.'.format(
                             reco_space.ndim, geometry.ndim))
    if out is None:
        out = reco_space.element()
    else:
        if not isinstance(out, DiscreteLpVector):
            raise TypeError('out {} is neither `None` nor a '
                            '`DiscreteLpVector` instance'.format(out))

    ndim = proj_data.ndim

    # Create astra geometries
    vol_geom = astra_volume_geometry(reco_space)
    proj_geom = astra_projection_geometry(geometry)

    # Create ASTRA data structures
    vol_id = astra_data(vol_geom, datatype='volume', data=out,
                        ndim=reco_space.ndim)
    sino_id = astra_data(proj_geom, datatype='projection', data=proj_data)

    # Create projector
    vol_interp = proj_data.space.interp
    proj_id = astra_projector(vol_interp, vol_geom, proj_geom, ndim,
                              impl='cpu')

    # Create algorithm
    algo_id = astra_algorithm('backward', ndim, vol_id, sino_id, proj_id,
                              impl='cpu')

    # Run algorithm and delete it
    astra.algorithm.run(algo_id)

    # flip both dimensions = rotate 180 degrees
    # TODO: Maybe flipping angles in the geometry would be better
    if isinstance(geometry, FanFlatGeometry):
        out[:] = out.asarray()[::-1, ::-1]

    # Delete ASTRA objects
    astra.algorithm.delete(algo_id)
    astra.data2d.delete((vol_id, sino_id))
    astra.projector.delete(proj_id)

    return out
