# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Backend for ASTRA using CPU."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

try:
    import astra
except ImportError:
    pass
import numpy as np

from odl.discr import DiscreteLp, DiscreteLpElement
from odl.tomo.backends.astra_setup import (
    astra_projection_geometry, astra_volume_geometry, astra_data,
    astra_projector, astra_algorithm)
from odl.tomo.geometry import Geometry
from odl.util import writable_array


__all__ = ('astra_cpu_forward_projector', 'astra_cpu_back_projector')


# TODO: use context manager when creating data structures
# TODO: is magnification scaling at the right place?

def astra_cpu_forward_projector(vol_data, geometry, proj_space, out=None):
    """Run an ASTRA forward projection on the given data using the CPU.

    Parameters
    ----------
    vol_data : `DiscreteLpElement`
        Volume data to which the forward projector is applied
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
        raise TypeError('volume data {!r} is not a `DiscreteLpElement` '
                        'instance.'.format(vol_data))
    if vol_data.space.impl != 'numpy':
        raise TypeError('dspace {!r} of the volume is not an '
                        'instance of `NumpyNtuples`'
                        ''.format(vol_data.space.dspace))
    if not isinstance(geometry, Geometry):
        raise TypeError('geometry  {!r} is not a Geometry instance'
                        ''.format(geometry))
    if not isinstance(proj_space, DiscreteLp):
        raise TypeError('projection space {!r} is not a DiscreteLp '
                        'instance.'.format(proj_space))
    if proj_space.impl != 'numpy':
        raise TypeError('data type {!r} of the reconstruction space is not an '
                        'instance of NumpyNtuples'.format(proj_space.dspace))
    if vol_data.ndim != geometry.ndim:
        raise ValueError('dimensions {} of volume data and {} of geometry '
                         'do not match'
                         ''.format(vol_data.ndim, geometry.ndim))
    if out is None:
        out = proj_space.element()
    else:
        if out not in proj_space:
            raise TypeError('`out` {} is neither None nor a '
                            'DiscreteLpElement instance'.format(out))

    ndim = vol_data.ndim

    # Create astra geometries
    vol_geom = astra_volume_geometry(vol_data.space)
    proj_geom = astra_projection_geometry(geometry)

    # Create projector
    if not all(s == vol_data.space.interp_byaxis[0]
               for s in vol_data.space.interp_byaxis):
        raise ValueError('volume interpolation must be the same in each '
                         'dimension, got {}'.format(vol_data.space.interp))
    vol_interp = vol_data.space.interp
    proj_id = astra_projector(vol_interp, vol_geom, proj_geom, ndim,
                              impl='cpu')

    # Create ASTRA data structures
    # Since ASTRA uses (rows, cols) == (-y, x) as coordinate system, we need
    # to rotate by 90 degrees counter-clockwise to match the axes (x, y) as
    # used in `astra_volume_geometry`.
    vol_data_arr = np.rot90(np.asarray(vol_data), 1)
    vol_id = astra_data(vol_geom, datatype='volume', data=vol_data_arr,
                        allow_copy=True)

    with writable_array(out, dtype='float32', order='C') as out_arr:
        sino_id = astra_data(proj_geom, datatype='projection', data=out_arr,
                             ndim=proj_space.ndim)

        # Create algorithm
        algo_id = astra_algorithm('forward', ndim, vol_id, sino_id, proj_id,
                                  impl='cpu')

        # Run algorithm
        astra.algorithm.run(algo_id)

    # Delete ASTRA objects
    astra.algorithm.delete(algo_id)
    astra.data2d.delete((vol_id, sino_id))
    astra.projector.delete(proj_id)

    return out


def astra_cpu_back_projector(proj_data, geometry, reco_space, out=None):
    """Run an ASTRA back-projection on the given data using the CPU.

    Parameters
    ----------
    proj_data : `DiscreteLpElement`
        Projection data to which the back-projector is applied
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
    if proj_data.space.impl != 'numpy':
        raise TypeError('data type {!r} of the projection space is not an '
                        'instance of NumpyNtuples'
                        ''.format(proj_data.shape.dspace))
    if not isinstance(geometry, Geometry):
        raise TypeError('geometry  {!r} is not a Geometry instance'
                        ''.format(geometry))
    if not isinstance(reco_space, DiscreteLp):
        raise TypeError('reconstruction space {!r} is not a DiscreteLp '
                        'instance'.format(reco_space))
    if reco_space.impl != 'numpy':
        raise TypeError('data type {!r} of the reconstruction space is not an '
                        'instance of NumpyNtuples'.format(reco_space.dspace))
    if reco_space.ndim != geometry.ndim:
        raise ValueError('dimensions {} of reconstruction space and {} of '
                         'geometry do not match'.format(
                             reco_space.ndim, geometry.ndim))
    if out is None:
        out = reco_space.element()
    else:
        if out not in reco_space:
            raise TypeError('`out` {} is neither None nor a '
                            'DiscreteLpElement instance'.format(out))

    ndim = proj_data.ndim

    # Create astra geometries
    vol_geom = astra_volume_geometry(reco_space)
    proj_geom = astra_projection_geometry(geometry)

    # Create ASTRA data structure
    sino_id = astra_data(proj_geom, datatype='projection', data=proj_data,
                         allow_copy=True)

    # Create projector
    # TODO: implement with different schemes for angles and detector
    if not all(s == proj_data.space.interp_byaxis[0]
               for s in proj_data.space.interp_byaxis):
        raise ValueError('data interpolation must be the same in each '
                         'dimension, got {}'
                         ''.format(proj_data.space.interp_byaxis))
    proj_interp = proj_data.space.interp
    proj_id = astra_projector(proj_interp, vol_geom, proj_geom, ndim,
                              impl='cpu')

    # Convert out to correct dtype and order if needed.
    # Since we transpose, we need to use F-contiguousness.
    with writable_array(out, dtype='float32', order='F') as out_arr:
        vol_id = astra_data(vol_geom, datatype='volume', data=out_arr.T,
                            ndim=reco_space.ndim)
        # Create algorithm
        algo_id = astra_algorithm('backward', ndim, vol_id, sino_id, proj_id,
                                  impl='cpu')

        # Run algorithm
        astra.algorithm.run(algo_id)

        # If we hadn't transposed `out_arr` we would have to rotate
        # clockwise by 90 degrees to invert the transition from (rows, cols)
        # to (x, y). Since transposition happens automatically when exiting
        # this context, we need to flip vertically, because
        # rot90(a, -1) == fliplr(a.T)
        out_arr[:] = np.fliplr(out_arr)

    # Weight the adjoint by appropriate weights
    scaling_factor = float(proj_data.space.weighting.const)
    scaling_factor /= float(reco_space.weighting.const)

    out *= scaling_factor

    # Delete ASTRA objects
    astra.algorithm.delete(algo_id)
    astra.data2d.delete((vol_id, sino_id))
    astra.projector.delete(proj_id)

    return out


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
