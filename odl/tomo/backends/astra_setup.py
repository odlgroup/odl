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

"""Helper functions to prepare ASTRA algorithms.

This module contains utility functions to convert data structures from
the TomODL representation to ASTRA's data structures, including:

* volume geometries
* projection geometries
* data arrays
* algorithm configuration dictionaries
* projectors and backprojectors
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import astra
import numpy as np
import odl

# Internal
from odltomo.geometry.geometry import Geometry
from odltomo.geometry.parallel import (Parallel2dGeometry, Parallel3dGeometry)
from odltomo.geometry.fanbeam import (FanBeamGeometry, FanFlatGeometry)
from odltomo.geometry.conebeam import (CircularConeFlatGeometry,
                                       HelicalConeFlatGeometry)


__all__ = ('astra_volume_geometry', 'astra_projection_geometry',
           'astra_data', 'astra_projector', 'astra_algorithm',
           'astra_geom_to_vec', 'astra_cleanup')


def astra_volume_geometry(discr_reco):
    """Create an ASTRA volume geometry from the discretized domain.

    Parameters
    ----------
    discr_reco : `odl.DiscreteLp`
        Discretization of an L2 space on the reconstruction domain.
        It must be 2- or 3-dimensional and sampled by a regular grid.

    Returns
    -------
    astra_geo : dict
        The ASTRA volume geometry

    Raises
    ------
    NotImplementedError
        if in 3d, the grid strides (voxel sizes) are not the same in
        each dimension. This is currently only supported in 2d by
        ASTRA.
    """
    # TODO: allow other discretizations?
    if not isinstance(discr_reco, odl.DiscreteLp):
        raise TypeError('discretized domain {!r} is not a `DiscreteLp` '
                        'instance.'.format(discr_reco))

    if not isinstance(discr_reco.grid, odl.RegularGrid):
        raise TypeError('sampling grid {!r} is not a `RegularGrid` '
                        'instance.'.format(discr_reco.grid))

    vol_shp = discr_reco.grid.shape
    vol_min = discr_reco.grid.min()
    vol_max = discr_reco.grid.max()

    if discr_reco.grid.ndim == 2:
        # ASTRA does in principle support custom minimum and maximum
        # values for the volume extent, but projector creation fails
        # if voxels are non-isotropic. We raise an exception here in
        # the meanwhile.
        if not np.allclose(discr_reco.grid.stride[1:],
                           discr_reco.grid.stride[:-1]):
            # TODO: for parallel geometries, one can work around this issue
            raise NotImplementedError('non-isotropic voxels not supported by '
                                      'ASTRA.')
        vol_geom = astra.create_vol_geom(vol_shp[1], vol_shp[0],
                                         vol_min[0], vol_max[0],
                                         vol_min[1], vol_max[1])
    elif discr_reco.grid.ndim == 3:
        # Non-isotropic voxels are not yet supported in 3d ASTRA
        if not np.allclose(discr_reco.grid.stride[1:],
                           discr_reco.grid.stride[:-1]):
            # TODO: for parallel geometries, one can work around this issue
            raise NotImplementedError('non-isotropic voxels not supported by '
                                      'ASTRA.')
        # rows (y) first, columns (y) second, slices (z) third
        vol_geom = astra.create_vol_geom(vol_shp[1], vol_shp[0], vol_shp[2])
    else:
        raise ValueError('{}-dimensional volume geometries not supported '
                         'by ASTRA.'.format(discr_reco.ndim))
    return vol_geom


def astra_geom_to_vec(geometry):
    """Create 2d or 3d vectors from an `odltomo` geometry object to be used
    for ASTRA's `flat_vec` or `parallel3d` and `cone_vec`, respectively.

    2d geometry: `fanflat_vec`
    Each row of vectors corresponds to a single projection, and consists of:
     ( srcX, srcY, dX, dY, uX, uY )
    src : the ray source
    d : the center of the detector
    u : the vector between the centers of detector pixels 0 and 1

    3d geometry: `parallel3d_vec`
    Each row of vectors corresponds to a single projection, and consists of:
     ( rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ )
    ray : the ray direction
    d   : the center of the detector
    u   : the vector from detector pixel (0,0) to (0,1)
    v   : the vector from detector pixel (0,0) to (1,0)

    3d geometry: `cone_vec`
    Each row of vectors corresponds to a single projection, and consists of:
     ( srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ ):
     src : the ray source
     d   : the center of the detector
     u   : the vector from detector pixel (0,0) to (0,1)
     v   : the vector from detector pixel (0,0) to (1,0)

    Parameters
    ----------
    geometry : `Geometry`
        The odl-tomo geometry object from which the ASTRA geometry is
        created.

    Returns
    -------
    vectors : ndarray
        Numpy array of shape (number of angles, 12)
    """

    angles = geometry.angle_grid
    num_angles = geometry.angle_grid.ntotal

    if isinstance(geometry, (CircularConeFlatGeometry,
                             HelicalConeFlatGeometry)):
        vectors = np.zeros((num_angles, 12))
        det_pix_width = geometry.det_grid.stride[0]
        det_pix_height = geometry.det_grid.stride[1]

        for nn in range(num_angles):
            angle = angles[nn][0]

            # source position
            vectors[nn, 0:3] = geometry.src_position(angle)

            # center of detector
            vectors[nn, 3:6] = geometry.det_refpoint(angle)

            # vector from detector pixel (0,0) to (0,1)
            # TODO: use det_rotation method instead of
            vectors[nn, 6] = np.cos(angle) * det_pix_width
            vectors[nn, 7] = np.sin(angle) * det_pix_width
            # vectors[nn, 8] = 0

            # vector from detector pixel (0,0) to (1,0)
            # vectors[nn, 9] = 0
            # vectors[nn, 10] = 0
            vectors[nn, 11] = det_pix_height

    elif isinstance(geometry, Parallel3dGeometry):
        vectors = np.zeros((num_angles, 12))
        det_pix_width = geometry.det_grid.stride[0]
        det_pix_height = geometry.det_grid.stride[1]

        for nn in range(num_angles):
            angle = angles[nn][0]

            # ray direction
            vectors[nn, 0] = np.sin(angle)
            vectors[nn, 1] = np.cos(angle)
            # vectors[nn, 2] = 0

            # center of detector
            # vectors[nn, 3:6] = 0

            # vector from detector pixel (0,0) to (0,1)
            # TODO: use det_rotation method instead of
            vectors[nn, 6] = np.cos(angle) * det_pix_width
            vectors[nn, 7] = np.sin(angle) * det_pix_width
            # vectors[nn, 8] = 0

            # vector from detector pixel (0,0) to (1,0)
            # vectors[nn, 9] = 0
            # vectors[nn, 10] = 0
            vectors[nn, 11] = det_pix_height

    elif isinstance(geometry, FanBeamGeometry):
        vectors = np.zeros((num_angles, 6))
        det_pix_width = geometry.det_grid.stride[0]

        for nn in range(num_angles):
            angle = angles[nn][0]

            # source position
            vectors[nn, 0:2] = geometry.src_position(angle)

            # center of detector
            vectors[nn, 2:4] = geometry.det_refpoint(angle)

            # vector from detector pixel (0,0) to (0,1)
            # TODO: use det_rotation method instead of
            vectors[nn, 4] = np.cos(angle) * det_pix_width
            vectors[nn, 5] = np.sin(angle) * det_pix_width

    else:
        raise ValueError('invalid geometry type {!r}.'.format(
            geometry.__class__.__name__))

    return vectors


def astra_projection_geometry(geometry):
    """Create an ASTRA projection geometry from the given geometry.

    Parameters
    ----------
    geometry : `Geometry`
        The odl-tomo geometry object from which the ASTRA geometry is
        created.

    Returns
    -------
    astra_geo : dict
        The ASTRA projection geometry
    """
    if not isinstance(geometry, Geometry):
        raise TypeError('geometry {!r} is not a `Geometry` instance.'
                        ''.format(geometry))
    if not geometry.has_det_sampling:
        raise ValueError('geometry has no detector sampling grid.')
    if not geometry.has_motion_sampling:
        raise ValueError('geometry has no motion sampling grid.')
    if not isinstance(geometry.det_grid, odl.RegularGrid):
        raise TypeError('detector sampling grid {!r} is not a `RegularGrid` '
                        'instance.'.format(geometry.det_grid))

    # TODO: fanflat_vec
    if isinstance(geometry, Parallel2dGeometry):
        det_width = geometry.det_grid.stride[0]
        det_count = geometry.det_grid.shape[0]
        angles = geometry.motion_grid.coord_vectors[0]
        proj_geom = astra.create_proj_geom('parallel', det_width, det_count,
                                           angles)
    elif isinstance(geometry, FanFlatGeometry):
        det_width = geometry.det_grid.stride[0]
        det_count = geometry.det_grid.shape[0]
        angles = geometry.motion_grid.coord_vectors[0]
        source_origin = geometry.src_radius
        origin_det = geometry.det_radius  # ! In PyASTRA doc falsely labelled
        # `source_det`
        proj_geom = astra.create_proj_geom('fanflat', det_width, det_count,
                                           angles, source_origin, origin_det)
    elif isinstance(geometry, Parallel3dGeometry):
        det_width = geometry.det_grid.stride[0]  # ASTRA: x
        det_height = geometry.det_grid.stride[1]  # ASTRA: y
        det_row_count = geometry.det_grid.shape[0]
        det_col_count = geometry.det_grid.shape[1]
        angles = geometry.motion_grid.coord_vectors[0]
        proj_geom = astra.create_proj_geom(
            'parallel3d', det_width, det_height, det_row_count,
            det_col_count, angles)
    elif isinstance(geometry, CircularConeFlatGeometry):
        det_width = geometry.det_grid.stride[0]  # ASTRA: x
        det_height = geometry.det_grid.stride[1]  # ASTRA: y
        det_row_count = geometry.det_grid.shape[0]
        det_col_count = geometry.det_grid.shape[1]
        angles = geometry.motion_grid.coord_vectors[0]
        source_origin = geometry.src_radius
        origin_det = geometry.det_radius  # ! In PyASTRA doc falsely labelled
        # `source_det`
        proj_geom = astra.create_proj_geom(
            'cone', det_width, det_height, det_row_count, det_col_count,
            angles, source_origin, origin_det)
    elif isinstance(geometry, HelicalConeFlatGeometry):
        det_row_count = geometry.det_grid.shape[0]
        det_col_count = geometry.det_grid.shape[1]
        vec = astra_geom_to_vec(geometry)
        proj_geom = astra.create_proj_geom('cone_vec', det_row_count,
                                           det_col_count, vec)
    else:
        raise NotImplementedError('ASTRA geometry creation not supported for '
                                  'geometry type {}.'.format(type(geometry)))

    return proj_geom


def astra_data(astra_geom, datatype, data=None, ndim=2):
    """Create an ASTRA data structure.

    Parameters
    ----------
    astra_geom : dict
        ASTRA geometry for the data creator, must correspond to the
        given data dype
    datatype : {'volume', 'projection'}
        Type of the data container
    data : `odl.DiscreteLp` element, optional
        Data for the initialization of the data structure
    ndim : {2, 3}, optional
        Dimension of the data. If `data` is not `None`, this parameter
        has no effect.

    Returns
    -------
    id : int
        ASTRA internal id for the new data structure
    """
    if data is not None:
        if not isinstance(data, odl.DiscreteLp.Vector):
            raise TypeError('data {!r} is not a `DiscreteLp.Vector` instance.'
                            ''.format(data))
        ndim = data.space.grid.ndim
    else:
        ndim = int(ndim)

    if datatype == 'volume':
        astra_dtype_str = '-vol'
    elif datatype == 'projection':
        astra_dtype_str = '-sino'
    else:
        raise ValueError('data type {!r} not understood.'.format(datatype))

    # Get the functions from the correct module
    if ndim == 2:
        link = astra.data2d.link
        create = astra.data2d.create
    elif ndim == 3:
        link = astra.data3d.link
        create = astra.data3d.create
    else:
        raise ValueError('{}-dimensional data structures not supported.'
                         ''.format(ndim))

    if data is not None:
        if not isinstance(data.ntuple, odl.Fn.Vector):
            # Something else than NumPy data representation
            raise NotImplementedError('ASTRA supports data wrapping only for '
                                      '`numpy.ndarray` instances.')

        dshape = data.space.grid.shape
        if ndim == 2:
            # Axes 0 and 1 need to be swapped in input (and output)
            # if datatype == ''
            if datatype == 'volume':
                astra_shape = (dshape[1], dshape[0])
            else:
                astra_shape = (dshape[0], dshape[1])
        else:
            # TODO: check what to do in this case - swapaxes is not an option
            # since it results in a non-contiguous array. Maybe one can
            # compensate afterwards?

            astra_shape = (dshape[2], dshape[1], dshape[0])
        print(' \ndatatype: ', datatype, ', data_shape', data.shape,
              ', astra_shape:', astra_shape, '\n')
        # ASTRA checks if data is c-contiguous and aligned
        return link(astra_dtype_str, astra_geom,
                    data.ntuple.data.reshape(astra_shape))
    else:
        return create(astra_dtype_str, astra_geom)


def astra_projector(vol_interp, astra_vol_geom, astra_proj_geom, ndim, impl):
    """Create an ASTRA projector configuration dictionary.

    Parameters
    ----------
    vol_interp : {'nearest', 'linear'}
        Interpolation type of the volume discretization
    astra_vol_geom : dict
        ASTRA volume geometry dictionary
    astra_proj_geom : dict
        ASTRA projection geometry dictionary
    ndim : {2, 3}
        Number of dimensions of the projector
    impl : {'cpu', 'cuda'}
        Implementation of the projector

    Returns
    -------
    cfg : dict
        ASTRA dict with initialized 'type' key
    """
    if vol_interp not in ('nearest', 'linear'):
        raise ValueError('volume interpolation type {!r} not understood.'
                         ''.format(vol_interp))
    impl = str(impl).lower()
    if impl not in ('cpu', 'cuda'):
        raise ValueError('implementation type {!r} not understood.'
                         ''.format(impl))

    if 'type' not in astra_proj_geom:
        raise ValueError('invalid projection geometry dict {}.'
                         ''.format(astra_proj_geom))

    ndim = int(ndim)
    if ndim != 2 or impl == 'cuda':
        # Remove this as soon as ASTRA supports 3d/CUDA projectors
        raise ValueError('projector initialization in ASTRA is only supported '
                         'for 2d and CPU. GPU algorithms do not need a '
                         'projector.')

    proj_type = astra_proj_geom['type']
    if proj_type not in ('parallel', 'fanflat'):
        raise ValueError('invalid 2d geometry type {!r}.'.format(proj_type))

    # Mapping from interpolation type and geometry to ASTRA projector type.
    # "I" means probably mathematically inconsistent.
    # Some projectors are not implemented, e.g. CPU 3d projectors in general
    # TODO: ASTRA supports area weights (strip) for parallel and fanflat on CPU
    type_map_cpu = {'parallel': {'nearest': 'line',
                                 'linear': 'linear'},  # I
                    'fanflat': {'nearest': 'line_fanflat',
                                'linear': 'line_fanflat'},  # I
                    'parallel3d': {'nearest': 'linear3d',  # I
                                   'linear': 'linear3d'},  # I
                    'cone': {'nearest': 'linearcone',  # I
                             'linear': 'linearcone'}}  # I
    type_map_cpu['fanflat_vec'] = type_map_cpu['fanflat']
    type_map_cpu['parallel3d_vec'] = type_map_cpu['parallel3d']
    type_map_cpu['cone_vec'] = type_map_cpu['cone']

    # GPU algorithms do not requiure a projector. This is just for
    # completeness.
    type_map_cuda = {'parallel': 'cuda',  # I
                     'parallel3d': 'cuda3d'}  # I
    type_map_cuda['fanflat'] = type_map_cuda['parallel']
    type_map_cuda['fanflat_vec'] = type_map_cuda['fanflat']
    type_map_cuda['cone'] = type_map_cuda['parallel3d']
    type_map_cuda['parallel3d_vec'] = type_map_cuda['parallel3d']
    type_map_cuda['cone_vec'] = type_map_cuda['cone']

    proj_cfg = {}
    if impl == 'cpu':
        proj_cfg['type'] = type_map_cpu[proj_type][vol_interp]
    else:  # impl == 'cuda'
        proj_cfg['type'] = type_map_cuda[proj_type]

    proj_cfg['VolumeGeometry'] = astra_vol_geom
    proj_cfg['ProjectionGeometry'] = astra_proj_geom

    if ndim == 2:
        return astra.projector.create(proj_cfg)
    else:
        return astra.projector3d.create(proj_cfg)


def astra_algorithm(direction, ndim, vol_id, sino_id, proj_id, impl):
    """Create an ASTRA algorithm object to run the projector.

    Parameters
    ----------
    direction : {'forward', 'backward'}
        Apply the forward projection if 'forward', otherwise the
        backprojection
    ndim : {2, 3}
        Number of dimensions of the projector
    vol_id : int
        ASTRA ID of the volume data object
    sino_id : int
        ASTRA ID of the projection data object
    proj_id : int or None
        ASTRA ID of the `cpu` projector. None for `cuda`
    impl : {'cpu', 'cuda'}
        Implementation of the projector

    Returns
    -------
    id : int
        ASTRA internal id for the new algorithm structure
    """
    if direction not in ('forward', 'backward'):
        raise ValueError('direction {!r} not understood.'.format(direction))
    if ndim not in (2, 3):
        raise ValueError('{}-dimensional projectors not supported.'
                         ''.format(ndim))
    if impl not in ('cpu', 'cuda'):
        raise ValueError('implementation type {!r} not understood.'
                         ''.format(impl))
    if ndim is 3 and impl is 'cpu':
        raise NotImplementedError(
            '3d algorithms for cpu is not supported by ASTRA')

    algo_map = {'forward': {2: {'cpu': 'FP', 'cuda': 'FP_CUDA'},
                            3: {'cpu': None, 'cuda': 'FP3D_CUDA'}},
                'backward': {2: {'cpu': 'BP', 'cuda': 'BP_CUDA'},
                             3: {'cpu': None, 'cuda': 'BP3D_CUDA'}}}

    algo_cfg = {'type': algo_map[direction][ndim][impl]}
    if impl is 'cpu':
        algo_cfg['ProjectorId'] = proj_id
    algo_cfg['ProjectionDataId'] = sino_id
    if direction is 'forward':
        algo_cfg['VolumeDataId'] = vol_id
    elif direction is 'backward':
        algo_cfg['ReconstructionDataId'] = vol_id
    else:
        # should not happen
        raise ValueError('unkown direction {}'.format(direction))
    return astra.algorithm.create(algo_cfg)


def astra_cleanup():
    """Delete all ASTRA objects."""

    modules = [astra.data2d, astra.data3d,
               astra.projector, astra.projector3d,
               astra.algorithm]

    for module in modules:
        module.clear()
