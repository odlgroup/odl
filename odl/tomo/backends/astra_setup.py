# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Helper functions to prepare ASTRA algorithms.

This module contains utility functions to convert data structures from the
ODL geometry representation to ASTRA's data structures, including:

* volume geometries
* projection geometries
* create vectors from geometries
* data arrays
* projectors
* algorithm configuration dictionaries

`ASTRA documentation <http://www.astra-toolbox.com/>`_.

`ASTRA on GitHub <https://github.com/astra-toolbox/>`_.
"""

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np

from odl.discr import DiscreteLp, DiscreteLpElement
from odl.tomo.geometry import (
    DivergentBeamGeometry, Flat1dDetector, Flat2dDetector, Geometry,
    ParallelBeamGeometry)
from odl.tomo.util.utility import euler_matrix
from odl.util.npy_compat import moveaxis

try:
    import astra
except ImportError:
    ASTRA_AVAILABLE = False
    ASTRA_VERSION = ''
else:
    ASTRA_AVAILABLE = True


# Make sure that ASTRA >= 1.7 is used
if ASTRA_AVAILABLE:
    try:
        # Available from 1.8 on
        ASTRA_VERSION = astra.__version__
    except AttributeError:
        # Below version 1.8
        _maj = astra.astra.version() // 100
        _min = astra.astra.version() % 100
        ASTRA_VERSION = '.'.join([str(_maj), str(_min)])
        if (_maj, _min) < (1, 7):
            warnings.warn(
                'your version {}.{} of ASTRA is unsupported, please upgrade '
                'to 1.7 or higher'.format(_maj, _min), RuntimeWarning)


__all__ = ('ASTRA_AVAILABLE', 'ASTRA_VERSION', 'astra_supports',
           'astra_volume_geometry', 'astra_projection_geometry',
           'astra_data', 'astra_projector', 'astra_algorithm',
           'astra_conebeam_3d_geom_to_vec',
           'astra_conebeam_2d_geom_to_vec',
           'astra_parallel_3d_geom_to_vec')


# ASTRA_FEATURES contains a set of features along with version specifiers
# to track ASTRA support for those features. The version specifiers must
# be valid setuptools package requirements (without package name), e.g.,
# as exact version '==1.7', as inequality '>=1.7' or as several requirements
# that need to be satisfied simultaneously, e.g., '>=1.8,<=2.0'.
# To give multiple requirements that should be OR-ed together, use a
# sequence instead of a single string in the dictionary, e.g.,
# ['==1.7', '>=1.8,<=2.0'].

ASTRA_FEATURES = {
    # Cell sizes not equal in both axes in 2d, currently crashes.
    'anisotropic_voxels_2d': None,

    # Cell sizes not equal all 3 axes in 3d, see
    # https://github.com/astra-toolbox/astra-toolbox/pull/41
    'anisotropic_voxels_3d': '>=1.8',

    # ASTRA geometry defined by vectors supported in the current
    # development version, will be in the next release. See
    # https://github.com/astra-toolbox/astra-toolbox/issues/54
    'par2d_vec_geometry': '>1.8.3',

    # Density weighting for cone 2d (fan beam), not supported yet,
    # see the discussion in
    # https://github.com/astra-toolbox/astra-toolbox/issues/71
    'cone2d_density_weighting': None,

    # Hacky version of ray-density weighting in cone beam backprojection for
    # constant source-detector distance, see
    # https://github.com/astra-toolbox/astra-toolbox/issues/71
    # and
    # https://github.com/astra-toolbox/astra-toolbox/pull/84
    'cone3d_hacky_density_weighting': '>=1.8',

    # General case not supported yet, see the discussion in
    # https://github.com/astra-toolbox/astra-toolbox/issues/71
    'cone3d_density_weighting': None,

    # Fix for division by zero with detector midpoint normal perpendicular
    # to geometry axis in parallel 3d, see
    # https://github.com/astra-toolbox/astra-toolbox/issues/18
    'par3d_det_mid_pt_perp_to_axis': '>=1.7.2',

    # Linking instead of copying of GPU memory, see
    # https://github.com/astra-toolbox/astra-toolbox/pull/93
    'gpulink': '>=1.8.3',
}


def astra_supports(feature):
    """Return bool indicating whether current ASTRA supports ``feature``.

    Parameters
    ----------
    feature : str
        Name of a potential feature of ASTRA. See ``ASTRA_FEATURES`` for
        possible values.

    Returns
    -------
    supports : bool
        ``True`` if the currently imported version of ASTRA supports the
        feature in question, ``False`` otherwise.
    """
    from odl.util.utility import pkg_supports
    return pkg_supports(feature, ASTRA_VERSION, ASTRA_FEATURES)


def astra_volume_geometry(reco_space):
    """Create an ASTRA volume geometry from the discretized domain.

    From the ASTRA documentation:

    In all 3D geometries, the coordinate system is defined around the
    reconstruction volume. The center of the reconstruction volume is the
    origin, and the sides of the voxels in the volume have length 1.

    All dimensions in the projection geometries are relative to this unit
    length.


    Parameters
    ----------
    reco_space : `DiscreteLp`
        Discretized space where the reconstruction (volume) lives.
        It must be 2- or 3-dimensional and uniformly discretized.

    Returns
    -------
    astra_geom : dict

    Raises
    ------
    NotImplementedError
        If the cell sizes are not the same in each dimension.
    """
    if not isinstance(reco_space, DiscreteLp):
        raise TypeError('`reco_space` {!r} is not a DiscreteLp instance'
                        ''.format(reco_space))

    if not reco_space.is_uniform:
        raise ValueError('`reco_space` {} is not uniformly discretized')

    vol_shp = reco_space.partition.shape
    vol_min = reco_space.partition.min_pt
    vol_max = reco_space.partition.max_pt

    if reco_space.ndim == 2:
        # ASTRA does in principle support custom minimum and maximum
        # values for the volume extent also in earlier versions, but running
        # the algorithm fails if voxels are non-isotropic.
        if (not reco_space.partition.has_isotropic_cells and
                not astra_supports('anisotropic_voxels_2d')):
            raise NotImplementedError(
                'non-isotropic pixels in 2d volumes not supported by ASTRA '
                'v{}'.format(ASTRA_VERSION))
        # Given a 2D array of shape (x, y), a volume geometry is created as:
        #    astra.create_vol_geom(x, y, y_min, y_max, x_min, x_max)
        # yielding a dictionary:
        #   {'GridRowCount': x,
        #    'GridColCount': y,
        #    'WindowMinX': y_min,
        #    'WindowMaxX': y_max,
        #    'WindowMinY': x_min,
        #    'WindowMaxY': x_max}
        #
        # NOTE: this setting is flipped with respect to x and y. We do this
        # as part of a global rotation of the geometry by -90 degrees, which
        # avoids rotating the data.
        # NOTE: We need to flip the sign of the (ODL) x component since
        # ASTRA seems to move it in the other direction. Not quite clear
        # why.
        vol_geom = astra.create_vol_geom(vol_shp[0], vol_shp[1],
                                         vol_min[1], vol_max[1],
                                         -vol_max[0], -vol_min[0])
    elif reco_space.ndim == 3:
        # Not supported in all versions of ASTRA
        if (not reco_space.partition.has_isotropic_cells and
                not astra_supports('anisotropic_voxels_3d')):
            raise NotImplementedError(
                'non-isotropic voxels in 3d volumes not supported by ASTRA '
                'v{}'.format(ASTRA_VERSION))
        # Given a 3D array of shape (x, y, z), a volume geometry is created as:
        #    astra.create_vol_geom(y, z, x, z_min, z_max, y_min, y_max,
        #                          x_min, x_max),
        # yielding a dictionary:
        #   {'GridColCount': z,
        #    'GridRowCount': y,
        #    'GridSliceCount': x,
        #    'WindowMinX': z_max,
        #    'WindowMaxX': z_max,
        #    'WindowMinY': y_min,
        #    'WindowMaxY': y_min,
        #    'WindowMinZ': x_min,
        #    'WindowMaxZ': x_min}
        vol_geom = astra.create_vol_geom(vol_shp[1], vol_shp[2], vol_shp[0],
                                         vol_min[2], vol_max[2],
                                         vol_min[1], vol_max[1],
                                         vol_min[0], vol_max[0])
    else:
        raise ValueError('{}-dimensional volume geometries not supported '
                         'by ASTRA'.format(reco_space.ndim))
    return vol_geom


def astra_conebeam_3d_geom_to_vec(geometry):
    """Create vectors for ASTRA projection geometries from ODL geometry.

    The 3D vectors are used to create an ASTRA projection geometry for
    cone beam geometries, see ``'cone_vec'`` in the
    `ASTRA projection geometry documentation`_.

    Each row of the returned vectors corresponds to a single projection
    and consists of ::

        (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ)

    with

        - ``src``: the ray source position
        - ``d``  : the center of the detector
        - ``u``  : the vector from detector pixel ``(0,0)`` to ``(0,1)``
        - ``v``  : the vector from detector pixel ``(0,0)`` to ``(1,0)``

    Parameters
    ----------
    geometry : `Geometry`
        ODL projection geometry from which to create the ASTRA geometry.

    Returns
    -------
    vectors : `numpy.ndarray`
        Array of shape ``(num_angles, 12)`` containing the vectors.

    References
    ----------
    .. _ASTRA projection geometry documentation:
       http://www.astra-toolbox.com/docs/geom3d.html#projection-geometries
    """
    angles = geometry.angles
    vectors = np.zeros((angles.size, 12))

    # Source position
    vectors[:, 0:3] = geometry.src_position(angles)

    # Center of detector in 3D space
    mid_pt = geometry.det_params.mid_pt
    vectors[:, 3:6] = geometry.det_point_position(angles, mid_pt)

    # Vectors from detector pixel (0, 0) to (1, 0) and (0, 0) to (0, 1)
    # `det_axes` gives shape (N, 2, 3), swap to get (2, N, 3)
    det_axes = moveaxis(geometry.det_axes(angles), -2, 0)
    px_sizes = geometry.det_partition.cell_sides
    # Swap detector axes to have better memory layout in  projection data.
    # ASTRA produces `(v, theta, u)` layout, and to map to ODL layout
    # `(theta, u, v)` a complete roll must be performed, which is the
    # worst case (compeltely discontiguous).
    # Instead we swap `u` and `v`, resulting in the effective ASTRA result
    # `(u, theta, v)`. Here we only need to swap axes 0 and 1, which
    # keeps at least contiguous blocks in `v`.
    vectors[:, 9:12] = det_axes[0] * px_sizes[0]
    vectors[:, 6:9] = det_axes[1] * px_sizes[1]

    # ASTRA has (z, y, x) axis convention, in contrast to (x, y, z) in ODL,
    # so we need to adapt to this by changing the order.
    newind = []
    for i in range(4):
        newind += [2 + 3 * i, 1 + 3 * i, 0 + 3 * i]
    vectors = vectors[:, newind]

    return vectors


def astra_conebeam_2d_geom_to_vec(geometry):
    """Create vectors for ASTRA projection geometries from ODL geometry.

    The 2D vectors are used to create an ASTRA projection geometry for
    fan beam geometries, see ``'fanflat_vec'`` in the
    `ASTRA projection geometry documentation`_.

    Each row of the returned vectors corresponds to a single projection
    and consists of ::

        (srcX, srcY, dX, dY, uX, uY)

    with

        - ``src``: the ray source position
        - ``d``  : the center of the detector
        - ``u``  : the vector from detector pixel 0 to 1

    Parameters
    ----------
    geometry : `Geometry`
        ODL projection geometry from which to create the ASTRA geometry.

    Returns
    -------
    vectors : `numpy.ndarray`
        Array of shape ``(num_angles, 6)`` containing the vectors.

    References
    ----------
    .. _ASTRA projection geometry documentation:
       http://www.astra-toolbox.com/docs/geom2d.html#projection-geometries
    """
    # Instead of rotating the data by 90 degrees counter-clockwise,
    # we subtract pi/2 from the geometry angles, thereby rotating the
    # geometry by 90 degrees clockwise
    rot_minus_90 = euler_matrix(-np.pi / 2)
    angles = geometry.angles
    vectors = np.zeros((angles.size, 6))

    # Source position
    src_pos = geometry.src_position(angles)
    vectors[:, 0:2] = rot_minus_90.dot(src_pos.T).T  # dot along 2nd axis

    # Center of detector
    mid_pt = geometry.det_params.mid_pt
    # Need to cast `mid_pt` to float since otherwise the empty axis is
    # not removed
    centers = geometry.det_point_position(angles, float(mid_pt))
    vectors[:, 2:4] = rot_minus_90.dot(centers.T).T

    # Vector from detector pixel 0 to 1
    det_axis = rot_minus_90.dot(geometry.det_axis(angles).T).T
    px_size = geometry.det_partition.cell_sides[0]
    vectors[:, 4:6] = det_axis * px_size

    return vectors


def astra_parallel_3d_geom_to_vec(geometry):
    """Create vectors for ASTRA projection geometries from ODL geometry.

    The 3D vectors are used to create an ASTRA projection geometry for
    parallel beam geometries, see ``'parallel3d_vec'`` in the
    `ASTRA projection geometry documentation`_.

    Each row of the returned vectors corresponds to a single projection
    and consists of ::

        (rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ)

    with

        - ``ray``: the ray direction
        - ``d``  : the center of the detector
        - ``u``  : the vector from detector pixel ``(0,0)`` to ``(0,1)``
        - ``v``  : the vector from detector pixel ``(0,0)`` to ``(1,0)``

    Parameters
    ----------
    geometry : `Geometry`
        ODL projection geometry from which to create the ASTRA geometry.

    Returns
    -------
    vectors : `numpy.ndarray`
        Array of shape ``(num_angles, 12)`` containing the vectors.

    References
    ----------
    .. _ASTRA projection geometry documentation:
       http://www.astra-toolbox.com/docs/geom3d.html#projection-geometries
    """
    angles = geometry.angles
    mid_pt = geometry.det_params.mid_pt

    vectors = np.zeros((angles.shape[-1], 12))

    # Ray direction = -(detector-to-source normal vector)
    vectors[:, 0:3] = -geometry.det_to_src(angles, mid_pt)

    # Center of the detector in 3D space
    vectors[:, 3:6] = geometry.det_point_position(angles, mid_pt)

    # Vectors from detector pixel (0, 0) to (1, 0) and (0, 0) to (0, 1)
    # `det_axes` gives shape (N, 2, 3), swap to get (2, N, 3)
    det_axes = moveaxis(geometry.det_axes(angles), -2, 0)
    px_sizes = geometry.det_partition.cell_sides
    # Swap detector axes to have better memory layout in  projection data.
    # ASTRA produces `(v, theta, u)` layout, and to map to ODL layout
    # `(theta, u, v)` a complete roll must be performed, which is the
    # worst case (compeltely discontiguous).
    # Instead we swap `u` and `v`, resulting in the effective ASTRA result
    # `(u, theta, v)`. Here we only need to swap axes 0 and 1, which
    # keeps at least contiguous blocks in `v`.
    vectors[:, 9:12] = det_axes[0] * px_sizes[0]
    vectors[:, 6:9] = det_axes[1] * px_sizes[1]

    # ASTRA has (z, y, x) axis convention, in contrast to (x, y, z) in ODL,
    # so we need to adapt to this by changing the order.
    new_ind = []
    for i in range(4):
        new_ind += [2 + 3 * i, 1 + 3 * i, 0 + 3 * i]
    vectors = vectors[:, new_ind]
    return vectors


def astra_projection_geometry(geometry):
    """Create an ASTRA projection geometry from an ODL geometry object.

    As of ASTRA version 1.7, the length values are not required any more to be
    rescaled for 3D geometries and non-unit (but isotropic) voxel sizes.

    Parameters
    ----------
    geometry : `Geometry`
        ODL projection geometry from which to create the ASTRA geometry.

    Returns
    -------
    proj_geom : dict
        Dictionary defining the ASTRA projection geometry.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError('`geometry` {!r} is not a `Geometry` instance'
                        ''.format(geometry))

    if 'astra' in geometry.implementation_cache:
        # Shortcut, reuse already computed value.
        return geometry.implementation_cache['astra']

    if not geometry.det_partition.is_uniform:
        raise ValueError('non-uniform detector sampling is not supported')

    if (isinstance(geometry, ParallelBeamGeometry) and
            isinstance(geometry.detector, (Flat1dDetector, Flat2dDetector)) and
            geometry.ndim == 2):
        # TODO(kohr-h): change to parallel_vec when available
        det_width = geometry.det_partition.cell_sides[0]
        det_count = geometry.detector.size
        # Instead of rotating the data by 90 degrees counter-clockwise,
        # we subtract pi/2 from the geometry angles, thereby rotating the
        # geometry by 90 degrees clockwise
        angles = geometry.angles - np.pi / 2
        proj_geom = astra.create_proj_geom('parallel', det_width, det_count,
                                           angles)

    elif (isinstance(geometry, DivergentBeamGeometry) and
          isinstance(geometry.detector, (Flat1dDetector, Flat2dDetector)) and
          geometry.ndim == 2):
        det_count = geometry.detector.size
        vec = astra_conebeam_2d_geom_to_vec(geometry)
        proj_geom = astra.create_proj_geom('fanflat_vec', det_count, vec)

    elif (isinstance(geometry, ParallelBeamGeometry) and
          isinstance(geometry.detector, (Flat1dDetector, Flat2dDetector)) and
          geometry.ndim == 3):
        # Swap detector axes (see astra_*_3d_to_vec)
        det_row_count = geometry.det_partition.shape[0]
        det_col_count = geometry.det_partition.shape[1]
        vec = astra_parallel_3d_geom_to_vec(geometry)
        proj_geom = astra.create_proj_geom('parallel3d_vec', det_row_count,
                                           det_col_count, vec)

    elif (isinstance(geometry, DivergentBeamGeometry) and
          isinstance(geometry.detector, (Flat1dDetector, Flat2dDetector)) and
          geometry.ndim == 3):
        # Swap detector axes (see astra_*_3d_to_vec)
        det_row_count = geometry.det_partition.shape[0]
        det_col_count = geometry.det_partition.shape[1]
        vec = astra_conebeam_3d_geom_to_vec(geometry)
        proj_geom = astra.create_proj_geom('cone_vec', det_row_count,
                                           det_col_count, vec)
    else:
        raise NotImplementedError('unknown ASTRA geometry type {!r}'
                                  ''.format(geometry))

    if 'astra' not in geometry.implementation_cache:
        # Save computed value for later
        geometry.implementation_cache['astra'] = proj_geom

    return proj_geom


def astra_data(astra_geom, datatype, data=None, ndim=2, allow_copy=False):
    """Create an ASTRA data object.

    Parameters
    ----------
    astra_geom : dict
        ASTRA geometry object for the data creator, must correspond to the
        given ``datatype``.
    datatype : {'volume', 'projection'}
        Type of the data container.
    data : `DiscreteLpElement` or `numpy.ndarray`, optional
        Data for the initialization of the data object. If ``None``,
        an ASTRA data object filled with zeros is created.
    ndim : {2, 3}, optional
        Dimension of the data. If ``data`` is provided, this parameter
        has no effect.
    allow_copy : `bool`, optional
        If ``True``, allow copying of ``data``. This means that anything
        written by ASTRA to the returned object will not be written to
        ``data``.

    Returns
    -------
    id : int
        Handle for the new ASTRA internal data object.
    """
    if data is not None:
        if isinstance(data, (DiscreteLpElement, np.ndarray)):
            ndim = data.ndim
        else:
            raise TypeError('`data` {!r} is neither DiscreteLpElement '
                            'instance nor a `numpy.ndarray`'.format(data))
    else:
        ndim = int(ndim)

    if datatype == 'volume':
        astra_dtype_str = '-vol'
    elif datatype == 'projection':
        astra_dtype_str = '-sino'
    else:
        raise ValueError('`datatype` {!r} not understood'.format(datatype))

    # Get the functions from the correct module
    if ndim == 2:
        link = astra.data2d.link
        create = astra.data2d.create
    elif ndim == 3:
        link = astra.data3d.link
        create = astra.data3d.create
    else:
        raise ValueError('{}-dimensional data not supported'
                         ''.format(ndim))

    # ASTRA checks if data is c-contiguous and aligned
    if data is not None:
        if allow_copy:
            data_array = np.asarray(data, dtype='float32', order='C')
            return link(astra_dtype_str, astra_geom, data_array)
        else:
            if isinstance(data, np.ndarray):
                return link(astra_dtype_str, astra_geom, data)
            elif data.tensor.impl == 'numpy':
                return link(astra_dtype_str, astra_geom, data.asarray())
            else:
                # Something else than NumPy data representation
                raise NotImplementedError('ASTRA supports data wrapping only '
                                          'for `numpy.ndarray` instances, got '
                                          '{!r}'.format(data))
    else:
        return create(astra_dtype_str, astra_geom)


def astra_projector(vol_interp, astra_vol_geom, astra_proj_geom, ndim, impl):
    """Create an ASTRA projector configuration dictionary.

    Parameters
    ----------
    vol_interp : {'nearest', 'linear'}
        Interpolation type of the volume discretization. This determines
        the projection model that is chosen.
    astra_vol_geom : dict
        ASTRA volume geometry.
    astra_proj_geom : dict
        ASTRA projection geometry.
    ndim : {2, 3}
        Number of dimensions of the projector.
    impl : {'cpu', 'cuda'}
        Implementation of the projector.

    Returns
    -------
    proj_id : int
        Handle for the created ASTRA internal projector object.
    """
    if vol_interp not in ('nearest', 'linear'):
        raise ValueError("`vol_interp` '{}' not understood"
                         ''.format(vol_interp))
    impl = str(impl).lower()
    if impl not in ('cpu', 'cuda'):
        raise ValueError("`impl` '{}' not understood"
                         ''.format(impl))

    if 'type' not in astra_proj_geom:
        raise ValueError('invalid projection geometry dict {}'
                         ''.format(astra_proj_geom))
    if ndim == 3 and impl == 'cpu':
        raise ValueError('3D projectors not supported on CPU')

    ndim = int(ndim)

    proj_type = astra_proj_geom['type']
    if proj_type not in ('parallel', 'fanflat', 'fanflat_vec',
                         'parallel3d', 'parallel3d_vec', 'cone', 'cone_vec'):
        raise ValueError('invalid geometry type {!r}'.format(proj_type))

    # Mapping from interpolation type and geometry to ASTRA projector type.
    # "I" means probably mathematically inconsistent. Some projectors are
    # not implemented, e.g. CPU 3d projectors in general.
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

    # GPU algorithms not necessarily require a projector, but will in future
    # releases making the interface more coherent regarding CPU and GPU
    type_map_cuda = {'parallel': 'cuda',  # I
                     'parallel3d': 'cuda3d'}  # I
    type_map_cuda['fanflat'] = type_map_cuda['parallel']
    type_map_cuda['fanflat_vec'] = type_map_cuda['fanflat']
    type_map_cuda['cone'] = type_map_cuda['parallel3d']
    type_map_cuda['parallel3d_vec'] = type_map_cuda['parallel3d']
    type_map_cuda['cone_vec'] = type_map_cuda['cone']

    # create config dict
    proj_cfg = {}
    if impl == 'cpu':
        proj_cfg['type'] = type_map_cpu[proj_type][vol_interp]
    else:  # impl == 'cuda'
        proj_cfg['type'] = type_map_cuda[proj_type]
    proj_cfg['VolumeGeometry'] = astra_vol_geom
    proj_cfg['ProjectionGeometry'] = astra_proj_geom
    proj_cfg['options'] = {}

    # Add the hacky 1/r^2 weighting exposed in intermediate versions of
    # ASTRA
    if (proj_type in ('cone', 'cone_vec') and
            astra_supports('cone3d_hacky_density_weighting')):
        proj_cfg['options']['DensityWeighting'] = True

    if ndim == 2:
        return astra.projector.create(proj_cfg)
    else:
        return astra.projector3d.create(proj_cfg)


def astra_algorithm(direction, ndim, vol_id, sino_id, proj_id, impl):
    """Create an ASTRA algorithm object to run the projector.

    Parameters
    ----------
    direction : {'forward', 'backward'}
        For ``'forward'``, apply the forward projection, for ``'backward'``
        the backprojection.
    ndim : {2, 3}
        Number of dimensions of the projector.
    vol_id : int
        Handle for the ASTRA volume data object.
    sino_id : int
        Handle for the ASTRA projection data object.
    proj_id : int
        Handle for the ASTRA projector object.
    impl : {'cpu', 'cuda'}
        Implementation of the projector.

    Returns
    -------
    id : int
        Handle for the created ASTRA internal algorithm object.
    """
    if direction not in ('forward', 'backward'):
        raise ValueError("`direction` '{}' not understood".format(direction))
    if ndim not in (2, 3):
        raise ValueError('{}-dimensional projectors not supported'
                         ''.format(ndim))
    if impl not in ('cpu', 'cuda'):
        raise ValueError("`impl` type '{}' not understood"
                         ''.format(impl))
    if ndim == 3 and impl == 'cpu':
        raise NotImplementedError(
            '3d algorithms for CPU not supported by ASTRA')
    if proj_id is None and impl == 'cpu':
        raise ValueError("'cpu' implementation requires projector ID")

    algo_map = {'forward': {2: {'cpu': 'FP', 'cuda': 'FP_CUDA'},
                            3: {'cpu': None, 'cuda': 'FP3D_CUDA'}},
                'backward': {2: {'cpu': 'BP', 'cuda': 'BP_CUDA'},
                             3: {'cpu': None, 'cuda': 'BP3D_CUDA'}}}

    algo_cfg = {'type': algo_map[direction][ndim][impl],
                'ProjectorId': proj_id,
                'ProjectionDataId': sino_id}
    if direction is 'forward':
        algo_cfg['VolumeDataId'] = vol_id
    else:
        algo_cfg['ReconstructionDataId'] = vol_id

    # Create ASTRA algorithm object
    return astra.algorithm.create(algo_cfg)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
