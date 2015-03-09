# -*- coding: utf-8 -*-
"""
xray.py -- helper functions for X-ray tomography

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from math import pi

from RL.datamodel import ugrid as ug
from RL.datamodel import gfunc as gf
from RL.geometry import curve as crv
from RL.geometry import source as src
from RL.geometry import sample as spl
from RL.geometry import detector as det
from RL.geometry import geometry as geo
from RL.operator.projector import Projector, BackProjector
from RL.utility.utility import InputValidationError, errfmt


def xray_ct_parallel_3d_projector(geometry, backend='astra_cuda'):

    # FIXME: this construction is probably only temporary. Think properly
    # about how users would construct projectors
    return Projector(xray_ct_parallel_projection_3d, geometry,
                     backend=backend)


def xray_ct_parallel_3d_backprojector(geometry, backend='astra_cuda'):

    # FIXME: this construction is probably only temporary. Think properly
    # about how users would construct projectors
    return BackProjector(xray_ct_parallel_backprojection_3d, geometry,
                         backend=backend)


def xray_ct_parallel_geom_3d(spl_grid, det_grid, axis, angles=None,
                             rotating_sample=True, **kwargs):
    """
    Create a 3D parallel beam geometry for X-ray CT with a flat detector.

    Parameters
    ----------
    spl_grid: ugrid.Ugrid
        3D grid for the sample domain
    det_grid: ugrid.Ugrid
        2D grid for the detector domain
    axis: int or array-like
        rotation axis; if integer, interpreted as corresponding standard
        unit vecor
    angles: array-like, optional
        specifies the rotation angles
    rotating_sample: boolean, optional
        if True, the sample rotates, otherwise the source-detector system

    Keyword arguments
    -----------------
    init_rotation: matrix-like or float
        initial rotation of the sample; if float, ???

    Returns
    -------
    out: geometry.Geometry
        the new parallel beam geometry
    """

    spl_grid = ug.ugrid(spl_grid)
    det_grid = ug.ugrid(det_grid)
    if not spl_grid.dim == 3:
        raise InputValidationError(spl_grid.dim, 3, 'spl_grid.dim')
    if not det_grid.dim == 2:
        raise InputValidationError(det_grid.dim, 2, 'det_grid.dim')

    if angles is not None:
        angles = np.array(angles)

    init_rotation = kwargs.get('init_rotation', None)

    if rotating_sample:
        # TODO: make axis between source and detector flexible; now: -x axis
        direction = (1., 0., 0.)
        src_loc = (-1., 0., 0.)
        source = src.ParallelRaySource(direction, src_loc)
        sample = spl.RotatingGridSample(spl_grid, axis, init_rotation,
                                        angles=angles, **kwargs)
        det_loc = (1., 0., 0.)
        detector = det.FlatDetectorArray(det_grid, det_loc)
    else:
        src_circle = crv.Circle3D(1., axis, angles=angles, axes_map='tripod')
        source = src.ParallelRaySource((1, 0, 0), src_circle)

        sample = spl.FixedSample(spl_grid)

        det_circle = crv.Circle3D(1., axis, angle_shift=pi, angles=angles,
                                  axes_map='tripod')
        detector = det.FlatDetectorArray(det_grid, det_circle)
    return geo.Geometry(source, sample, detector)


def xray_ct_parallel_projection_3d(geometry, vol_func, backend='astra_cuda'):

    if backend == 'astra':
        proj_func = _xray_ct_par_fp_3d_astra(geometry, vol_func,
                                             use_cuda=False)
    elif backend == 'astra_cuda':
        proj_func = _xray_ct_par_fp_3d_astra(geometry, vol_func,
                                             use_cuda=True)
    else:
        raise NotImplementedError(errfmt('''\
        Only `astra` and `astra_cuda` backends supported'''))

    return proj_func


def _xray_ct_par_fp_3d_astra(geom, vol, use_cuda=True):

    import astra as at

    print('compute forward projection')

    # FIXME: we assume fixed sample rotating around z axis for now
    # TODO: include shifts (volume and detector)
    # TODO: allow custom detector grid (e.g. only partial projection)

    # Initialize volume geometry and data and wrap it into a data3d object

    # ASTRA uses a different axis labeling. We need to cycle x->y->z->x
    astra_vol = vol.fvals.swapaxes(0, 1).swapaxes(0, 2)
    astra_vol_geom = at.create_vol_geom(vol.shape)
    astra_vol_id = at.data3d.create('-vol', astra_vol_geom, astra_vol)

    # Create the ASTRA algorithm config
    if use_cuda:
        astra_algo_conf = at.astra_dict('FP3D_CUDA')
    else:
        # TODO: slice into 2D forward projections
        raise NotImplementedError('No CPU 3D forward projection available.')

    # Initialize detector geometry

    # Since ASTRA assumes voxel size (1., 1., 1.), some scaling and adaption
    # of tilt angles is necessary

    det_grid = geom.detector.grid
    # FIXME: treat case when no discretization is given
    angles = geom.sample.angles
#    print('old angles: ', astra_angles)

    # FIXME: lots of repetition in the following lines
    # TODO: this should be written without angles, just using direction
    # vectors
    a, b, c = vol.spacing
    print('a, b, c = ', a, b, c)

    # FIXME: assuming z axis tilt
    if a != b:
        astra_angles, px_scaling, factors = _astra_scaling(a, b, angles, 'fp')
        proj_fvals = np.empty((det_grid.shape[0], det_grid.shape[1],
                               len(astra_angles)))

        for i, angle, px_scal, scal_fac in enumerate(zip(astra_angles,
                                                         px_scaling, factors)):

            astra_px_spacing = det_grid.spacing.copy()
            print('[{}] orig det px size: '.format(i), astra_px_spacing)
            astra_px_spacing *= (px_scal, c)
            print('[{}] scaled det px size: '.format(i), astra_px_spacing)

            # ASTRA lables detector axes as 'rows, columns', so we need to swap
            # axes 0 and 1
            # We must project one by one since pixel sizes vary
            astra_proj_geom = at.create_proj_geom('parallel3d',
                                                  astra_px_spacing[0],
                                                  astra_px_spacing[1],
                                                  det_grid.shape[1],
                                                  det_grid.shape[0],
                                                  angle)
            # Some wrapping code
            astra_proj_id = at.data3d.create('-sino', astra_proj_geom)

            # Configure and create the algorithm
            astra_algo_conf['VolumeDataId'] = astra_vol_id
            astra_algo_conf['ProjectionDataId'] = astra_proj_id
            astra_algo_id = at.algorithm.create(astra_algo_conf)

            # Run it and remove afterwards
            at.algorithm.run(astra_algo_id)
            at.algorithm.delete(astra_algo_id)

            # Get the projection data. ASTRA creates an (nrows, 1, ncols)
            # array, so we need to squeeze and swap to get (nx, ny)
            proj_fvals[:, :, i] = at.data3d.get(
                astra_proj_id).squeeze().swapaxes(0, 1)
            proj_fvals[:, :, i] *= scal_fac
    else:
        astra_px_spacing = det_grid.spacing * (a, c)

        # ASTRA lables detector axes as 'rows, columns', so we need to swap
        # axes 0 and 1
        astra_proj_geom = at.create_proj_geom('parallel3d',
                                              astra_px_spacing[0],
                                              astra_px_spacing[1],
                                              det_grid.shape[1],
                                              det_grid.shape[0],
                                              angles)

        # Some wrapping code
        astra_proj_id = at.data3d.create('-sino', astra_proj_geom)

        # Configure and create the algorithm
        astra_algo_conf['VolumeDataId'] = astra_vol_id
        astra_algo_conf['ProjectionDataId'] = astra_proj_id
        astra_algo_id = at.algorithm.create(astra_algo_conf)

        # Run it and remove afterwards
        at.algorithm.run(astra_algo_id)
        at.algorithm.delete(astra_algo_id)

        # Get the projection data. ASTRA creates an (nrows, ntilts, ncols)
        # array, so we need to cycle to the right to get (nx, ny, ntilts)
        proj_fvals = at.data3d.get(astra_proj_id)
        proj_fvals = proj_fvals.swapaxes(1, 2).swapaxes(0, 1)

        proj_fvals *= 1. / a

    # Create the projection grid function and return it
    proj_spacing = np.ones(3)
    proj_spacing[:-1] = det_grid.spacing
    proj_func = gf.Gfunc(proj_fvals, spacing=proj_spacing)

    return proj_func


def _astra_scaling(a, b, angles, op):

    from math import sin, cos, atan2
    from scipy.linalg import norm

    # See the documentation on this issue for details

    fwd_angles = np.empty_like(angles)
    fwd_px_scaling = np.empty_like(angles)
    fwd_scaling_factors = np.empty_like(angles)

    bwd_angles = np.empty_like(angles)
    bwd_px_scaling = np.empty_like(angles)
    bwd_weights = np.empty_like(angles)

    for i, ang in enumerate(angles):
        print('[{}] ang: '.format(i), ang)
        old_dir = np.array((cos(ang), sin(ang)))
        print('[{}] old dir: '.format(i), old_dir)
        scaled_fwd_dir = (1 / a, 1 / b) * old_dir
        print('[{}] scaled fwd dir: '.format(i), scaled_fwd_dir)
        scaled_bwd_dir = (a, b) * old_dir
        print('[{}] scaled bwd dir: '.format(i), scaled_bwd_dir)
        norm_scaled_fwd_dir = norm(scaled_fwd_dir)
        print('[{}] norm fwd: '.format(i), norm_scaled_fwd_dir)
        norm_scaled_bwd_dir = norm(scaled_bwd_dir)
        print('[{}] norm_bwd: '.format(i), norm_scaled_bwd_dir)
        fwd_dir = scaled_fwd_dir / norm_scaled_fwd_dir
        print('[{}] fwd dir: '.format(i), fwd_dir)
        bwd_dir = scaled_bwd_dir / norm_scaled_bwd_dir
        print('[{}] bwd dir: '.format(i), bwd_dir)
        fwd_angles[i] = atan2(fwd_dir[1], fwd_dir[0])
        print('[{}] new fwd ang: '.format(i), fwd_angles[i])
        bwd_angles[i] = atan2(bwd_dir[1], bwd_dir[0])
        print('[{}] new bwd ang: '.format(i), bwd_angles[i])
        fwd_px_scaling[i] = np.dot((1 / b, 1 / a) * old_dir, fwd_dir)
        print('[{}] fwd px scaling: '.format(i), fwd_px_scaling[i])
        bwd_px_scaling[i] = np.dot((b, a) * old_dir, bwd_dir)
        print('[{}] bwd px scaling: '.format(i), bwd_px_scaling[i])
        bwd_weights[i] = 1. / (norm((1. / b, 1. / a) * old_dir) *
                               norm_scaled_bwd_dir)
        print('[{}] bwd weight: '.format(i), bwd_weights[i])

    if op.lower() == 'fp':
        return fwd_angles, fwd_px_scaling, fwd_scaling_factors
    elif op.lower() == 'bp':
        return bwd_angles, bwd_px_scaling, bwd_weights


def xray_ct_parallel_backprojection_3d(geometry, proj_func,
                                       backend='astra_cuda'):

    if backend == 'astra':
        vol_func = _xray_ct_par_bp_3d_astra(geometry, proj_func,
                                            use_cuda=False)
    elif backend == 'astra_cuda':
        vol_func = _xray_ct_par_bp_3d_astra(geometry, proj_func,
                                            use_cuda=True)
    else:
        raise NotImplementedError(errfmt('''\
        Only `astra` and `astra_cuda` backends supported'''))

    return vol_func


def _xray_ct_par_bp_3d_astra(geom, proj_, use_cuda=True):

    import astra as at

    print('compute backprojection')
    # FIXME: we assume fixed sample rotating around z axis for now
    # TODO: include shifts (volume and detector)
    # TODO: allow custom volume grid (e.g. partial backprojection)

    # Initialize volume geometry
    vol_grid = geom.sample.grid
    astra_vol_geom = at.create_vol_geom(vol_grid.shape)
    astra_vol_id = at.data3d.create('-vol', astra_vol_geom)

    # Initialize projection geometry and data

    # ASTRA assumes a (nrows, ntilts, ncols) array, we have (nx, ny, ntilts).
    # We must cycle axes to the left
    astra_proj = proj_.fvals.swapaxes(0, 2).swapaxes(0, 1)
    astra_px_spacing = proj_.spacing.copy()

    # FIXME: treat case when no discretization is given
    angles = geom.sample.angles
    # FIXME: handle case when only one angle is given. This code should go
    # someplace else anyway
    integr_weights = np.empty_like(angles)
    if angles[0] == angles[-1]:
        range_fraction = 1.
    else:
        max_ang = angles[-1] + (angles[-1] - angles[-2])
        print('max angle: ', max_ang)
        min_ang = angles[1] + (angles[1] - angles[0])
        print('min angle: ', min_ang)
        range_fraction = abs(max_ang - min_ang) / pi

    print('range fraction: ', range_fraction)

    integr_weights[1:-1] = (angles[2:] - angles[:-2]) / 2.
    integr_weights[0] = (angles[1] - angles[0]) / 2.
    integr_weights[-1] = (angles[-1] - angles[-2]) / 2.
    integr_weights /= range_fraction

    print('integration weights: ', integr_weights)

    # Create the ASTRA algorithm
    if use_cuda:
        astra_algo_conf = at.astra_dict('BP3D_CUDA')
    else:
        # TODO: slice into 2D forward projections
        raise NotImplementedError('No CPU 3D backprojection available.')

    astra_algo_conf['ReconstructionDataId'] = astra_vol_id

    a, b, c = vol_grid.spacing

    # FIXME: assuming z axis tilt
    if a != b:
        astra_angles, px_scaling, weights = _astra_scaling(a, b, angles, 'bp')
        bp_fvals = np.zeros(vol_grid.shape)

        for i, ang, px_scal, scal_w in enumerate(zip(astra_angles, px_scaling,
                                                     weights)):
            astra_px_spacing *= (px_scal, c)

            # ASTRA assumes (nrows, ncols) on the detector, so we provide
            # (ny, nx)
            astra_proj_geom = at.create_proj_geom('parallel3d',
                                                  astra_px_spacing[0],
                                                  astra_px_spacing[1],
                                                  proj_.shape[1],
                                                  proj_.shape[0],
                                                  ang)
            astra_proj_id = at.data3d.create('-sino', astra_proj_geom,
                                             astra_proj[:, :, i])
            astra_algo_conf['ProjectionDataId'] = astra_proj_id
            astra_algo_id = at.algorithm.create(astra_algo_conf)
            at.algorithm.run(astra_algo_id)
            at.algorithm.delete(astra_algo_id)

            # Get the volume data and cycle back axes to translate from ASTRA
            # axes convention
            astra_bp_fvals = at.data3d.get(astra_vol_id)
            astra_bp_fvals = astra_bp_fvals.swapaxes(0, 2).swapaxes(0, 1)

            bp_fvals += astra_bp_fvals * scal_w * integr_weights[i]
    else:
        astra_px_spacing *= (a, c)

        # ASTRA assumes (nrows, ncols) on the detector, so we provide (ny, nx)
        astra_proj_geom = at.create_proj_geom('parallel3d',
                                              astra_px_spacing[0],
                                              astra_px_spacing[1],
                                              proj_.shape[1],
                                              proj_.shape[0],
                                              angles)
        astra_proj_id = at.data3d.create('-sino', astra_proj_geom, astra_proj)
        astra_algo_conf['ProjectionDataId'] = astra_proj_id
        astra_algo_id = at.algorithm.create(astra_algo_conf)
        at.algorithm.run(astra_algo_id)
        at.algorithm.delete(astra_algo_id)

        # Get the volume data and cycle back axes to translate from ASTRA axes
        # convention
        bp_fvals = at.data3d.get(astra_vol_id)
        bp_fvals = bp_fvals.swapaxes(0, 2).swapaxes(0, 1)

        # FIXME: find out how ASTRA weights in the backprojection

    # Create the volume grid function and return it
    vol_func = gf.Gfunc(bp_fvals, spacing=vol_grid.spacing)

    return vol_func
