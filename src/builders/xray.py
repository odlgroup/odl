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
from RL.geometry import curve as crv
from RL.geometry import source as src
from RL.geometry import sample as spl
from RL.geometry import detector as det
from RL.geometry import geometry as geo
from RL.utility.utility import InputValidationError


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


def xray_projection_map(grid_func, geometry, backend='astra'):

    if backend == 'astra':
        pass


def _xray_proj_map_parbeam_astra(gf, geom):

    pass
