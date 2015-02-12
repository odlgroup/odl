# -*- coding: utf-8 -*-
"""
sample.py -- samples in tomography

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
from builtins import super
from builtins import int
from future import standard_library
standard_library.install_aliases()

from math import sin, cos
from functools import partial
import numpy as np
from scipy.linalg import norm

from RL.geometry.labcomp import LabComponent
import RL.geometry.curve as crv
import RL.utility.utility as util
from RL.utility.utility import errfmt, InputValidationError, is_rotation_matrix


class Sample(LabComponent):

    def __init__(self, support, location=None, **kwargs):

        if location is None:
            location = support.ref_point
        super().__init__(location, **kwargs)

    @property
    def support(self):
        return self._support


class FixedSample(Sample):

    def __init__(self, support):

        super().__init__(support)


class FixedGridSample(Sample):

    def __init__(self, grid):

        super().__init__(grid)

    @property
    def grid(self):
        return self.support


class RotatingSample(Sample):

    def __init__(self, support, rot_axis=None, init_rotation=None, angles=None,
                 **kwargs):

        dim = support.dim
        if dim == 2:
            rot_axis = None
            if init_rotation is None:
                start_rot = np.eye(2)
            else:
                try:
                    phi = float(init_rotation)
                    start_rot = np.matrix([[cos(phi), -sin(phi)],
                                           [sin(phi), cos(phi)]])
                except TypeError:
                    start_rot = np.matrix(init_rotation)
                    if not is_rotation_matrix(init_rotation):
                        raise ValueError(errfmt("""\
                        `init_rotation` is neither an angle nor a rotation
                        matrix."""))

            def axes_map_(ang, start_rot):
                rot_matrix = np.matrix([[cos(ang), -sin(ang)],
                                        [sin(ang), cos(ang)]]) * start_rot
                return (np.asarray(rot_matrix[:, 0]).flatten(),
                        np.asarray(rot_matrix[:, 1]).flatten())

            axes_map = partial(axes_map_, start_rot=start_rot)

        elif dim == 3:
            if rot_axis is None:
                raise ValueError("`rot_axis` must be specified for `dim`==3.")
            try:
                rot_axis = int(rot_axis)
                rot_axis = np.asarray(np.eye(3)[rot_axis, :])
            except TypeError:
                rot_axis = np.array(rot_axis)
                rot_axis /= norm(rot_axis, 2)

            if init_rotation is None:
                start_rot = np.eye(3)
            else:
                try:  # init_rotation is a sequence of angles
                    phi, theta, psi = (float(ang) for ang in init_rotation)
                    start_rot = util.euler_matrix(phi, theta, psi)
                except TypeError:
                    start_rot = np.matrix(init_rotation)
                    if not is_rotation_matrix(init_rotation):
                        raise ValueError(errfmt("""\
                        `init_rotation` is neither a list of angles nor a
                        rotation matrix."""))

            def axes_map_(ang, rot_axis, start_rot):
                rot_mat = util.axis_rotation_matrix(rot_axis, ang) * start_rot
                return tuple([np.asarray(rot_mat[:, i]).flatten()
                             for i in (0, 1, 2)])

            axes_map = partial(axes_map_, rot_axis=rot_axis,
                               start_rot=start_rot)

        else:
            raise InputValidationError(dim, '2 or 3', 'dim')

        location = crv.FixedPoint(support.ref_point, axes_map)
        super().__init__(support, location, **kwargs)
        self._rot_axis = rot_axis
        self._init_rotation = start_rot
        self._angles = np.array(angles) if angles is not None else None

    @property
    def rot_axis(self):
        return self._rot_axis

    @property
    def init_rotation(self):
        return self._init_rotation

    @property
    def angles(self):
        return self._angles


class RotatingGridSample(RotatingSample):

    def __init__(self, grid, rot_axis=None, init_rotation=None, angles=None,
                 **kwargs):
        super().__init__(grid, rot_axis, init_rotation, angles, **kwargs)

    @property
    def grid(self):
        return self.support
