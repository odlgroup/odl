# -*- coding: utf-8 -*-
"""
curve.py -- curves in n-dimensional space

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

from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from builtins import int, super, object
from future import standard_library
standard_library.install_aliases()

from math import sin, cos
import numpy as np
from scipy.linalg import norm
from functools import partial
from copy import deepcopy

from RL.utility.utility import errfmt, is_rotation_matrix


class Curve(object):

    def __init__(self, curve_fun, stops=None, **kwargs):

        self._curve_fun = curve_fun
        self._stops = None if stops is None else np.array(stops)

        start = kwargs.get('start', None)
        end = kwargs.get('end', None)
        step = kwargs.get('step', None)

        self._dim = 0
        if stops is not None:
            self._dim = len(curve_fun(stops[0]))
        elif start is not None:
            self._dim = len(curve_fun(start))
        elif end is not None:
            self._dim = len(curve_fun(end))
        else:  # Try some range of values
            for k in np.arange(-6, 6):
                pos = 10**k
                try:
                    self._dim = len(curve_fun(pos))
                    break
                except (ValueError, TypeError):
                    pass
                pos = -10**k
                try:
                    self._dim = len(curve_fun(pos))
                    break
                except (ValueError, TypeError):
                    pass

        if self._dim == 0:
            raise ValueError("unable to determine curve dimension.")

        self._start = start
        self._end = end
        self._step = step

        self._tangent = kwargs.get('tangent', None)
        self._normal = kwargs.get('normal', None)
        if self.dim != 3:
            self._binormal = None
        else:
            self._binormal = kwargs.get('binormal', None)

        axes_map = kwargs.get('axes_map', None)
        if axes_map == 'tripod':
            if self._dim not in (2, 3):
                raise ValueError(errfmt("""\
                tripod can only be used in 2- or 3-dimensional curves."""))
            if self._tangent is None:
                raise ValueError("`tangent` must be defined to use tripod.")
            if self._normal is None:
                raise ValueError("`normal` must be defined to use tripod.")

            def axes_map_2_(tang, nor, time):
                return (-nor(time), tang(time))

            def axes_map_3_(tang, nor, binor, time):
                tangent = tang(time)
                normal = nor(time)
                if binor is not None:
                    binormal = binor(time)
                else:
                    binormal = np.cross(tangent, normal)
                return (-normal, tangent, binormal)

            if self._dim == 2:
                self._axes_map = partial(axes_map_2_, self._tangent,
                                         self._normal)
            else:
                self._axes_map = partial(axes_map_3_, self._tangent,
                                         self._normal, self._binormal)

        else:
            self._axes_map = axes_map

        if axes_map is not None:
            self._coord_sys = self._axes_map
        else:
            self._coord_sys = lambda x: tuple(col for col in np.eye(self._dim))

    @property
    def curve_fun(self):
        return self._curve_fun

    @property
    def start(self):
        if self._stops is not None:
            return self.stops[0]
        else:
            return -np.inf if self._start is None else self._start

    @property
    def stops(self):
        if self._stops is not None:
            return self._stops
        elif (self.start == -np.inf or self.end == np.inf or
              self._step is None):
            return None
        else:
            return np.array(list(self.iter_stops))

    @stops.setter
    def stops(self, new_stops):
        new_stops = np.array(new_stops)
        self._stops = new_stops

    @property
    def iter_stops(self):
        if self._stops is not None:
            def iter_stops_():
                return iter(self._stops)
        elif self.start is None or self.step is None:
            raise ValueError(errfmt("""\
            Either `stops` or `start` and `step` must be defined."""))
        else:
            def iter_stops_():
                stop = self.start
                while True:
                    yield stop
                    if stop >= self.end:
                        break
                    stop += self.step

        return iter_stops_

    @property
    def end(self):
        if self._stops is not None:
            return self.stops[-1]
        else:
            return np.inf if self._end is None else self._end

    @property
    def step(self):
        return self._step

    @property
    def dim(self):
        return self._dim

    @property
    def tangent(self):
        return self._tangent

    @property
    def normal(self):
        return self._normal

    @property
    def binormal(self):
        return self._binormal

    @property
    def startpos(self):
        if self._start is not None:
            return self.curve_fun(self.start)
        else:
            return None

    @property
    def endpos(self):
        if self._end is not None:
            return self.curve_fun(self.end)
        else:
            return None

    @property
    def axes_map(self):
        return self._axes_map

    @property
    def coord_sys(self):
        return self._coord_sys

    @property
    def start_coord_sys(self):
        if self._start is not None:
            return self.coord_sys(self.start)
        elif self.axes_map is None:
            return self.coord_sys(0)  # constantly standard system
        else:
            return None

    def coord_sys_is_ons_rhs(self, times=None, show_diff=False):

        if times is not None:
            if np.asarray(times).ndim >= 1:
                times = np.asarray(times).flatten()
            else:
                times = (times,)
        else:
            times = self.iter_stops()

        for time in times:
            coord_sys = self.coord_sys(time)
            coord_matrix = np.matrix(coord_sys)
            if not is_rotation_matrix(coord_matrix, show_diff=show_diff):
                print('at time: ', time)
                return False
        return True

    def __call__(self, time):
        return self.curve_fun(time)

    def copy(self):
        return deepcopy(self)


def curve(obj, **kwargs):

    if isinstance(obj, Curve):
        return obj.copy()
    else:
        raise TypeError(errfmt("""\
        {!r} cannot be converted to {!r}.""".format(type(obj), Curve)))


class FixedPoint(Curve):

    def __init__(self, location, axes_map=None):

        location = np.array(location)
        curve_fun = lambda x: location
        super().__init__(curve_fun, axes_map=axes_map)

    @property
    def location(self):
        return self.curve_fun(0)


class Circle2D(Curve):

    def __init__(self, radius, angle_shift=0., **kwargs):

        def circle2d_map_(rad, start_ang, ang):
            ang += start_ang
            return np.array((cos(ang), sin(ang))) * rad

        def tangent_map_(start_ang, ang):
            ang += start_ang
            return np.array((-sin(ang), cos(ang)))

        def normal_map_(start_ang, ang):
            return -circle2d_map_(1., start_ang, ang)

        radius = float(radius)
        self._radius = radius

        start = kwargs.get('start', None)
        angles = kwargs.get('angles', None)
        if angles is not None:
            stops = np.array(angles)
            if start is not None:
                stops -= stops[0] - start
        else:
            stops = None

        self._angles = angles

        circle2d_map = partial(circle2d_map_, radius, angle_shift)
        tangent_map = partial(tangent_map_, angle_shift)
        normal_map = partial(normal_map_, angle_shift)
        super().__init__(circle2d_map, start=start, stops=stops,
                         tangent=tangent_map, normal=normal_map, **kwargs)

    @property
    def radius(self):
        return self._radius

    @property
    def angles(self):
        return self._angles


class Circle3D(Curve):

    def __init__(self, radius, axis, angle_shift=0., **kwargs):

        def circle3d_map_(rad, axis, start_ang, ang):
            ang += start_ang
            return util.axis_rotation((rad, 0, 0), axis, ang)

        def tangent_map_(axis, start_ang, ang):
            return np.cross(axis, circle3d_map_(1., axis, start_ang, ang))

        def normal_map_(axis, start_ang, ang):
            return -circle3d_map_(1., axis, start_ang, ang)

        radius = float(radius)
        self._radius = radius

        try:
            axis = int(axis)
            if axis >= 3:
                raise ValueError("`axis` must be 0, 1 or 2.")
            axis = np.eye(3)[axis]
        except TypeError:
            axis = np.array(axis)
            axis = axis / norm(axis, 2)

        self._axis = axis[:]

        start = kwargs.get('start', None)
        angles = kwargs.get('angles', None)
        if angles is not None:
            stops = np.array(angles)
            if start is not None:
                stops -= stops[0] - start
        else:
            stops = None

        circle3d_map = partial(circle3d_map_, radius, axis, angle_shift)
        tangent_map = partial(tangent_map_, axis, angle_shift)
        normal_map = partial(normal_map_, axis, angle_shift)
        binormal_map = lambda x: axis
        super().__init__(circle3d_map, start=start, stops=stops,
                         tangent=tangent_map, normal=normal_map,
                         binormal=binormal_map, **kwargs)

    @property
    def radius(self):
        return self._radius

    @property
    def angles(self):
        return self._angles

    @property
    def axis(self):
        return self._axis
