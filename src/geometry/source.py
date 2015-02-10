# -*- coding: utf-8 -*-
"""
source.py -- (radiation) source in tomography

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
from future import standard_library
standard_library.install_aliases()

import numpy as np
from math import pi

from RL.geometry.labcomp import LabComponent
import RL.utility.utility as util
# from utility import errfmt


class Source(LabComponent):

    def __init__(self, location, **kwargs):
        super().__init__(location, **kwargs)

        self._mask = kwargs.get('mask', None)

        wavenum = kwargs.get('wavenum', None)
        if wavenum is not None and wavenum <= 0.:
            raise ValueError("wavenum must be positive.")
        self._wavenum = wavenum

    @property
    def wavenum(self):
        return self._wavenum

    @property
    def wavelen(self):
        return None if self.wavenum is None else 2 * pi / self.wavenum

    @property
    def mask(self):
        return self._mask


class PointRaySource(Source):

    def __init__(self, location, solid_angle_mask=None, **kwargs):
        super().__init__(location, mask=solid_angle_mask, **kwargs)


class PointWaveSource(Source):

    def __init__(self, wavenum, location, solid_angle_mask=None, **kwargs):
        super().__init__(location, mask=solid_angle_mask, wavenum=wavenum,
                         **kwargs)


class ParallelRaySource(Source):

    def __init__(self, direction, location, line_mask=None, **kwargs):
        super().__init__(location, mask=line_mask, **kwargs)

        # direction either a vector or a map
        try:
            direction = np.array(direction)
            self._direction_map = lambda x: direction
        except TypeError:
            self._direction_map = direction

    @property
    def direction_map(self):
        return self._direction_map

    def direction(self, time, system='lab'):
        if system.lower() == 'lab':
            return util.to_lab_sys(self.direction_map(time),
                                   self.coord_sys(time))
        else:
            return self.direction_map(time)


class PlaneWaveSource(Source):

    def __init__(self, wavenum, direction, location, wavelen=None,
                 aperture=None, **kwargs):
        super().__init__(location, mask=aperture, wavenum=wavenum,
                         wavelen=wavelen, **kwargs)

        try:
            direction = np.array(direction)
            self._direction_map = lambda x: direction
        except TypeError:
            self._direction_map = direction

    @property
    def direction_map(self):
        return self._direction_map

    def direction(self, time, system='local'):
        if system.lower() == 'lab':
            return util.to_lab_sys(self.direction_map(time),
                                   self.coord_sys(time))
        else:
            return self.direction_map(time)


class CoherentElectronSource(Source):

    def __init__(self, em_config):

        super().__init__(wavenum=em_config.wavenum)
        self._em_config = em_config

    @property
    def em_config(self):
        return self._em_config

    @property
    def voltage(self):
        return self.em_config.voltage

    @property
    def energy_spread(self):
        return self.em_config.energy_spread

    @property
    def aperture_angle(self):
        return self.em_config.cond_ap_angle

    @property
    def wavenum(self):
        return self.em_config.wavenum

    def direction(self, time, system='local'):
        if system.lower() == 'lab':
            return util.to_lab_sys((0, 0, 1), self.coord_sys(time))
        else:
            return np.array((0, 0, 1))  # TODO: more generic?
