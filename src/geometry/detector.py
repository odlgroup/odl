# -*- coding: utf-8 -*-
"""
detector.py -- detectors in tomography

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

from __future__ import unicode_literals, print_function, division, absolute_import
from future.builtins import super
from future import standard_library
standard_library.install_aliases()

import numpy as np

from RL.geometry.labcomp import LabComponent


class Detector(LabComponent):

    def __init__(self, support, location, **kwargs):
        self._support = support
        super().__init__(location, **kwargs)

    @property
    def support(self):
        return self._support


class FlatDetectorArray(Detector):

    def __init__(self, grid, location, **kwargs):
        super().__init__(grid, location, **kwargs)

    @property
    def grid(self):
        return self.support

    @property
    def pixel_size(self):
        return self.grid.spacing

    @property
    def pixel_area(self):
        return np.prod(self.grid.spacing)


class SphericalDetectorArray(Detector):

    def __init__(self, grid, location, **kwargs):
        super().__init__(grid, location, **kwargs)

    @property
    def grid(self):
        return self.support

    @property
    def pixel_size(self):
        return self.grid.spacing


class PointDetectors(Detector):

    # TODO: implement point colletion type support
    def __init__(self, points, **kwargs):
        return NotImplementedError
        super().__init__(points, None, **kwargs)

    @property
    def points(self):
        return self.support
