# -*- coding: utf-8 -*-
"""
detector.py -- detectors in tomography

Copyright 2014 Holger Kohr

This file is part of tomok.

tomok is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tomok is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with tomok.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from future import standard_library
standard_library.install_aliases()

import numpy as np

from labcomp import LabComponent


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
