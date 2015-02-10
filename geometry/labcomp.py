# -*- coding: utf-8 -*-
"""
labcomp.py -- component in a tomography lab

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
from future import standard_library
standard_library.install_aliases()
from builtins import object

import numpy as np

import curve as crv
from utility import errfmt


class LabComponent(object):

    def __init__(self, location, **kwargs):
        if isinstance(location, crv.Curve):
            self._location = crv.curve(location, **kwargs)
        else:
            try:
                location = np.array(location)
                self._location = crv.FixedPoint(location)
            except TypeError:
                raise TypeError(errfmt("""\
                `location` must either be array-like or a curve."""))

        self._cur_location = self._location.startpos
        self._cur_coord_sys = self._location.start_coord_sys

    @property
    def location(self):
        return self._location

    @property
    def start_location(self):
        return self.location.startpos

    @property
    def coord_sys(self):
        return self.location.coord_sys

    @property
    def cur_location(self):
        return self._cur_location

    @property
    def start_coord_sys(self):
        return self.location.start_coord_sys

    @property
    def cur_coord_sys(self):
        return self._cur_coord_sys

    def totime(self, time):
        self._cur_location = self.location.curve_fun(time)
        self._cur_coord_sys = self.location.coord_sys(time)

    def reset(self):
        self._cur_location = self.startpos
        self._cur_coord_sys = self.location.start_coord_sys
