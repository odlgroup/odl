# -*- coding: utf-8 -*-
"""
tomogeom.py -- acquisition geometries in tomography

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

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

from scipy.linalg import norm


class Geometry(object):

    def __init__(self, source, sample, detector, *args, **kwargs):

        self._source = source
        self._sample = sample
        self._detector = detector

    @property
    def source(self):
        return self._source

    @property
    def sample(self):
        return self._sample

    @property
    def detector(self):
        return self._detector

    def vec_source_to_sample(self, time=None, normalize=False):
        if time is None:
            vector = self.sample.cur_location - self.source.cur_location
        else:
            vector = self.sample.location(time) - self.source.location(time)
        if normalize:
            vector = vector / norm(vector, 2)
        return vector

    def dist_source_sample(self, time=None):
        vector = self.vec_source_to_sample(time)
        return norm(vector, 2)

    def vec_sample_to_detector(self, time=None, normalize=False):
        if time is None:
            vector = self.detector.cur_location - self.sample.cur_location
        else:
            vector = self.detector.location(time) - self.sample.location(time)
        if normalize:
            vector = vector / norm(vector, 2)
        return vector

    def dist_sample_detector(self, time=None):
        vector = self.vec_sample_to_detector(time)
        return norm(vector, 2)

    def vec_source_to_detector(self, time=None, normalize=False):
        if time is None:
            vector = self.detector.cur_location - self.source.cur_location
        else:
            vector = self.detector.location(time) - self.source.location(time)
        if normalize:
            vector = vector / norm(vector, 2)
        return vector

    def dist_source_detector(self, time=None):
        vector = self.vec_source_to_detector(time)
        return norm(vector, 2)

    def totime(self, time):
        self.source.totime(time)
        self.sample.totime(time)
        self.detector.totime(time)

    # TODO: add more features
