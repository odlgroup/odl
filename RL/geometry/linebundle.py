# Copyright 2014, 2015 Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


class LineBundle(object):
    # ABSTRACT...

    def position(self, *pos):
        pass

    def lineDirection(self, *pos):
        pass


class SourceOnCurve(object):
    def __call__(self, t):
        return np.array([sin(t), cos(t)])


class FlatPanelDetector(object):
    def __call__(self,x,y):
        return np.array([x,y])

class PointSourceLineBundle(LineBundle):
    def __init__(self,sourceFunc,detectorFunc):
        self.sourcePos = sourcePos
        self.detectorFunc = detectorFunc

    def position(self, t, *pos):
        return self.sourcePos(t)

    def lineDirection(self, t, *pos):
        return (self.detectorFunc(t,*pos)-self.sourcePos).normalize()

source = SourceOnCurve()
fpd = FlatPanelDetector()
geometry = PointSourceLineBundle(source,fpd)

