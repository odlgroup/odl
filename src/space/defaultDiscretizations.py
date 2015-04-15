# -*- coding: utf-8 -*-
"""
operator.py -- functional analytic operators

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
from future.builtins import object
from future import standard_library
standard_library.install_aliases()

import RL.operator.function as fun 
from RL.space.space import *
from RL.space.functionSpaces import L2
from RL.space.measure import *
from RL.space.set import *
import RLcpp.PyCuda
import numpy as np

def makeDefaultUniformDiscretization(parent, rnimpl):
    RNType = type(rnimpl)
    RNVectortype = RNType.Vector

    class DefaultUniformDiscretization(RNType, Discretization):
        """ Uniform discretization of an interval
            Represents vectors by RN elements
            Uses trapezoid method for integration
        """

        def __init__(self, parent, rn):
            if not isinstance(parent.domain, Interval):
                raise NotImplementedError("Can only discretize intervals")

            if not isinstance(rn, HilbertSpace):
                raise NotImplementedError("RN has to be a hilbert space")

            if not isinstance(rn, Algebra):
                raise NotImplementedError("RN has to be an algebra")

            self.parent = parent
            self.rn = rn
    
        def __eq__(self, other):
            return isinstance(other, DefaultUniformDiscretization) and self.parent.equals(other.parent) and self.rn.equals(other.rn)

        def makeVector(self, *args, **kwargs):
            if len(args) == 1 and isinstance(args[0], L2.Vector):
                return self.makeVector(np.array([args[0](point) for point in self.points()], dtype=np.float))
            else:
                return DefaultUniformDiscretization.Vector(self, *args, **kwargs)

        def integrate(self, vector):
            dx = (self.parent.domain.end-self.parent.domain.begin)/(self.n-1)
            return float(self.rn.sum(vector) * dx)

        def points(self):
            return np.linspace(self.parent.domain.begin, self.parent.domain.end, self.n)

        def __getattr__(self, name):
            return getattr(self.rn, name)

        class Vector(RNVectortype):
            pass

    return DefaultUniformDiscretization(parent, rnimpl)

def makeDefaultPixelDiscretization(parent, rnimpl, cols, rows):
    RNType = type(rnimpl)
    RNVectortype = RNType.Vector

    print(RNType, RNVectortype)

    class DefaultPixelDiscretization(RNType, Discretization):
        """ Uniform discretization of an square
            Represents vectors by RN elements
            Uses sum method for integration
        """

        def __init__(self, parent, rn, cols, rows):
            if not isinstance(parent.domain, Square):
                raise NotImplementedError("Can only discretize Squares")

            if not isinstance(rn, HilbertSpace):
                raise NotImplementedError("RN has to be a hilbert space")

            if not isinstance(rn, Algebra):
                raise NotImplementedError("RN has to be an algebra")

            if not rn.dimension == cols*rows:
                raise NotImplementedError("Dimensions do not match, expected {}x{} = {}, got {}".format(cols,rows,cols*rows,rn.dimension))

            self.parent = parent
            self.cols = cols
            self.rows = rows
            self.rn = rn
    
        def equals(self, other):
            return isinstance(other, DefaultPixelDiscretization) and self.cols == other.cols and self.rows == other.rows and self.rn.equals(other.rn)

        def makeVector(self, *args, **kwargs):
            return DefaultPixelDiscretization.Vector(self, *args, **kwargs)

        def integrate(self, vector):
            dx = (self.parent.domain.end[1]-self.parent.domain.begin[0])/(self.cols-1)
            dy = (self.parent.domain.end[1]-self.parent.domain.begin[1])/(self.rows-1)
            return float(self.rn.sum(vector) * dx * dy)

        def points(self):
            return np.meshgrid(np.linspace(self.parent.domain.begin[0], self.parent.domain.end[0],self.cols),
                               np.linspace(self.parent.domain.begin[1], self.parent.domain.end[1],self.rows))

        def __getattr__(self, name):
            return getattr(self.rn, name)

        class Vector(RNVectortype):
            pass

    return DefaultPixelDiscretization(parent, rnimpl, cols, rows)