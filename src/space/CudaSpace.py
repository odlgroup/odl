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

from RL.space.space import *
from RL.space.functionSpaces import L2
from RL.space.measure import *
from RL.space.set import *
from RLcpp.PyCuda import *
import numpy as np

class CudaRN(HilbertSpace):
    """The real space R^n
    """

    def __init__(self, n):
        self.n = n
        self._field = RealNumbers()
        self.impl = CudaRNImpl(n)

    def innerImpl(self, x, y):
        return self.impl.inner(x.impl, y.impl)

    def normSqImpl(self, x): #Optimized separately from inner
        return self.impl.normSq(x.impl)
    
    def linCombImpl(self, a, x, b, y):
        self.impl.linComb(a, x.impl, b, y.impl)

    def zero(self):
        return self.makeVector(self.impl.zero())

    def empty(self):
        return self.makeVector(self.impl.empty())

    @property
    def field(self):
        return self._field

    @property
    def dimension(self):
        return self.n

    def equals(self, other):
        return isinstance(other, CudaRN) and self.n == other.n

    def makeVector(self, *args, **kwargs):
        return CudaRN.Vector(self, *args, **kwargs)

    class Vector(HilbertSpace.Vector):
        def __init__(self, space, *args):
            HilbertSpace.Vector.__init__(self, space)
            if isinstance(args[0], CudaRNVectorImpl):
                self.impl = args[0]
            elif isinstance(args[0], CudaRN.Vector): #Move constructor
                self.impl = args[0].impl
            elif isinstance(args[0], np.ndarray):
                self.impl = space.impl.empty()
                self[:] = args[0]
            elif isinstance(args[0], list):
                self.impl = space.impl.empty()
                self[:] = np.array(args[0], dtype=np.float)
            else:
                self.impl = CudaRNVectorImpl(*args)
            
        def __str__(self):
            return "[" + self[:].__str__() + "]"

        def __repr__(self):
            return "CudaRNVector("+self[:].__repr__()+")"

        #Slow get and set, for testing and nothing else!
        def __getitem__(self, index):
            if isinstance(index, slice):
                if index.start is not None and 0 > index.start or index.stop is not None and index.stop > self.space.n:
                    raise IndexError("Out of range")
                
                return self.impl.getSlice(index)
            else:
                if index < 0: #Some negative indices should work (y[-1] is last element)
                    index = self.space.n+index

                if index < 0 or index >= self.space.n:
                    raise IndexError("Out of range")

                return self.impl.__getitem__(index)

        def __setitem__(self, index, value):
            if isinstance(index, slice):
                if index.start is not None and 0 > index.start or index.stop is not None and index.stop > self.space.n:
                    raise IndexError("Out of range")
                
                #Convert value to the correct type
                if not isinstance(value, np.ndarray):
                    value = np.array(value, dtype=np.float)
                elif value.dtype.type is not np.float:
                    value = value.astype(np.float)

                self.impl.setSlice(index, value)
            else:
                if 0 > index or index >= self.space.n:
                    raise IndexError("Out of range")

                self.impl.__setitem__(index, value)


class CudaUniformDiscretization(CudaRN, Discretization):
    """ Uniform discretization of an interval
    Represents vectors by RN elements
    Uses trapezoid method for integration
    """

    def __init__(self, parent, n):
        if not isinstance(parent.domain, Interval):
            raise NotImplementedError("Can only discretize intervals")

        self.parent = parent
        CudaRN.__init__(self, n)

    def zero(self):
        return self.makeVector(self.impl.zero())

    def empty(self):
        return self.makeVector(self.impl.empty())
    
    def __eq__(self, other):
        return isinstance(other, CudaUniformDiscretization) and CudaRN.__eq__(self, other)

    def makeVector(self, *args, **kwargs):
        return CudaUniformDiscretization.Vector(self, *args, **kwargs)

    def integrate(self, vector):
        dx = (self.parent.domain.end-self.parent.domain.begin)/(self.n-1)
        return float(vector.impl.sum() * dx)

    def points(self):
        return np.linspace(self.parent.domain.begin, self.parent.domain.end, self.n)

    class Vector(CudaRN.Vector):
        def __init__(self, space, *args, **kwargs):
            if len(args) == 1 and isinstance(args[0], L2.Vector) and args[0].space == space.parent:
                CudaRN.Vector.__init__(self, space, np.array([args[0](point) for point in space.points()], dtype=np.float))
            else:
                CudaRN.Vector.__init__(self, space, *args, **kwargs)
