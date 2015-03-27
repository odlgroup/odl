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

import numpy as np

import RL.operator.functional as FUN
from RL.space.measure import *
from RL.space.space import *
from RL.space.defaultSpaces import *

#Example of a space:
class FunctionSpace(Algebra):
    """The space of functions on some domain
    """

    def __init__(self, domain):
        if not isinstance(domain, AbstractSet): 
            raise TypeError("domain ({}) is not a set".format(domain))

        self.domain = domain
        self._field = RealNumbers()

    def linCombImpl(self, a, x, b, y):
        return a*x + b*y #Use operator overloading

    def multiplyImpl(self, x, y):
        return self.makeVector(lambda *args: x(*args)*y(*args))

    @property
    def field(self):
        return self._field

    @property
    def dimension(self):
        raise NotImplementedError("TODO: infinite")
    
    def equals(self, other):
        return isinstance(other, FunctionSpace) and self.domain == other.domain

    def zero(self):
        return self.makeVector(lambda *args: 0)

    def makeVector(self, *args, **kwargs):
        return FunctionSpace.Vector(self, *args, **kwargs)

    class Vector(HilbertSpace.Vector, Algebra.Vector, FUN.Functional):
        """ L2 Vectors are functions from the domain
        """

        def __init__(self, space, function):
            HilbertSpace.Vector.__init__(self, space)
            self.function = function

        def applyImpl(self, rhs):
            return self.function(rhs)

        def assign(self, other):     
            self.function = other.function

        @property
        def domain(self):           
            return self.space.domain
        
        @property
        def range(self):
            return self.space.field


class L2(FunctionSpace, HilbertSpace):
    """The space of square integrable functions on some domain
    """

    def __init__(self, domain):
        FunctionSpace.__init__(self, domain)

    def innerImpl(self, v1, v2):
        raise NotImplementedError("You cannot calculate inner products in non-discretized spaces")

    def equals(self, other):
        return isinstance(other, L2) and FunctionSpace.equals(self, other)


class UniformDiscretization(EuclidianSpace, Discretization):
    """ Uniform discretization of an interval
    Represents vectors by RN elements
    Uses trapezoid method for integration
    """

    def __init__(self, parent, n):
        if not isinstance(parent.domain, Interval):
            raise NotImplementedError("Can only discretize intervals")

        self.parent = parent
        EuclidianSpace.__init__(self, n)

    def innerImpl(self, v1, v2): #Delegate to main space
        dx = (self.parent.domain.end-self.parent.domain.begin)/(self.n-1)
        return EuclidianSpace.innerImpl(self, v1, v2)*dx

    def zero(self):
        return self.makeVector(np.zeros(self.n), copy=False)

    def empty(self):
        return self.makeVector(np.empty(self.n), copy=False)
    
    def equals(self, other):
        return isinstance(other, UniformDiscretization) and EuclidianSpace.equals(self, other)

    def makeVector(self, *args, **kwargs):
        return UniformDiscretization.Vector(self, *args, **kwargs)

    def integrate(self, f):
        return float(np.trapz(f.values, dx=(self.parent.domain.end-self.parent.domain.begin)/(self.n-1)))

    def points(self):
        return np.linspace(self.parent.domain.begin, self.parent.domain.end, self.n)

    class Vector(EuclidianSpace.Vector):
        def __init__(self, space, *args, **kwargs):
            if len(args) == 1 and isinstance(args[0], L2.Vector) and args[0].space == space.parent:
                data = EuclidianSpace.Vector.__init__(self, space, [args[0](point) for point in space.points()], copy=False)
            else:
                data = EuclidianSpace.Vector.__init__(self, space, *args, **kwargs)
