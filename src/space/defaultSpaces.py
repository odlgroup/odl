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
from future.builtins import object, zip
from future import standard_library
standard_library.install_aliases()

import numpy as np

from RL.utility.utility import allEqual
from RL.space.space import *
from RL.space.set import *
    
class ProductSpace(HilbertSpace):
    """Product space (X1 x X2 x ... x Xn)
    """

    def __init__(self, *spaces):
        if len(spaces) == 0:
            raise TypeError("Empty product not allowed")
        if not all(spaces[0].field == y.field for y in spaces):
            raise TypeError("All spaces must have the same field")

        self.spaces = spaces
        self._dimension = len(self.spaces)
        self._field = spaces[0].field  #X_n has same field

    def zero(self):
        return self.makeVector(*[space.zero() for space in self.spaces])

    def empty(self):
        return self.makeVector(*[space.empty() for space in self.spaces])
    
    def innerImpl(self, x, y):
        return sum(space.innerImpl(xp, yp) for space, xp, yp in zip(self.spaces, x.parts, y.parts))

    def linCombImpl(self, a, x, b, y):
        for space, xp, yp in zip(self.spaces, x.parts, y.parts):
            space.linCombImpl(a, xp, b, yp)

    @property
    def field(self):
        return self._field

    @property
    def dimension(self):
        return self._dimension

    def equals(self, other):
        return isinstance(other,ProductSpace) and all(x.equals(y) for x, y in zip(self.spaces, other.spaces))
    
    def makeVector(self, *args):
        return ProductSpace.Vector(self, *args)

    def __getitem__(self, index):
        return self.spaces[index]

    def __len(self):
        return self.dimension

    def __str__(self):
        return "ProductSpace(" + ", ".join(str(space) for space in self.spaces) + ")"

    class Vector(HilbertSpace.Vector):
        def __init__(self, space, *args):
            HilbertSpace.Vector.__init__(self, space)

            if not isinstance(args[0], HilbertSpace.Vector): #Delegate constructors
                self.parts = tuple(space.makeVector(arg) for arg, space in zip(args, space.spaces))
            else: #Construct from existing tuple
                if any(part.space != space for part, space in zip(args, space.spaces)):
                    raise TypeError("The spaces of all parts must correspond to this space's parts")

                self.parts = args

        def __getitem__(self,index):
            return self.parts[index]

        def __str__(self):          
            return self.space.__str__() + "::Vector(" + ", ".join(str(part) for part in self.parts) + ")"

        def __repr__(self):         
            return self.space.__repr__() + "::Vector(" + ", ".join(part.__repr__() for part in self.parts) + ")"


class PowerSpace(HilbertSpace):
    """Product space with the same underlying space (X x X x ... x X)
    """

    def __init__(self,underlying_space,dimension):
        if dimension <= 0:
            raise TypeError("Empty or negative product not allowed")

        self.underlying_space = underlying_space
        self._dimension = dimension

    def zero(self):
        return self.makeVector(*[self.underlying_space.zero() for _ in range(self.dimension)])

    def empty(self):
        return self.makeVector(*[self.underlying_space.empty() for _ in range(self.dimension)])
    
    def innerImpl(self, x, y):
        return sum(self.underlying_space.innerImpl(xp, yp) for xp, yp in zip(x.parts, y.parts))

    def linCombImpl(self, a, x, b, y):
        for xp, yp in zip(x.parts, y.parts):
            self.underlying_space.linCombImpl(a, xp, b, yp)

    @property
    def field(self):
        return self.underlying_space.field

    @property
    def dimension(self):
        return self._dimension

    def equals(self, other):
        return isinstance(other, PowerSpace) and self.underlying_space.equals(other.underlying_space) and self.dimension == other.dimension
    
    def makeVector(self, *args):
        return PowerSpace.Vector(self, *args)

    def __getitem__(self, index):
        if index < -self.dimension or index >= self.dimension:
            raise IndexError("Index out of range") 
        return self.underlying_space

    def __len(self):
        return self.dimension

    def __str__(self):
        return "PowerSpace(" + str(self.underlying_space) + ", " + str(self.dimension) + ")"

    class Vector(HilbertSpace.Vector):
        def __init__(self, space, *args):
            HilbertSpace.Vector.__init__(self, space)

            if not isinstance(args[0], HilbertSpace.Vector): #Delegate constructors
                self.parts = tuple(space.makeVector(arg) for arg, space in zip(args, space.spaces))
            else: #Construct from existing tuple
                if len(args) != self.space.dimension:
                    raise TypeError("The dimension of the space must be correct")

                if any(part.space != self.space.underlying_space for part in args):
                    raise TypeError("The spaces of all parts must correspond to this space's parts")

                self.parts = args

        def __getitem__(self, index):
            return self.parts[index]

        def __str__(self):          
            return self.space.__str__() + "::Vector(" + ", ".join(str(part) for part in self.parts) + ")"

        def __repr__(self):         
            return self.space.__repr__() + "::Vector(" + ", ".join(part.__repr__() for part in self.parts) + ")"


class Reals(HilbertSpace):
    """The real numbers
    """

    def __init__(self):
        self._field = RealNumbers()

    def innerImpl(self, x, y):        
        return x.__val__ * y.__val__

    def linCombImpl(self, a, x, b, y):        
        y.__val__ = a*x.__val__ + b*y.__val__

    def empty(self):
        return self.makeVector(0.0)

    @property
    def field(self):
        return self._field

    @property
    def dimension(self):
        return 1
    
    def equals(self, other):
        return isinstance(other, Reals)

    def makeVector(self, value):
        return Reals.Vector(self, value)

    class Vector(HilbertSpace.Vector):
        """Real vectors are floats
        """

        __val__ = None
        def __init__(self, space, v):
            HilbertSpace.Vector.__init__(self, space)
            self.__val__ = v

        def __float__(self):        
            return self.__val__.__float__()

        def __str__(self):          
            return self.space.__str__() + "::Vector(" + self.__val__.__str__() + ")"

        def __repr__(self):         
            return self.space.__repr__() + "::Vector(" + self.__val__.__repr__() + ")"


class RN(LinearSpace):
    """The real space R^n
    """

    def __init__(self, n):
        if not isinstance(n, Integral) or n<1:
            raise TypeError("n ({}) has to be a positive integer".format(np))
        self.n = n
        self._field = RealNumbers()
    
    def linCombImpl(self, a, x, b, y):
        y.values[:] = a*x.values + b*y.values

    def zero(self):
        return self.makeVector(np.zeros(self.n), dtype=float, copy=False)

    def empty(self):
        return self.makeVector(np.empty(self.n), dtype=float, copy=False)

    @property
    def field(self):
        return self._field

    @property
    def dimension(self):
        return self.n

    def equals(self, other):
        return isinstance(other, RN) and self.n == other.n

    def makeVector(self, *args, **kwargs):
        if isinstance(args[0], np.ndarray):
            if args[0].shape == (self.n,):
                return RN.Vector(self, args[0])
            else:
                raise ValueError("Input numpy array ({}) is of shape {}, expected shape shape {}".format(args[0],args[0].shape, (self.n,)))
        else:
            return self.makeVector(np.array(*args, **kwargs))

    class Vector(HilbertSpace.Vector, Algebra.Vector):        
        def __init__(self, space, values):
            HilbertSpace.Vector.__init__(self, space)
            self.values = values
        
        def __abs__(self):                  
            return self.space.makeVector(abs(self.values))

        def __str__(self):                  
            return self.space.__str__() + "::Vector(" + self.values.__str__() + ")"

        def __repr__(self):                 
            return self.space.__repr__() + "::Vector(" + self.values.__repr__() + ")"

        def __getitem__(self, index):        
            return self.values.__getitem__(index)

        def __setitem__(self, index, value):  
            return self.values.__setitem__(index, value)

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.n) + ")"

    def __repr__(self):                 
        return "RN(" + str(self.n) + ")"

class EuclidianSpace(RN, HilbertSpace, Algebra):
    """The real space R^n with the euclidean norm
    """

    def innerImpl(self, x, y):
        return float(np.vdot(x.values, y.values))

    def multiplyImpl(self, x, y):
        y.values[:] = x.values*y.values


def makePooledSpace(base, *args, **kwargs):
    """ Pooled space provides a optimization in reusing vectors and returning them from empty.
    """
    BaseType = type(base)
    BaseVectorType = BaseType.Vector

    class PooledSpace(BaseType):
        def __init__(self, base, *args, **kwargs):
            self._pool = []
            self._poolMaxSize = kwargs.pop('maxPoolSize', 1)
            self._base = base

        def empty(self):
            if self._pool:
                return self._pool.pop()
            else:
                return BaseType.empty(self)

        def __getattr__(self, name):
            return getattr(self._base, name)

        def __str__(self):
            return "PooledSpace(" + str(self._base) + ", Pool size:" + str(len(self._pool)) + ")"

        class Vector(BaseVectorType):
            def __del__(self):
                if len(self.space._pool) < self.space._poolMaxSize:
                    self.space._pool.append(self)
                else:
                    pass#TODO BaseVectorType.__del__(self)

    return PooledSpace(base, *args, **kwargs)