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
from scipy.lib.blas import get_blas_funcs

from RL.utility.utility import allEqual
from RL.space.space import *
from RL.space.set import *

class RN(LinearSpace):
    """The real space R^n
    """

    def __init__(self, n):
        if not isinstance(n, Integral) or n<1:
            raise TypeError("n ({}) has to be a positive integer".format(np))
        self.n = n
        self._field = RealNumbers()
        self._axpy, self._scal, self._copy = get_blas_funcs(['axpy','scal','copy'])
    
    def linCombImpl(self, z, a, x, b, y):
        #Implement y = a*x + b*y using optimized BLAS rutines

        if x is y and b != 0:
            self.linCombImpl(z, a+b, x, 0, x)
        elif z is x and z is y:
            self._scal(a+b, z.values)
        elif z is x:
            if a != 1:
                self._scal(a, z.values)
            if b != 0:
                self._axpy(y.values, z.values, self.dimension, b)
        elif z is y:
            if b != 1:
                self._scal(b, z.values)
            if a != 0:
                self._axpy(x.values, z.values, self.dimension, a)
        else:
            if b == 0:
                if a == 0:
                    z.values[:] = 0
                else:
                    self._copy(x.values, z.values)
                    if a != 1:
                        self._scal(a, z.values)
            else:
                if a == 0:
                    self._copy(y.values, z.values)
                    if b!= 1:
                        self._scal(b, z.values)

                elif a == 1:
                    self._copy(x.values, z.values)
                    self._axpy(y.values, z.values, self.dimension, b)
                else:
                    self._copy(y.values, z.values)
                    if b != 1:
                        self._scal(b, z.values)
                    self._axpy(x.values, z.values, self.dimension, a)


    def zero(self):
        return self.makeVector(np.zeros(self.n, dtype=float))

    def empty(self):
        return self.makeVector(np.empty(self.n, dtype=float))

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
            return self.makeVector(np.array(*args, **kwargs).astype(float, copy = False))

    class Vector(HilbertSpace.Vector, Algebra.Vector):        
        def __init__(self, space, values):
            HilbertSpace.Vector.__init__(self, space)
            self.values = values
        
        def __abs__(self):                  
            return self.space.makeVector(abs(self.values))

        def __str__(self):                  
            return str(self.space) + "::Vector(" + str(self.values) + ")"

        def __repr__(self):                 
            return repr(self.space) + "::Vector(" + repr(self.values) + ")"

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