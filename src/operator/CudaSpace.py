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
from abc import ABCMeta, abstractmethod, abstractproperty
from math import sqrt
from RL.operator.space import *
from SimRec2DPy.PyCuda import *

class CudaRN(HilbertSpace):
    """The real space R^n
    """

    def __init__(self,n):
        self.n = n
        self.impl = CudaRNImpl(n)

    def inner(self,x,y):
        return self.impl.inner(x.impl,y.impl)
    
    def linComb(self,a,x,b,y):
        self.impl.linComb(a,x.impl,b,y.impl)

    def zero(self):
        return self.makeVector(self.impl.zero())

    def empty(self):
        return self.makeVector(self.impl.empty())

    @property
    def field(self):
        return Field.Real

    @property
    def dimension(self):
        return self.n

    def __eq__(self, other):
        return isinstance(other,CudaRN) and self.n == other.n

    def makeVector(self, *args, **kwargs):
        return CudaRN.Vector(self,*args, **kwargs)

    class Vector(HilbertSpace.Vector):        
        def __init__(self,space,*args, **kwargs):
            HilbertSpace.Vector.__init__(self,space)
            if isinstance(args[0],CudaRNVectorImpl):
                self.impl = args[0]
            else:
                self.impl = CudaRNVectorImpl(*args)

        def assign(self,other):
            self.impl.assign(other.impl)

        def asRNVector(self,rn):
            return rn.makeVector(self.impl.copyToHost(), copy = False)
            
        def __str__(self):                      return "" + self.impl.__str__()
        def __repr__(self):                     return "CudaRNVector("+self.impl.__str__()+")"

        #Slow get and set, for testing and nothing else!
        def __getitem__(self,index):
            if isinstance(index,slice):
                if 0>index.start or index.stop>=self.space.n:
                    raise IndexError("Out of range")
                
                return self.impl.getSlice(index)
            else:
                if 0>index or index>=self.space.n:
                    raise IndexError("Out of range")

                return self.impl.__getitem__(index)

        def __setitem__(self,index,value):    
            if isinstance(index,slice):
                if 0>index.start or index.stop>=self.space.n:
                    raise IndexError("Out of range")
                
                return self.impl.setSlice(index,value)
            else:
                if 0>index or index>=self.space.n:
                    raise IndexError("Out of range")

            return self.impl.__setitem__(index,value)

