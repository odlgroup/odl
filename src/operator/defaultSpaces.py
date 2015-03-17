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
import numpy as np

import RL.operator.operatorAlternative as OP
import RL.operator.functional as FUN
from RL.utility.utility import allEqual
from RL.operator.measure import *
from RL.operator.space import *
    
class ProductSpace(HilbertSpace):
    """Product space (X1 x X2 x ... x Xn)
    """

    def __init__(self,*spaces):
        if (len(spaces)==0):
            raise TypeError("Empty product not allowed")
        if (not allEqual(spaces, lambda x,y: x.field == y.field)):
            raise TypeError("All spaces must have the same field")

        self.spaces = spaces
        self._dimension = sum(space.dimension for space in self.spaces)
        self._field = spaces[0].field  #X_n has same field

    def zero(self):
        return self.makeVector(*[A.zero() for A in self.spaces])

    def zero(self):
        return self.makeVector(*[A.empty() for A in self.spaces])
    
    def inner(self,v1,v2):
        return sum(space.inner(v1p,v2p) for [space,v1p,v2p] in zip(self.spaces,v1.parts,v2.parts))

    def linComb(self,a,x,b,y):
        for [space,xp,yp] in zip(self.spaces,x.parts,y.parts):
            space.linComb(a,xp,b,yp)

    @property
    def field(self):
        return self._field

    @property
    def dimension(self):
        return self._dimension

    def __eq__(self, other):
        return all(x == y for [x,y] in zip(self.spaces,other.spaces))
    
    def makeVector(self,*args):
        return ProductSpace.Vector(self,*args)

    def __getitem__(self,index):
        return self.spaces[index]

    class Vector(HilbertSpace.Vector):
        def __init__(self,space,*args):
            HilbertSpace.Vector.__init__(self,space)

            if (not isinstance(args[0],HilbertSpace.Vector)): #Delegate constructors
                self.parts = [space.makeVector(arg) for [arg,space] in zip(args,space.spaces)]
            else: #Construct from existing tuple
                if any(part.space != space for [part,space] in zip(args,space.spaces)):
                    raise TypeError("The spaces of all parts must correspond to this space's parts")

                self.parts = args

        def assign(self, other):
            for [p1,p2] in zip(self.parts,other.parts):
                p1.assign(p2)

        def __getitem__(self,index):
            return self.parts[index]

        def __str__(self):          return "[" + ",".join(str(part) for part in self.parts) + "]"
        def __repr__(self):         return "%s(%s)" % (self.__class__.__name__, str(self))


class Reals(HilbertSpace):
    """The real numbers
    """

    def inner(self,x,y):
        return x.__val__ * y.__val__

    def linComb(self,a,x,b,y):
        y.__val__ = a * x.__val__ + b * y.__val__

    def zero(self):
        return self.makeVector(0.0)

    @property
    def field(self):
        return Field.Real

    @property
    def dimension(self):
        return 1
    
    def __eq__(self, other):
        return isinstance(other,Reals)

    def makeVector(self,value):
        return Reals.Vector(self,value)

    class Vector(HilbertSpace.Vector):
        """Real vectors are floats
        """

        __val__ = None
        def __init__(self, space, v):
            HilbertSpace.Vector.__init__(self,space)
            self.__val__ = v

        def assign(self,other):     self.__val__ = other.__val__
        def __float__(self):        return self.__val__.__float__()
        def __str__(self):          return "" + self.__val__.__str__()
        def __repr__(self):         return "Real(%d)" % (self.__val__)

    class MultiplyOp(OP.SelfAdjointOperator):    
        """Multiply with scalar
        """

        def __init__(self,space,a):
            self.space = space
            self.a = a

        def apply(self,rhs,out):
            out.assign(self.a*rhs)

        def domain(self):           return self.space
        def range(self):            return self.space


    class AddOp(OP.SelfAdjointOperator):
        """Add scalar
        """

        def __init__(self,space,a):
            self.space = space
            self.a = a

        def apply(self,rhs,out):
            out.assign(self.a + rhs)

        def domain(self):           return self.space
        def range(self):            return self.space


class RN(HilbertSpace):
    """The real space R^n
    """

    def __init__(self,n):
        self.n = n

    def inner(self,x,y):
        return np.vdot(x.values,y.values)
    
    def linComb(self,a,x,b,y):
        y.values[:]=a*x.values+b*y.values

    def zero(self):
        return self.makeVector(np.zeros(self.n),copy = False)

    def empty(self):
        return self.makeVector(np.empty(self.n),copy = False)

    @property
    def field(self):
        return Field.Real

    @property
    def dimension(self):
        return self.n

    def __eq__(self, other):
        return isinstance(other,RN) and self.n == other.n

    def makeVector(self, *args, **kwargs):
        return RN.Vector(self,*args, **kwargs)

    class Vector(HilbertSpace.Vector):        
        def __init__(self,space,*args, **kwargs):
            HilbertSpace.Vector.__init__(self,space)
            self.values = np.array(*args,**kwargs)

        def assign(self,other):
            self.values[:] = other.values
            
        def __str__(self):              return "" + self.values.__str__()
        def __repr__(self):             return "RNVector("+self.values.__str__()+")"
        def __getitem__(self,index):    return self.values[index]

    def MakeMultiplyOp(self,A):
        class MultiplyOp(OP.LinearOperator):
            """Multiply with matrix
            """

            def __init__(self,space,A):
                self.space = space
                self.A = A

            def apply(self,rhs,out):
                out.values[:] = np.dot(self.A,rhs.values)

            def applyAdjoint(self,rhs,out):
                out.values[:] = np.dot(self.A.T,rhs.values)

            @property
            def mat(self):              return self.A
            def domain(self):           return self.space
            def range(self):            return self.space

        return MultiplyOp(self,A)

#Example of a space:
class L2(HilbertSpace):
    """The space of square integrable functions on some domain
    """

    def __init__(self,domain):
        self.domain = domain

    def inner(self,v1,v2):
        raise NotImplementedError("You cannot calculate inner products in non-discrete spaces")

    def innerImpl(self,integrator,v1,v2):
        #v1 and v2 elements in discretized space
        return integrator.integrate(v1*v2)

    def linComb(self,a,x,b,y):
        return a*x+b*y #Use operator overloading

    def field(self):
        return Field.Real

    def dimension(self):
        raise NotImplementedError("TODO: infinite")
    
    def __eq__(self, other):
        return isinstance(other,L2) and self.domain == other.domain

    def zero(self):
        return self.makeVector(lambda t: 0)

    def makeVector(self, *args, **kwargs):
        return L2.Vector(self,*args, **kwargs)

    class Vector(FUN.Functional,HilbertSpace.Vector):
        """ L2 Vectors are operators from the domain onto R(C)
        """

        def __init__(self, space, function):
            HilbertSpace.Vector.__init__(self,space)
            self.function = function

        def apply(self,rhs):
            return self.function(rhs)

        def assign(self,other):     self.function = other.function
        def domain(self):           return self.space.domain
        def range(self):            return self.space.field


class UniformDiscretization(RN, Discretization):
    """ Uniform discretization of an interval
    Represents vectors by RN elements
    Uses trapezoid method for integration
    """

    def __init__(self,parent,n):
        if not isinstance(parent.domain,Interval):
            raise NotImplementedError("Can only discretize intervals")

        self.parent = parent
        RN.__init__(self,n)

    def inner(self,v1,v2): #Delegate to main space
        return self.parent.innerImpl(self,v1,v2)

    def zero(self):
        return self.makeVector(np.zeros(self.n),copy = False)

    def empty(self):
        return self.makeVector(np.empty(self.n),copy = False)
    
    def __eq__(self, other):
        return isinstance(other,UniformDiscretization) and RN.__eq__(self,other)

    def makeVector(self, *args, **kwargs):
        return UniformDiscretization.Vector(self,*args, **kwargs)

    def integrate(self,f):
        return np.trapz(f.values,dx=(self.parent.domain.end-self.parent.domain.begin)/(self.n-1))

    def points(self):
        return np.linspace(self.parent.domain.begin,self.parent.domain.end,self.n)

    class Vector(RN.Vector):
        def __init__(self,space, *args,**kwargs):
            if (len(args)==1 and isinstance(args[0],L2.Vector) and args[0].space == space.parent):
                data = RN.Vector.__init__(self,space,args[0](space.points()),copy = False)
            else:
                data = RN.Vector.__init__(self,space,*args,**kwargs)

        def __mul__(self,other):
            if isinstance(other,UniformDiscretization.Vector):
                return self.space.makeVector(self.values*other.values)
            else:
                HilbertSpace.Vector.__mul__(self,other)

        def assign(self,other):
            RN.Vector.assign(self,other)