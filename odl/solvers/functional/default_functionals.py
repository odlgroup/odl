# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Default functionals defined on any (reasonable) space."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.solvers.functional.functional import Functional
from odl.operator.operator import Operator
#from odl.space.pspace import ProductSpace
#from odl.set.space import LinearSpace, LinearSpaceVector
#from odl.set.sets import Field
import numpy as np


__all__ = ('L1Norm','L2Norm','L2NormSquare')

class L1Norm(Functional):
    def __init__(self, domain):
        super().__init__(domain=domain, linear=False, convex=True, concave=False, smooth=False, grad_lipschitz=np.inf)
    
    def _call(self, x):
        return np.abs(x).inner(self.domain.one())
    
    @property
    def gradient(x):
        raise NotImplementedError
    
    def proximal(self, sigma=1.0):
        functional = self
        
        class L1Proximal(Operator):
            def __init__(self):
                super().__init__(functional.domain, functional.domain,
                                 linear=False)
                self.sigma = sigma
            
            #TODO: Check that this works for complex x
            def _call(self, x):
                return np.maximum(np.abs(x)-sigma,0)*np.sign(x) 
        
        return L1Proximal()
    @property
    def conjugate_functional(self):
        functional = self
        
        class L1Conjugate_functional(Functional):
            def __init__(self):
                super().__init__(functional.domain, linear=False)
                
            def _call(self, x):
                if np.max(np.abs(x)) > 1:
                    return np.inf
                else:
                    return 0
                    
        return L1Conjugate_functional()            

class L2Norm(Functional):
    def __init__(self, domain):
        super().__init__(domain=domain, linear=False, convex=True, concave=False, smooth=False, grad_lipschitz=np.inf)
    
    def _call(self, x):
        return np.sqrt(np.abs(x).inner(np.abs(x)))
    
    @property
    def gradient(self):
        functional = self
        
        class L2Gradient(Operator):
            def __init__(self):
                super().__init__(functional.domain, functional.domain,
                                 linear=False)
            
            def _call(self, x):
                return x/x.norm()
        
        return L2Gradient()
    
    def proximal(self, sigma=1.0):
        functional = self
        
        class L2Proximal(Operator):
            def __init__(self):
                super().__init__(functional.domain, functional.domain,
                                 linear=False)
                self.sigma = sigma
            
            #TODO: Check that this works for complex x
            def _call(self, x):
                return np.maximum(x.norm()-sigma,0)*(x/x.norm())
        
        return L2Proximal()

    @property
    def conjugate_functional(self):
        functional = self
        
        class L2Conjugate_functional(Functional):
            def __init__(self):
                super().__init__(functional.domain, linear=False)
                
            def _call(self, x):
                if x.norm() > 1:
                    return np.inf
                else:
                    return 0
                    
        return L2Conjugate_functional()            

class L2NormSquare(Functional):
    def __init__(self, domain):
        super().__init__(domain=domain, linear=False, convex=True, concave=False, smooth=True, grad_lipschitz=2)
    
    def _call(self, x):
        return np.abs(x).inner(np.abs(x))
    
    @property
    def gradient(self):
        functional = self
        
        class L2SquareGradient(Operator):
            def __init__(self):
                super().__init__(functional.domain, functional.domain,
                                 linear=False)
            
            def _call(self, x):
                return 2*x
        
        return L2SquareGradient()
    
    def proximal(self, sigma=1.0):
        functional = self
        
        class L2SquareProximal(Operator):
            def __init__(self):
                super().__init__(functional.domain, functional.domain,
                                 linear=False)
                self.sigma = sigma
            
            #TODO: Check that this works for complex x
            def _call(self, x):
                return x/3
        
        return L2SquareProximal()

    @property
    def conjugate_functional(self):
        functional = self
        
        class L2SquareConjugateFunctional(Functional):
            def __init__(self):
                super().__init__(functional.domain, linear=False)
                
            def _call(self, x):
                return x.norm()/4                
                
        return L2SquareConjugateFunctional()            




