# Copyright 2014, 2015 The ODL development group
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

import warnings
import numpy as np
from odl.operator.operator import LinearOperator
from odl.space.ntuples import NtuplesBase
from odl.discr.l2_discr import DiscreteL2

def vector_examples(space):
    #All spaces should yield the zero element
    yield ('Zero', space.zero())
    
    if isinstance(space, DiscreteL2):
        #Get the points and calculate some statistics on them
        points = space.points()        
        mins = np.apply_along_axis(np.min, axis=1, arr=points)
        maxs = np.apply_along_axis(np.max, axis=1, arr=points)
        means = np.apply_along_axis(np.mean, axis=1, arr=points)
        stds = np.apply_along_axis(np.std, axis=1, arr=points)
        
        #indicator function in first dimension
        mean = points[0,...]
        ind_fun = space.uspace.element(lambda *args: args[0]>=mean/2)    
        yield ('Step', space.element(ind_fun))      
        
        #indicator function on hypercube
        def _cube_fun(*args):
            inside = np.ones(np.broadcast(*args).shape, dtype=bool)
            for points, minv, maxv in zip(args, mins, maxs):
                inside = np.logical_and(inside, minv < points < maxv)
            return inside.astype(space.dtype)
        
        yield ('Cube', space.element(_cube_fun))
        
        #indicator function on hypersphere
        def _sphere_fun(*args):
            r = np.zeros(np.broadcast(*args).shape)
            
            for points, mean, std in zip(args, means, stds):
                r += (points - mean) / std
                
            return (r<1.0).astype(space.dtype)
        
        yield ('Sphere', space.element(_sphere_fun))
        
        #Gaussian function
        def _gaussian_fun(*args):
            r2 = np.zeros(np.broadcast(*args).shape)
            
            for points, mean, std in zip(args, means, stds):
                r2 += (points - mean)**2 / (2*std**2)
                
            return np.exp(-r2)
        
        yield ('Gaussian', space.element(_gaussian_fun))
        
        #Gradient in each dimensions
        for dim in range(space.dim):
            def _gradient_fun(*args):
                s = np.zeros(np.broadcast(*args).shape)
                s += args[dim]-mins[dim]
                    
                return s
            
            yield ('gradient {}'.format(dim), space.element(_gradient_fun))
        
        #Gradient in all dimensions
        def _all_gradient_fun(*args):
            s = np.zeros(np.broadcast(*args).shape)
            
            for points, minv, std in zip(args, mins, stds):
                s += (points - minv) / std
                
            return s
        
        yield ('Gradient all', space.element(_all_gradient_fun))
        
    elif isinstance(space, NtuplesBase):
        yield ('Linspaced', space.element(np.linspace(0, 1, space.dim)))

        yield ('Ones', space.element(np.ones(space.dim)))
        
        yield ('Random noise', space.element(np.random.rand(space.dim)))
           
        yield ('Normally distributed random noise', space.element(np.random.randn(space.dim)))
        
    else:
        warnings.warn('No known examples in this space')
        

class OpeartorTest(object):
    def __init__(self, operator):
        self.operator = operator
        
    def norm(self):
        print('Calculating operator norm')        
        
        operator_norm = 0.0
        for [name, vec] in vector_examples(self.operator.domain):
            estimate = self.operator(vec).norm() / vec.norm()
            print('Norm estimate for {}: {}', name, estimate)
            operator_norm = max(operator_norm, estimate)
            
        print('Norm is at least: {}', operator_norm)
        return operator_norm        
        
    def adjoint(self):
        """ Verifies that the adjoint works appropriately
        """
        try:
            self.operator.adjoint
        except NotImplementedError:
            print('Operator has no adjoint')
            return
            
        print('Verifying adjoint of operator')
        
        for [name_dom, vec_dom] in vector_examples(self.operator.domain):
            for [name_ran, vec_ran] in vector_examples(self.operator.range):
                Axy = self.operator(vec_dom).inner(vec_ran)
                xAty = vec_dom.inner(self.operator.adjoint(vec_ran))
                print('With vectors dom={} and ran={} got: (Ax, y)={}, (x, A^T y)={}', name_dom, name_ran, Axy, xAty)
                
    
    def derivative(self):
        """ Verifies that the derivative works appropriately
        """
        try:
            self.operator.derivative(self.operator.domain.zero())
        except NotImplementedError:
            print('Operator has no derivative')
            return
            
        print('Verifying derivative of operator')
        
        for [name_x, x] in vector_examples(self.operator.domain):
            deriv = self.operator.derivative(x)
            opx = self.operator(x)
            for [name_dx, dx] in vector_examples(self.operator.domain):
                for c in np.logspace(-2, -8, 7):
                    exact_step = self.operator(x+c*dx)-opx
                    expected_step = deriv(c*dx)
                    
                    error = (exact_step-expected_step).norm()
                
                    print("With vectors x={}, dx={} and step={} got ||A(x+d*dx)-A(x)-A'(x)(dx)||={}", name_x, name_dx, c, error)
                    
    def linear(self):
        """ Verifies that the operator is actually linear
        """
        if not isinstance(self, LinearOperator):
            print('Operator is not linear')
            return
        else:
            print('Verifying linearity of operator')

        #Test scaling
        for [name_x, x] in vector_examples(self.operator.domain):
            opx = self.operator(x)
            for scale in [-5.0, -0.5, 0.0, 0.5, 5.0, 100.0]:
                scaled_opx = self.operator(scale*x)
                error = (scaled_opx - scale * opx).norm()
                print("with vector x={}, scale={} got ||A(c*x)-c*A(x)||={}".format(name_x, scale, error))
        
        #Test addition
        for [name_x, x] in vector_examples(self.operator.domain):
            opx = self.operator(x)
            for [name_y, y] in vector_examples(self.operator.domain):
                opy = self.operator(y)
                opxy = self.operator(x+y)
                error = (opxy - opx - opy).norm()
                print("with vector x={}, y={} got ||A(x+y)-A(x)-A(y)||={}".format(name_x, name_y, error))
                    
    def run_tests(self):
        """Runs all tests on this operator
        """
        self.norm()
        self.adjoint()
        self.derivative()
        self.linear()