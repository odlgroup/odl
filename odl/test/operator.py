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
from itertools import product

from odl.set.pspace import ProductSpace
from odl.operator.operator import LinearOperator
from odl.space.ntuples import FnBase, NtuplesBase
from odl.discr.l2_discr import DiscreteL2

from math import floor, log10

__all__ = ('vector_examples', 'OpeartorTest')

def _arg_shape(*args):
    if len(args) == 1:
        return args[0].shape
    else:
        return np.broadcast(*args).shape
        
def _round_sig(x, sig=3):
    if x == 0:
        return x
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)
        
def vector_examples(space):
    #All spaces should yield the zero element
    yield ('Zero', space.zero())
    
    if isinstance(space, ProductSpace):
        for examples in product(*[vector_examples(spc) for spc in space]):
            name = ', '.join(name for name, _ in examples)
            vector = space.element([vec for _, vec in examples])
            yield (name, space.element(vector))
    
    elif isinstance(space, DiscreteL2):
        uspace = space.uspace
        
        #Get the points and calculate some statistics on them
        points = space.points()        
        mins = space.grid.min()
        maxs = space.grid.max()
        means = (maxs+mins)/2.0
        stds = np.apply_along_axis(np.std, axis=0, arr=points)
        
        #indicator function in first dimension
        def _step_fun(*args):
            z = np.zeros(_arg_shape(*args))
            z[:space.grid.shape[0]//2,...] = 1
            return z
            
        yield ('Step', space.element(uspace.element(_step_fun)))
        
        #indicator function on hypercube
        def _cube_fun(*args):
            inside = np.ones(_arg_shape(*args), dtype=bool)
            for points, mean, std in zip(args, means, stds):
                inside = np.logical_and(inside, points < mean+std)
                inside = np.logical_and(inside, mean-std < points)
                
            return inside.astype(space.dtype)
        
        yield ('Cube', space.element(uspace.element(_cube_fun)))
        
        #indicator function on hypersphere
        def _sphere_fun(*args):
            r = np.zeros(_arg_shape(*args))
            
            for points, mean, std in zip(args, means, stds):
                r += (points - mean)**2 / std**2
            return (r<1.0).astype(space.dtype)
        
        yield ('Sphere', space.element(uspace.element(_sphere_fun)))
        
        #Gaussian function
        def _gaussian_fun(*args):
            r2 = np.zeros(_arg_shape(*args))
            
            for points, mean, std in zip(args, means, stds):
                r2 += (points - mean)**2 / ((std/2)**2)
                
            return np.exp(-r2)
        
        yield ('Gaussian', space.element(uspace.element(_gaussian_fun)))
        
        #Gradient in each dimensions
        for dim in range(space.grid.ndim):
            def _gradient_fun(*args):
                s = np.zeros(_arg_shape(*args))
                s += (args[dim]-mins[dim]) / (maxs[dim]-mins[dim])
                    
                return s
            
            yield ('grad {}'.format(dim), space.element(uspace.element(_gradient_fun)))
        
        #Gradient in all dimensions
        def _all_gradient_fun(*args):
            s = np.zeros(_arg_shape(*args))
            
            for points, minv, maxv in zip(args, mins, maxs):
                s += (points - minv) / (maxv-minv)
                
            return s
        
        yield ('Grad all', space.element(uspace.element(_all_gradient_fun)))
        
    elif isinstance(space, FnBase):
        yield ('Linspaced', space.element(np.linspace(0, 1, space.dim)))

        yield ('Ones', space.element(np.ones(space.dim)))
        
        yield ('Random noise', space.element(np.random.rand(space.dim)))
           
        yield ('Normally distributed random noise', space.element(np.random.randn(space.dim)))
                
    else:
        warnings.warn('No known examples in this space')
        

class OpeartorTest(object):
    def __init__(self, operator, operator_norm=None):
        self.operator = operator
        self.operator_norm = operator_norm
        
    def norm(self):
        print('\n== Calculating operator norm ==\n')        
        
        operator_norm = 0.0
        for [name, vec] in vector_examples(self.operator.domain):
            result = self.operator(vec)
            vecnorm = vec.norm()
            estimate = 0 if vecnorm == 0 else result.norm() / vecnorm
                
            operator_norm = max(operator_norm, estimate)
            
        print('Norm is at least: {}'.format(operator_norm))
        self.operator_norm = operator_norm
        return operator_norm
        
    def adjoint(self):
        """ Verifies that the adjoint works appropriately
        """
        try:
            self.operator.adjoint
        except NotImplementedError:
            print('Operator has no adjoint')
            return
            
        print('\n== Verifying adjoint of operator ==\n')
        print('Verifying the identity (Ax, y) = (x, A^T y)')
        print('error = ||(Ax, y) - (x, A^T y)|| / ||A|| ||x|| ||y||')
        
        x = []
        y = []
        
        num_failed = 0
        
        for [name_dom, vec_dom] in vector_examples(self.operator.domain):
            vec_dom_norm = vec_dom.norm()
            for [name_ran, vec_ran] in vector_examples(self.operator.range):
                vec_ran_norm = vec_ran.norm()
            
                Axy = self.operator(vec_dom).inner(vec_ran)
                xAty = vec_dom.inner(self.operator.adjoint(vec_ran))
                
                denom = self.operator_norm * vec_dom_norm * vec_ran_norm
                error = 0 if denom == 0 else abs(Axy-xAty)/denom
                    
                if error > 0.00001:
                    print('x={:25s} y={:25s} : error={:6.5f}'.format(name_dom, name_ran, error))
                    num_failed += 1
                
                x.append(Axy)
                y.append(xAty)
                
        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:         
            print('*** FAILED {} TEST CASES ***'.format(num_failed))
                
        scale = np.polyfit(x, y, 1)[0]  
        print('\nThe adjoint seems to be scaled according to:')
        print('(x, A^T y) / (Ax, y) = {}. Should be 1.0'.format(scale))
        
    def derivative(self, step=0.0001):
        """ Verifies that the derivative works appropriately
        """
        try:
            self.operator.derivative(self.operator.domain.zero())
        except NotImplementedError:
            print('Operator has no derivative')
            return
            
        if self.operator_norm is None:
            print('Cannot do tests before norm is calculated, run test.norm() or give norm as a parameter')
            return
            
        print('\n== Verifying derivative of operator with step = {} ==\n'.format(step))
        print("error = ||A(x+c*dx)-A(x)-c*A'(x)(dx)|| / |c| ||dx|| ||A||")
        
        num_failed = 0        
        
        for [name_x, x] in vector_examples(self.operator.domain):
            deriv = self.operator.derivative(x)
            opx = self.operator(x)
            for [name_dx, dx] in vector_examples(self.operator.domain):
                exact_step = self.operator(x+dx*step)-opx
                expected_step = deriv(dx*step)
                denom = step * dx.norm() * self.operator_norm
                error = 0 if denom == 0 else (exact_step-expected_step).norm()/denom
                
                if error > 0.00001:
                    print("x={:15s} dx={:15s} : error={:6.5f}".format(name_x, name_dx, step, error))
                    num_failed += 1
                          
        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:         
            print('*** FAILED {} TEST CASES ***'.format(num_failed))    
                    
    def linear(self):
        """ Verifies that the operator is actually linear
        """
        if not isinstance(self.operator, LinearOperator):
            print('Operator is not linear')
            return
            
        if self.operator_norm is None:
            print('Cannot do tests before norm is calculated, run test.norm() or give norm as a parameter')
            return
            
        print('\n== Verifying linearity of operator ==\n')
        
        #Test zero gives zero
        result = self.operator(self.operator.domain.zero())
        print("||A(0)||={:6.5f}. Should be 0.0000".format(result.norm()))

        print("\nCalculating invariance under scaling")
        print("error = ||A(c*x)-c*A(x)|| / |c| ||A|| ||x||")

        #Test scaling
        num_failed = 0

        for [name_x, x] in vector_examples(self.operator.domain):
            opx = self.operator(x)
            for scale in [-5.0, -0.5, 0.0, 0.5, 5.0, 100.0]:
                scaled_opx = self.operator(scale*x)
                
                denom = self.operator_norm * scale * x.norm()
                error = 0 if denom == 0 else (scaled_opx - opx * scale).norm()/denom
                
                if error > 0.00001:
                    print("x={:25s} scale={:7.2f} error={:6.5f}".format(name_x, scale, error))
                    num_failed += 1
                    
        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:         
            print('*** FAILED {} TEST CASES ***'.format(num_failed))  
        
        print("\nCalculating invariance under addition")
        print("error = ||A(x+y)-A(x)-A(y)|| / ||A||(||x|| + ||y||)")        
        
        #Test addition
        num_failed = 0
        
        for [name_x, x] in vector_examples(self.operator.domain):
            opx = self.operator(x)
            for [name_y, y] in vector_examples(self.operator.domain):
                opy = self.operator(y)
                opxy = self.operator(x+y)
                
                denom = self.operator_norm * (x.norm() + y.norm())
                error = 0 if denom == 0 else (opxy - opx - opy).norm()/denom                
                
                if error > 0.00001:
                    print("x={:25s} y={:25s} error={:6.5f}".format(name_x, name_y, error))
                    num_failed += 1
                
        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:         
            print('*** FAILED {} TEST CASES ***'.format(num_failed))
                    
    def run_tests(self):
        """Runs all tests on this operator
        """
        print('\n== RUNNING ALL TESTS ==\n')
        print('Operator = {}'.format(self.operator))
        
        self.norm()
        if isinstance(self.operator, LinearOperator):
            self.linear()
            self.adjoint()
        else:
            self.derivative()
        
    def __repr__(self):
        return 'OperatorTest({!r})'.format(self.operator)
        
if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)