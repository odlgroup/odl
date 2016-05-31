# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:36:12 2016

@author: johan79
"""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import odl

# Discretization parameters
n = 4

# Discretized spaces
space = odl.uniform_discr([0, 0], [1, 1], [n, n])


print(space)

class L1Functional(odl.solvers.functional.Functional):
    def __init__(self, domain):
        super().__init__(domain=domain, linear=False, convex=True, concave=False, smooth=False, lipschitz=1)
    
    def _call(self, x):
        return np.abs(x).inner(self.domain.one())
    
    def gradient(x):
        raise NotImplementedError
    
    def proximal(self, sigma=1.0):
        functional = self
        
        class L1Proximal(odl.operator.Operator):
            def __init__(self):
                super().__init__(functional.domain, functional.domain,
                                 linear=False)
                self.sigma = sigma
            
            #TODO: Check that this works for complex x
            def _call(self, x):
                return np.maximum(np.abs(x)-sigma,0)*np.sign(x) 
        
        return L1Proximal()

    def conjugate_functional(self):
        functional = self
        
        class L1Conjugate_functional(odl.solvers.functional.Functional):
            def __init__(self):
                super().__init__(functional.domain, linear=False)
                
            def _call(self, x):
                if np.max(np.abs(x)) > 1:
                    return np.inf
                else:
                    return 0
                    
        return L1Conjugate_functional()            
                    
        
l1func = L1Functional(space)
l1prox = l1func.proximal(sigma=1.5)
l1conjFun = l1func.conjugate_functional()

# Create phantom
phantom = odl.util.shepp_logan(space, modified=True)*5+1

onevector=space.one()*5

prox_phantom=l1prox(phantom)
l1conjFun_phantom = l1conjFun(phantom)


'''
# Initialize convolution operator by Fourier formula
#     conv(f, g) = F^{-1}[F[f] * F[g]]
# Where F[.] is the Fourier transform and the fourier transform of a guassian
# with standard deviation filter_width is another gaussian with width
# 1 / filter_width
filter_width = 2.0  # standard deviation of the Gaussian filter
ft = odl.trafos.FourierTransform(space)
c = filter_width**2 / 4.0**2
gaussian = ft.range.element(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) * c))
convolution = ft.inverse * gaussian * ft

# Optional: Run diagnostics to assure the adjoint is properly implemented
# odl.diagnostics.OperatorTest(conv_op).run_tests()

# Create phantom
phantom = odl.util.shepp_logan(space, modified=True)

# Create vector of convolved phantom
data = convolution(phantom)
data.show('Convolved data')

# Set up the Chambolle-Pock solver:

# Initialize gradient operator
gradient = odl.Gradient(space, method='forward')

# Column vector of two operators
op = odl.BroadcastOperator(convolution, gradient)

# Create the proximal operator for unconstrained primal variable
proximal_primal = odl.solvers.proximal_zero(op.domain)

# Create proximal operators for the dual variable

# l2-data matching
prox_convconj_l2 = odl.solvers.proximal_cconj_l2_squared(space, g=data)

# Isotropic TV-regularization i.e. the l1-norm
prox_convconj_l1 = odl.solvers.proximal_cconj_l1(gradient.range, lam=0.0003,
                                                 isotropic=True)

# Combine proximal operators, order must correspond to the operator K
proximal_dual = odl.solvers.combine_proximals(prox_convconj_l2,
                                              prox_convconj_l1)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op, 5)

niter = 500  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable


# Optionally pass partial to the solver to display intermediate results
partial = (odl.solvers.PrintIterationPartial() &
           odl.solvers.ShowPartial(display_step=20))

# Choose a starting point
x = op.domain.zero()

# Run the algorithm
odl.solvers.chambolle_pock_solver(
    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual, niter=niter, partial=partial)

# Display images
phantom.show(title='original image')
data.show(title='convolved image')
x.show(title='deconvolved image', show=True)
'''

