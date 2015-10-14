from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy
import scipy.ndimage
import odl

class Convolution(odl.LinearOperator):
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel
        self.scale = kernel.space.domain.volume / len(kernel)
        
        super().__init__(space, space)

    def _apply(self, rhs, out):
        scipy.ndimage.convolve(rhs, 
                               self.kernel,
                               output=out.asarray(),
                               mode='constant')
                         
        out *= self.scale

    @property
    def adjoint(self):
        return Convolution(self.domain, self.adjkernel, self.kernel)

def kernel(x, y):
    mean = [0.0, 0.25]
    std = [0.05, 0.05]
    return np.exp(-(((x-mean[0])/std[0])**2 + ((y-mean[1])/std[1])**2))
    
def adjkernel(x, y):
    return kernel(-x, -y)

# Continuous definition of problem
domain = odl.L2(odl.Rectangle([-1, -1], [1, 1]))
kernel_domain = odl.L2(odl.Rectangle([-2, -2], [2, 2]))

# Complicated functions to check performance
kernel = kernel_domain.element(kernel)
adjkernel = kernel_domain.element(adjkernel)

# Discretization parameters
n = 20
nPoints = np.array([n+1, n+1])
nPointsKernel = np.array([2*n+1, 2*n+1])

# Discretization spaces
disc_domain = odl.l2_uniform_discretization(domain, nPoints)
disc_kernel_domain = odl.l2_uniform_discretization(kernel_domain, nPointsKernel)

# Discretize the functions
disc_kernel = disc_kernel_domain.element(kernel)
disc_adjkernel = disc_kernel_domain.element(adjkernel)

# Create operator
conv = Convolution(disc_domain, disc_kernel, disc_adjkernel)

odl.diagnostics.OperatorTest(conv).run_tests()
