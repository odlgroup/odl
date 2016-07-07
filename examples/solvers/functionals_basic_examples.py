# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:36:12 2016

@author: johan79
"""

# This file should be removed before pull request! (Or filled with some more interesting examples)

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
#from builtins import super

import numpy as np

import odl
from odl.solvers.functional.default_functionals import (L1Norm, L2Norm, L2NormSquare)

# Discretization parameters
n = 4

# Discretized spaces
space = odl.uniform_discr([0, 0], [1, 1], [n, n])


print(space)

l1func = L1Norm(space)
l1prox = l1func.proximal(sigma=1.5)
l1conjFun = l1func.conjugate_functional


# Create phantom
phantom = space.element(np.random.standard_normal((n, n)))
# phantom = odl.util.shepp_logan(space, modified=True)*5+1

LogPhantom = np.log(phantom)

onevector = space.one() * 5

prox_phantom = l1prox(phantom)
l1conjFun_phantom = l1conjFun(phantom)

l2func = odl.solvers.functional.L2Norm(space)
l2prox = l2func.proximal(sigma=1.5)
l2conjFun = l2func.conjugate_functional
l2Grad = l2func.gradient

prox2_phantom = l2prox(phantom*10)
l2conjFun_phantom = l2conjFun(phantom/10)

l22 = odl.solvers.functional.L2NormSquare(space)
prox22 = l22.proximal(1)(phantom)

l22(phantom)
cf22 = l22.conjugate_functional(phantom)

l1func3 = -3*l1func

l1func3(phantom)
l1func(phantom)

F = l22

epsK = 1e-8

print(epsK)

(F(phantom+epsK*LogPhantom)-F(phantom))/epsK

LogPhantom.inner(F.gradient(phantom))

print(l22(phantom))

print((-2*l22)(phantom))

print((l22*2)(phantom))

x = space.element(np.random.standard_normal((n, n)))
#    y = space.element(np.random.rand(n,n))

scal = np.random.standard_normal()
F = odl.solvers.functional.L2Norm(space)

Fs = F*scal

print(Fs(x))


l2grad = (scal*F).gradient

print((scal*F).gradient(x))
print((F*scal).gradient(x))

z = l2grad(x)
