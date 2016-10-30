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

"""Total variation MRI inversion using the Douglas-Rachford solver.

Solves the optimization problem

    min_{0 <= x <= 1} ||Ax - g||_2^2 + lam || |grad(x)| ||_1

where ``A`` is a simplified MRI imaging operator, ``grad`` is the spatial
gradient and ``g`` the given noisy data.
"""

# --- THIS EXAMPLE IS CURRENTLY BROKEN UNTILL ISSUE #590 IS FIXED ---

import numpy as np
import odl

# Parameters
n = 256
subsampling = 0.5  # propotion of data to use
lam = 0.003

space = odl.uniform_discr(0, 10, 10)
pspace = odl.ProductSpace(space, 2)

identity = odl.IdentityOperator(pspace)
l2err = odl.solvers.L1Norm(pspace)

gradient = odl.Gradient(space, pad_mode='order1')
pgradient = odl.DiagonalOperator(gradient, 2)
nuc_norm = odl.solvers.NuclearNorm(pgradient.range)

rhs = pspace.element([lambda x: x>3, lambda x: x>6])
rhs.show()
# rhs += odl.phantom.white_noise(pspace) * 0.1

# Assemble all operators
lin_ops = [identity, 0.1 * pgradient]

# Create functionals as needed
const = 0.001

g = [l2err.translated(rhs),
     const * nuc_norm]
f = odl.solvers.ZeroFunctional(pspace)


def printval(x):
    print(l2err(x-rhs) + const * nuc_norm(pgradient(x)))

# Solve
x = pspace.zero()
callback = (odl.solvers.CallbackShow(display_step=1) &
            odl.solvers.CallbackPrintIteration() &
            printval)
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.2, sigma=[1.0, 1.0],
                                niter=500, callback=callback)
