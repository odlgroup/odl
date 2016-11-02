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

    min ||x - g||_2^2 + lam || grad(x) ||_*

where ``grad`` is the spatial gradient, ``g`` the given noisy data and
``|| . ||_*`` is the nuclear-norm.
"""

import odl

space = odl.uniform_discr(0, 1, 100)
pspace = odl.ProductSpace(space, 2)

identity = odl.IdentityOperator(pspace)

gradient = odl.Gradient(space, pad_mode='order1')
pgradient = odl.DiagonalOperator(gradient, 2)

rhs = pspace.element([lambda x: x[0], lambda x: x[0] > 0.6])
rhs.show()

# Assemble all operators
lin_ops = [identity, pgradient]

# Create functionals as needed
l2err = odl.solvers.L2NormSquared(pspace)
nuc_norm = odl.solvers.NuclearNorm(pgradient.range)

const = 0.02

g = [l2err.translated(rhs),
     const * nuc_norm]
f = odl.solvers.ZeroFunctional(pspace)
func = f + l2err + const * nuc_norm * pgradient

# Solve
x = rhs.copy()
callback = (odl.solvers.CallbackShow(display_step=20) &
            odl.solvers.CallbackPrint(func))
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=1e-2, sigma=[1.0, 1e-3],
                                niter=200, callback=callback)
