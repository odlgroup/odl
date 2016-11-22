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

"""Nuclear norm minimization using the Douglas-Rachford solver.

Solves the optimization problem

    min_{x_1, x_2} ||x_1 - g_1||_2^2 + ||x_2 - g_2||_2^2 +
                   lam || [grad(x_1), grad(x_2)] ||_*

where ``grad`` is the spatial gradient, ``g`` the given noisy data and
``|| . ||_*`` is the nuclear norm.

The nuclear norm introduces a coupling between the channels, and hence we
expect that edges should coincide in the optimal solution.
"""

import odl

# Create space that the function should live in. Here, we want a vector valued
# function, so we create the tuple of two spaces.
space = odl.uniform_discr(0, 1, 100)
pspace = odl.ProductSpace(space, 2)

# Create the gradient operator on the set of vector-valued functions.
# We select pad_mode='order1' so that we have a Neumann-style boundary
# condition. Here we assume the gradient is continuous at the boundary.
gradient = odl.Gradient(space, pad_mode='order1')
pgradient = odl.DiagonalOperator(gradient, 2)

# Create the data. The first part is a linear function, the second is a step
# function at x=0.6
data = pspace.element([lambda x: x, lambda x: x > 0.6])
data.show('data')

# Create functionals for the data discrepancy (L2 squared) and for the
# regularizer (nuclear norm). The nuclear norm is defined on the range of
# the vectorial gradient, which is vector valued.
l2err = odl.solvers.L2NormSquared(pspace).translated(data)
nuc_norm = 0.02 * odl.solvers.NuclearNorm(pgradient.range)

# Assemble operators and functionals for the solver routine
lin_ops = [odl.IdentityOperator(pspace), pgradient]
g = [l2err, nuc_norm]

# The solver we want to use also takes an additional functional f which can be
# used to enforce bounds constraints and other prior information. Here we lack
# prior information so we set it to zero.
f = odl.solvers.ZeroFunctional(pspace)

# Create a callback that shows the current function value and also shows the
# iterate graphically every 20:th step.
func = f + l2err + nuc_norm * pgradient
callback = (odl.solvers.CallbackPrint(func) &
            odl.solvers.CallbackShow(display_step=20))

# Solve the problem. Here the parameters are chosen in order to ensure
# convergence, see the documentation for further information.
# We select the data as an initial guess.
x = data.copy()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=1e-2, sigma=[1.0, 1e-3],
                                niter=2000, callback=callback)
