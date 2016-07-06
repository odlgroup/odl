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

"""Total variation denoising using the forward-backward primal dual solver.

Solves the optimization problem

    min_{0 <= x <= 255}  1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

where ``grad`` is the spatial gradient and ``g`` the given noisy data.
"""

import numpy as np
import scipy.misc
import odl
import odl.solvers as odlsol

# Parameters
n = 256

# Create a space
space = odl.uniform_discr([0, 0], [n, n], [n, n])

# Load image and noise
data = space.element(np.rot90(scipy.misc.ascent()[::2, ::2], 3))
noise = odl.phantom.white_noise(space) * 10.0

# Create noisy data
noisy_data = data + noise
data.show('Original data')
noisy_data.show('Noisy convolved data')


# Gradient for TV regularization
gradient = odl.Gradient(space)

# Assemble the linear operators. Here the TV-term is mapped into a composition
# of the 1-norm and the gradient. See the documentation of the solver
# `forward_backward_pd` for the general form of the problem.
lin_ops = [gradient]

# Create proximals for the convex conjugate of the 1-norm and the bound
# constrains.
prox_cc_g = [odlsol.proximal_cconj_l1(gradient.range, lam=1e1,
                                      isotropic=False)]
prox_f = odlsol.proximal_box_constraint(space, 0, 255)

# This gradient encodes the differentiable term(s) of the goal functional,
# which corresponds to the "forward" part of the method. In this example the
# differentiable part is the 2-norm squared.
grad_h = odl.ResidualOperator(odl.IdentityOperator(space), noisy_data)

# Create initial guess for the solver.
x = noisy_data.copy()

# Used to display intermediate results and print iteration number.
callback = (odl.solvers.CallbackShow(display_step=20, clim=[0, 255]) &
            odl.solvers.CallbackPrintIteration())

# Call the solver
odlsol.forward_backward_pd(x, prox_f, prox_cc_g, lin_ops, grad_h, tau=1.0,
                           sigma=[0.01], niter=1000, callback=callback)
