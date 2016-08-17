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

"""Poisson's problem using the ProxImaL solver.

Solves the optimization problem

    min_x  10 ||laplacian(x) - g||_2^2 + || |grad(x)| ||_1

Where ``laplacian`` is the spatial Laplacian, ``grad`` the spatial
gradient and ``g`` is given noisy data.
"""

import numpy as np
import odl
import proximal

# Create space defined on a square from [0, 0] to [100, 100] with (100 x 100)
# points
space = odl.uniform_discr([0, 0], [100, 100], [100, 100])

# Create ODL operator for the Laplacian
laplacian = odl.Laplacian(space)

# Create right hand side
phantom = odl.phantom.shepp_logan(space, modified=True)
phantom.show('original image')
rhs = laplacian(phantom)
rhs += odl.phantom.white_noise(space) * np.std(rhs) * 0.1
rhs.show('rhs')

# Convert laplacian to ProxImaL operator
proximal_lang_laplacian = odl.as_proximal_lang_operator(laplacian)

# Convert to array
rhs_arr = rhs.asarray()

# Set up optimization problem
x = proximal.Variable(space.shape)
funcs = [10 * proximal.sum_squares(proximal_lang_laplacian(x) - rhs_arr),
         proximal.norm1(proximal.grad(x))]

# Solve the problem using ProxImaL
prob = proximal.Problem(funcs)
prob.solve(verbose=True)

# Convert back to odl and display result
result_odl = space.element(x.value)
result_odl.show('result from ProxImaL')
