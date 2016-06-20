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

"""
Solver for the shape-based reconstruction using
optimal information transportation.

The model is:

min sigma * (1 - sqrt{DetJacInvPhi})^2 + (T(phi.I) - g)^2,
where phi.I := DetJacInvPhi * I(InvPhi) is a mass-preserving deformation.

Note that:
If T is an identity operator, the above model reduces for image matching.
If T is a forward projection operator, the above model is
for image reconstruction.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
from odl.discr import Gradient
from odl.trafos import FourierTransform
standard_library.install_aliases()

__all__ = ('optimal_information_transport_solver',)


def optimal_information_transport_solver(gradS, I, niter, eps,
                                         sigma, callback=None):

    DPhiJacobian = gradS.domain.one()

    grad = Gradient(gradS.domain, method='central')

    # ft = FourierTransform(op.domain)
    ft = FourierTransform(gradS.domain)
    k2_values = sum((ft.range.points() ** 2).T)
    k2 = ft.range.element(np.maximum(np.abs(k2_values), 0.01))
    poisson_solver = ft.inverse * (1 / k2) * ft

    for _ in range(niter):

#        PhiStarX = DPhiJacobian * I
        PhiStarX = I

        grads = gradS(PhiStarX)

#        tmp = grad(grads)
        tmp = grad(PhiStarX)

#        tmp = tmp.space.element([tp * PhiStarX for tp in tmp])
        tmp = tmp.space.element([tp * grads for tp in tmp])

#        u = sigma * grad(1 - np.sqrt(DPhiJacobian)) - 2 * tmp
        u = sigma * grad(1 - np.sqrt(DPhiJacobian)) + 2 * tmp

        # solve for v
        v = grad.range.element()

        for i in range(u.size):
            v[i] = poisson_solver(u[i])
            v[i] -= v[i][0]

        new_points = gradS.domain.points().T

        for i in range(tmp.size):
            new_points[i] -= eps * v[i].ntuple.asarray()

        I.assign(gradS.domain.element(I.interpolation(new_points,
                                                      bounds_check=False)))
        DPhiJacobian = np.exp(eps * grad.adjoint(v)) * gradS.domain.element(
            DPhiJacobian.interpolation(new_points, bounds_check=False))

        if callback is not None:
            callback(I)
