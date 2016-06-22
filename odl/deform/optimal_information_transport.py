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
Shape-based reconstruction using optimal information transportation.

The Fisher-Rao metric is used in regularization term. And L2 data matching
term is used in fitting term.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
from odl.discr import Gradient
from odl.trafos import FourierTransform
from odl.deform import LinearDeformation, MassPreservingLinearDeformation
standard_library.install_aliases()

__all__ = ('optimal_information_transport_solver',)


def optimal_information_transport_solver(gradS, I, niter, eps,
                                         sigma, callback=None):
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
    invphi = MassPreservingLinearDeformation.identity(gradS.domain)

    grad = Gradient(gradS.domain, method='central')

    v = grad.range.element()

    ft = FourierTransform(gradS.domain)
    k2_values = sum((ft.range.points() ** 2).T)
    k2 = ft.range.element(np.maximum(np.abs(k2_values), 0.01))
    poisson_solver = ft.inverse * (1 / k2) * ft

    for _ in range(niter):

        # implementation for mass-preserving case
        PhiStarX = invphi(I)

        grads = gradS(PhiStarX)
        tmp = grad(grads)
        for i in range(tmp.size):
            tmp[i] *= PhiStarX

        # tmp = tmp.space.element([tp * PhiStarX for tp in tmp])
        u = sigma * grad(1 - np.sqrt(invphi.vector_field_jacobian)) - 2 * tmp

        for i in range(u.size):
            v[i] = poisson_solver(u[i])
            v[i] -= v[i][0]

        update_deform = LinearDeformation(- eps * v)
        invphi = invphi.compose(update_deform)

        if callback is not None:
            callback(PhiStarX)
