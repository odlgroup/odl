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
is used in fitting term.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.discr import Gradient
from odl.trafos import FourierTransform
import odl

__all__ = ('optimal_information_transport_solver',)


def padded_ft_op(space, padded_size):
    """Create zero-padding fft setting

    Parameters
    ----------
    space : the space needs to do FT
    padding_size : the percent for zero padding
    """
    padding_op = odl.ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = odl.trafos.FourierTransform(
        padding_op.range, halfcomplex=False, shift=shifts)

    return ft_op * padding_op


def optimal_information_transport_solver(gradS, I, niter, eps,
                                         sigma, impl='mp', callback=None):
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

    Parameters
    ----------
    gradS : `Operator`
        op.adjoint * odl.ResidualOperator(op, noise_proj_data),
        where op is a forward operator, noise_proj_data is the given data.
    I: `DiscreteLpElement`
        Fixed template deformed by the deformation.
    iter : 'int'
        The given maximum iteration number.
    eps : 'float'
        The given step size.
    sigma : 'float'
        The given regularization parameter. It's a
        wight on regularization-term side.
    impl : 'string'
        The given implementation method for mass preserving or not.
        The impl chooses 'mp' or 'nmp', where 'mp' means using
        mass-preserving method, and 'nmp' means non-mass-preserving method.
        Its defalt choice is 'mp'.
    callback : 'Class'
        Show the iterates.
    """
    # Initialize the determinant of Jacobian of inverse deformation
    DPhiJacobian = gradS.domain.one()

    # Create the space for inverse deformation
    pspace = gradS.domain.vector_field_space

    # Initialize the inverse deformation
    invphi = pspace.element(gradS.domain.points().T)

    # Create the identity mapping
    Id = gradS.domain.points().T

    # Create the temporary elements for update
    grad = Gradient(gradS.domain, method='forward', padding_method='symmetric')
    new_points = grad.range.element()
    v = grad.range.element()

    # Create poisson solver
    padded_size = 2 * gradS.domain.shape[0]
    padded_ft_shape_op = padded_ft_op(gradS.domain, padded_size)
    k2_values = sum((padded_ft_shape_op.range.points() ** 2).T)
    k2 = padded_ft_shape_op.range.element(np.maximum(np.abs(k2_values), 0.01))
    poisson_solver = padded_ft_shape_op.inverse * (1 / k2) * padded_ft_shape_op

#    ft = FourierTransform(gradS.domain)
#    k2_values = sum((ft.range.points() ** 2).T)
#    k2 = ft.range.element(np.maximum(np.abs(k2_values), 0.005))
#    poisson_solver = ft.inverse * (1 / k2) * ft

    # Begin iteration
    for _ in range(niter):
        # Convert the invphi into an array for efficient interpolation
        invphi_pts = np.empty([invphi.size, invphi[0].size])
        for i in range(invphi.size):
            invphi_pts[i] = invphi[i].ntuple.asarray()

        # Implementation for mass-preserving case
        if impl == 'mp':
            non_mp_def = I.space.element(
                I.interpolation(invphi_pts, bounds_check=False))
            PhiStarX = DPhiJacobian * non_mp_def

            grads = gradS(PhiStarX)
            tmp = grad(grads)
            for i in range(tmp.size):
                tmp[i] *= PhiStarX

        # Implementation for non-mass-preserving case
        if impl == 'nmp':
            PhiStarX = I.space.element(
                I.interpolation(invphi_pts, bounds_check=False))

            grads = gradS(PhiStarX)
            tmp = -grad(PhiStarX)
            for i in range(tmp.size):
                tmp[i] *= grads

        # Compute the minus L2 gradient
        u = sigma * grad(1 - np.sqrt(DPhiJacobian)) - 2 * tmp

        # Check the mass
        # print(np.sum(PhiStarX))

        # Compute the minus gradient in information metric
        for i in range(u.size):
            v[i] = poisson_solver(u[i])
            v[i] -= v[i][0]

        # Update the deformation for the inverse deformation and
        # its Jacobian's determinant
        for i in range(tmp.size):
            new_points[i] = Id[i] - eps * v[i].ntuple.asarray()

        # Update the inverse deformation
        for i in range(invphi.size):
            invphi[i] = invphi[i].space.element(
                invphi[i].interpolation(new_points, bounds_check=False))

        # Update the determinant of Jacobian of inverse deformation
        DPhiJacobian = np.exp(eps * grad.adjoint(v)) * gradS.domain.element(
            DPhiJacobian.interpolation(new_points, bounds_check=False))

        # Show intermediate result
        if callback is not None:
            callback(PhiStarX)

    return PhiStarX

