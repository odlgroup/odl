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

""":math:`L^p` type discretizations of function spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import matplotlib.pyplot as plt
from odl.discr import Gradient
from odl.trafos import FourierTransform

__all__ = ('optimal_information_transport_solver',)


def optimal_information_transport_solver(op, I, g, niter, eps, sigma,
                                         callback=None):
    DPhiJacobian = op.domain.one()

    grad = Gradient(op.domain, method='central')

    # We solve poisson using the fourier transform
    ft = FourierTransform(op.domain)
    k2_values = sum((ft.range.points() ** 2).T)
    k2 = ft.range.element(np.maximum(np.abs(k2_values), 0.01))
    poisson_solver = ft.inverse * (1 / k2) * ft

    for _ in range(niter):
        PhiStarX = DPhiJacobian * I

        tmp = grad(op.adjoint(op(PhiStarX) - g))
        for i in range(tmp.size):
            tmp[i] = PhiStarX * tmp[i]

        u = sigma * grad(1 - np.sqrt(DPhiJacobian)) - 2 * tmp

        # solve for v
        v = grad.range.element()
        for i in range(u.size):
            v[i] = poisson_solver(u[i])
            v[i] -= v[i][0]

        new_points = op.domain.points().T
        for i in range(tmp.size):
            new_points[i] -= eps * v[i].ntuple.asarray()

        # print(np.min(new_points), np.max(new_points))

        I.assign(op.domain.element(I.interpolation(new_points,
                                                   bounds_check=False)))
        DPhiJacobian = (np.exp(eps * grad.adjoint(v)) *
                        op.domain.element(DPhiJacobian.interpolation(
                            new_points, bounds_check=False)))

        if callback is not None:
            callback(I)

        # raise Exception


if __name__ == '__main__':
    import odl

    example = 3

    if example == 1:
        space = odl.uniform_discr(-1, 1, 100, interp='linear')

        op = odl.IdentityOperator(space)

        I = space.element(lambda x: np.exp(-20 * x**2))
        g = space.element(lambda x: np.exp(-20 * (x - 0.05)**2))

        I.show()
        g.show()

        niter = 5000
        eps = 0.01
        sigma = 0.1
        callback = odl.solvers.CallbackShow(display_step=10)

        optimal_information_transport_solver(op, I, g, niter, eps, sigma,
                                             callback)

    elif example == 2:
        space = odl.uniform_discr([-1] * 2, [1] * 2, [50] * 2, interp='linear')

        op = odl.IdentityOperator(space)

        I = space.element(lambda x: np.exp(-10 * x[0]**2 - 50 * x[1]**2))
        g = space.element(lambda x: np.exp(-50 * x[0]**2 - 10 * x[1]**2))

        I.show()
        g.show()

        niter = 1000
        eps = 0.5
        sigma = 0.1
        callback = odl.solvers.CallbackShow(display_step=20)

        optimal_information_transport_solver(op, I, g, niter, eps, sigma,
                                             callback)

    elif example == 3:
        I0name = 'c_highres.png'
        I1name = 'i_highres.png'

        I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
        I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]

        # Discrete reconstruction space: discretized functions on the rectangle
        # [-20, 20]^2 with 300 samples per dimension.
        space = odl.uniform_discr(
            min_corner=[-16, -16], max_corner=[16, 16], nsamples=[128, 128],
            dtype='float32', interp='linear')

        # Make a parallel beam geometry with flat detector
        # Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
        angle_partition = odl.uniform_partition(0, np.pi, 6)

        # Detector: uniformly sampled, n = 558, min = -30, max = 30
        detector_partition = odl.uniform_partition(-24, 24, 192)
        geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                               detector_partition)

        # ray transform aka forward projection. We use ASTRA CUDA backend.
        ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        # Create a discrete Shepp-Logan phantom (modified version)
        phantom = odl.util.submarine_phantom(space)
        phantom[:] = I0

        # Create projection data by calling the ray transform on the phantom
        proj_data = ray_trafo(phantom)
        proj_data += odl.util.white_noise(ray_trafo.range) * 3

        niter = 10000
        eps = 0.0002
        sigma = 1e3
        callback = (odl.solvers.CallbackShow('iterates', display_step=50) &
                    odl.solvers.CallbackPrintIteration())

        n = 15
        I = space.element(
            lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) ** (n / 2) / (8 ** n)))
        I *= phantom.norm() / I.norm()

        I[:] = I1
        I *= np.sum(phantom) / np.sum(I)

        phantom.show('phantom')
        I.show('template')

        optimal_information_transport_solver(ray_trafo, I, proj_data, niter,
                                             eps, sigma, callback)
