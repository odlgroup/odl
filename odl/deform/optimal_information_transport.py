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
Example of shape-based image reconstruction
using optimal information transformation.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
from odl.operator.operator import Operator
from builtins import super
import numpy as np
import matplotlib.pyplot as plt
from odl.discr import Gradient
from odl.trafos import FourierTransform
standard_library.install_aliases()

__all__ = ('optimal_information_transport_solver',)


def optimal_information_transport_solver(gradS, I, niter, eps,
                                         sigma, callback=None):

    DPhiJacobian = gradS.domain.one()

    grad = Gradient(gradS.domain, method='central')

    # We solve poisson using the fourier transform
    # ft = FourierTransform(op.domain)
    ft = FourierTransform(gradS.domain)
    k2_values = sum((ft.range.points() ** 2).T)
    k2 = ft.range.element(np.maximum(np.abs(k2_values), 0.01))
    poisson_solver = ft.inverse * (1 / k2) * ft

    for _ in range(niter):
        PhiStarX = DPhiJacobian * I

        grads = gradS(PhiStarX)

        tmp = grad(grads)

        tmp = tmp.space.element([tp * PhiStarX for tp in tmp])

        u = sigma * grad(1 - np.sqrt(DPhiJacobian)) - 2 * tmp

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


# Code for the example starts here


def SNR(signal, noise, impl='general'):
    """Compute the signal-to-noise ratio.
    This compute::
        impl='general'
            SNR = s_power / n_power
        impl='dB'
            SNR = 10 * log10 (
                s_power / n_power)
    Parameters
    ----------
    signal : projection
    noise : white noise
    impl : implementation method
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal)/signal.size
        ave2 = np.sum(noise)/noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            snr = s_power/n_power
        else:
            snr = 10.0 * np.log10(s_power/n_power)

        return snr

    else:
        return float('inf')


if __name__ == '__main__':
    import odl

    I0name = '../../ddmatch/Example3 letters/c_highres.png'
    I1name = '../../ddmatch/Example3 letters/i_highres.png'

    I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
    I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]

    # Discrete reconstruction space: discretized functions on the rectangle
    space = odl.uniform_discr(
        min_corner=[-16, -16], max_corner=[16, 16], nsamples=[128, 128],
        dtype='float32', interp='linear')

    # Make a parallel beam geometry with flat detector
    # Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
    # Create the uniformly distributed directions
    angle_partition = odl.uniform_partition(0, np.pi, 6)

    # Create 2-D projection domain
    # The length should be 1.5 times of that of the reconstruction space
    detector_partition = odl.uniform_partition(-24, 24, 192)

    # Create 2-D parallel projection geometry
    geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                           detector_partition)

    # ray transform aka forward projection. We use ASTRA CUDA backend.
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    # Create the ground truth as the given image
    ground_truth = space.element(I0)

    # Create projection data by calling the ray transform on the phantom
    proj_data = ray_trafo(ground_truth)

    # Add white Gaussion noise onto the noiseless data
    noise = odl.util.white_noise(ray_trafo.range) * 3

    # Create the noisy projection data
    noise_proj_data = proj_data + noise

    # Compute the signal-to-noise ratio in dB
    snr = SNR(proj_data, noise, impl='dB')

    # Output the signal-to-noise ratio
    print('snr = {!r}'.format(snr))

    # Maximum iteration number
    niter = 2000

    callback = odl.solvers.CallbackShow(
        'iterates', display_step=50) & odl.solvers.CallbackPrintIteration()

    template = space.element(I1)
    template *= np.sum(ground_truth) / np.sum(template)

    ground_truth.show('phantom')
    template.show('template')



#    # For image reconstruction
#    eps = 0.002
#    sigma = 1e2
#
#    # Create the forward operator for image reconstruction
#    op = ray_trafo
#
#    # Create the gradient operator for the L2 functional
#    gradS = op.adjoint * odl.ResidualOperator(op, noise_proj_data)
#
#    # Compute by optimal information transport solver
#    optimal_information_transport_solver(gradS, template, niter,
#                                         eps, sigma, callback)



    # For image matching
    eps = 0.2
    sigma = 1
    # Create the forward operator for image matching
    op = odl.IdentityOperator(space)

    # Create the gradient operator for the L2 functional
    gradS = op.adjoint * odl.ResidualOperator(op, ground_truth)

    # Compute by optimal information transport solver
    optimal_information_transport_solver(gradS, template, niter,
                                         eps, sigma, callback)
