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
import numpy as np
import matplotlib.pyplot as plt
import odl
from odl.discr import Gradient, Divergence
from odl.deform.mass_preserving import geometric_deform, mass_presv_deform
from odl.phantom import white_noise, disc_phantom, submarine, shepp_logan
from odl.deform.linearized import _linear_deform
standard_library.install_aliases()    


def optimal_information_transport_solver(gradS, I, niter, eps, lamb,
                                         inverse_inertia_op, impl='mp',
                                         callback=None):
    """
    Solver for the shape-based reconstruction using
    optimal information transportation.
    The model is:
    min lamb * (1 - sqrt{DetJacInvPhi})^2 + (T(phi.I) - g)^2,
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
    lamb : 'float'
        The given regularization parameter. It's a
        wight on regularization-term side.
    inverse_inertia_op : 'Operator'
        The implemantation of kernel (poisson or RKHS).
    impl : 'string'
        The given implementation method for mass preserving or not.
        The impl chooses 'mp' or 'nmp', where 'mp' means using
        mass-preserving method, and 'nmp' means non-mass-preserving method.
        Its defalt choice is 'mp'.
    callback : 'Class'
        Show the iterates.
    """
    # Create space of image
    image_space = gradS.domain

    # Initialize the determinant of Jacobian of inverse deformation
    DPhiJacobian = image_space.one()

    # Initialize the non-mass-preserving deformed template
    non_mp_deform_I = I

    # Create the space for inverse deformation
    pspace = image_space.vector_field_space

    # Create the temporary elements for update
    grad = Gradient(gradS.domain, method='forward', pad_mode='symmetric')
    div = Divergence(pspace, method='forward', pad_mode='symmetric')
    v = grad.range.element()

    # Store energy
    E = []
    kE = len(E)
    E = np.hstack((E, np.zeros(niter)))

    # Begin iteration
    for k in range(niter):

        E[k+kE] = np.asarray(lamb * (np.sqrt(DPhiJacobian) - 1) ** 2).sum()

        # Implementation for mass-preserving case
        if impl == 'mp':
            PhiStarI = DPhiJacobian * non_mp_deform_I
            grads = gradS(PhiStarI)
            E[k+kE] += np.asarray(grads**2).sum()
            tmp = grad(grads)
            for i in range(tmp.size):
                tmp[i] *= PhiStarI
            # Show intermediate result
            if callback is not None:
                callback(PhiStarI)

        # Implementation for non-mass-preserving case
        if impl == 'nmp':
            PhiStarI = non_mp_deform_I
            grads = gradS(PhiStarI)
            E[k+kE] += np.asarray(grads**2).sum()
            tmp = - grad(PhiStarI)
            for i in range(tmp.size):
                tmp[i] *= grads
            # Show intermediate result
            if callback is not None:
                callback(PhiStarI)

        # Compute the minus L2 gradient
        u = - lamb * grad(np.sqrt(DPhiJacobian)) - tmp
        
        v = inverse_inertia_op(u)

        # Check the mass
        # print(np.sum(PhiStarX))

        # Update the non-mass-preserving deformation of template
        non_mp_deform_I = image_space.element(
            _linear_deform(non_mp_deform_I, - eps * v))

        # Update the determinant of Jacobian of inverse deformation
        DPhiJacobian = np.exp(- eps * div(v)) * image_space.element(
            _linear_deform(DPhiJacobian, - eps * v))

    return PhiStarI, E


# Implement OIT solver based on Fisher-Rao distance case, need mass-preserving
def optimal_information_transport_solver2(I0, I1, niter, eps, lamb,
                                          inverse_inertia_op, callback=None):
    """
    Solver for image matching using optimal information transport
    with Fisher-Rao distance.
    The model is:

    min lamb * (1 - sqrt{DetJacInvPhi})^2 + (sqrt(phi.I_0) - sqrt(I_1))^2,

    where phi.I_0 := DetJacInvPhi * I_0(InvPhi) is a mass-preserving
    deformation.

    Parameters
    ----------
    I_0: `DiscreteLpElement`
        The source image, i.e., fixed template deformed by the deformation.
    I_1: `DiscreteLpElement`
        The target image.
    niter : 'int'
        The given maximum iteration number.
    eps : 'float'
        The given step size.
    lamb : 'float'
        The given regularization parameter. It's a
        wight on regularization-term side.
    inverse_inertia_op : 'Operator'
        The implemantation of kernel (poisson or RKHS).
    callback : 'Class'
        Show the iterates.
    """
    # Nomalize the mass of I0 and I1
    I0 *= np.linalg.norm(I1, 'fro')/np.linalg.norm(I0, 'fro')

    # Get the space of I0
    domain = I0.space
    
    # Initialize the determinant of Jacobian of inverse deformation
    DPhiJacobian = domain.one()

    # Create the space for inverse deformation
    pspace = domain.vector_field_space

    # Create the temporary elements for update
    grad_op = Gradient(domain, method='forward', pad_mode='symmetric')
    div_op = Divergence(pspace, method='forward', pad_mode='symmetric')
    v = grad_op.range.element()

    # Initialize the non-mass-preserving deformed template
    non_mp_deform_I0 = I0

    # Store energy
    E = []
    kE = len(E)
    E = np.hstack((E, np.zeros(niter)))

    # Begin iteration
    for k in range(niter):

        # Compute the energy of the regularization term
        E[k+kE] = np.asarray(lamb * (np.sqrt(DPhiJacobian) - 1) ** 2).sum()

        # Implementation for mass-preserving case
        PhiStarI0 = DPhiJacobian * non_mp_deform_I0

        # Show intermediate result
        if callback is not None:
            callback(PhiStarI0)

        # For Fisher-Rao distance
        sqrt_mp_I0 = np.sqrt(PhiStarI0)
        sqrt_I1 = np.sqrt(I1)
        grad_sqrt_mp_I0 = grad_op(sqrt_mp_I0)
        grad_sqrt_I1 = grad_op(sqrt_I1)
        
        # Compute the energy of the data fitting term         
        E[k+kE] += np.asarray((sqrt_mp_I0 - sqrt_I1)**2).sum()

        # Compute the L2 gradient of the data fiiting term
        grad_fitting = grad_op.range.zero()
        for i in range(grad_op.range.size):
            grad_fitting[i] = sqrt_I1 * grad_sqrt_mp_I0[i] - \
                sqrt_mp_I0 * grad_sqrt_I1[i]
                
        # Compute the minus L2 gradient
        u = - lamb * grad_op(np.sqrt(DPhiJacobian)) - grad_fitting

        # Check the mass
        # print(np.sum(PhiStarX))

        v = inverse_inertia_op(u)

        non_mp_deform_I0 = domain.element(
            _linear_deform(non_mp_deform_I0, - eps * v))

        # Update the determinant of Jacobian of inverse deformation
        DPhiJacobian = np.exp(- eps * div_op(v)) * domain.element(
            _linear_deform(DPhiJacobian, - eps * v))

    return PhiStarI0, E


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


def fitting_kernel_ft(kernel):
    """Compute the n-D Fourier transform of the discrete kernel ``K``.

    Calculate the n-D Fourier transform of the discrete kernel ``K`` on the
    image grid points {y_i} to its reciprocal points {xi_i}.
    """
    kspace = odl.ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return vectorial_ft_op(discretized_kernel)


def poisson_kernel_ft():
    """Compute the n-D Fourier transform of the inverse Laplacian.

    Calculate the n-D Fourier transform of the inverse Laplacian on the
    image grid points {y_i} to its reciprocal points {xi_i}.
    """
    k2_values = sum((padded_ft_op.range.points() ** 2).T)
    k2 = padded_ft_op.range.element(np.maximum(np.abs(k2_values), 0.0000001))
    inv_k2 = 1 / k2
    inv_k2 = padded_ft_op.range.element(np.minimum(np.abs(inv_k2), 200))

    kspace = odl.ProductSpace(padded_ft_op.range, padded_ft_op.range.ndim)
    return kspace.element([inv_k2 for _ in range(space.ndim)])


def inverse_inertia_op(impl3):
    """Create inverse inertia operator

    Parameters
    ----------
    impl3 : `string`
        implementation method, solving poisson equation or using RKHS
    """
    temp = 2 * np.pi
    if impl3 == 'poisson':
        return temp * vectorial_ft_op.inverse * poisson_kernel_ft * vectorial_ft_op
    
    elif impl3 == 'rkhs':
        return temp * vectorial_ft_op.inverse * ft_kernel_fitting * vectorial_ft_op


def snr(signal, noise, impl):
    """Compute the signal-to-noise ratio.
    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).
    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')


# Kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


# Give input images
I0name = './pictures/c_highres.png'
I1name = './pictures/i_highres.png'
# I0name = './pictures/handnew1.png'
# I1name = './pictures/DS0002AxialSlice80.png'
#I0name = './pictures/handnew1.png'
#I1name = './pictures/handnew2.png'
#I0name = './pictures/v.png'
#I1name = './pictures/j.png'

# Get digital images
#I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
#I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]
I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)

# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')

# Give the number of directions
num_angles = 60

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0, np.pi, num_angles,
                                        nodes_on_bdry=[(True, False)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, 384)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                       detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Create the ground truth as the given image
ground_truth = space.element(I0)

# Create the ground truth as the Shepp-Logan phantom
# ground_truth = shepp_logan(space, modified=True)

# # Create the ground truth as the submarine phantom
# ground_truth = odl.util.submarine_phantom(space, smooth=True, taper=50.0)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise = odl.phantom.white_noise(ray_trafo.range) * 0.0

# Add white Gaussion noise from file
#noise = ray_trafo.range.element(np.load('noise_20angles.npy'))

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Create the noisy data from file
#noise_proj_data = ray_trafo.range.element(
#    np.load('noise_proj_data_20angles_snr_4_98.npy'))

# Compute the signal-to-noise ratio in dB
snr = snr(proj_data, noise, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Maximum iteration number
niter = 1000

callback = odl.solvers.CallbackShow(
    'iterates', display_step=50) & odl.solvers.CallbackPrintIteration()

# Create the template as the given image
template = space.element(I1)

# # Create the template as the disc phantom
# template = odl.util.disc_phantom(space, smooth=True, taper=50.0)

## Create the template for Shepp-Logan phantom
#deform_field_space = space.vector_field_space
#disp_func = [
#    lambda x: 16.0 * np.sin(np.pi * x[0] / 40.0),
#    lambda x: 16.0 * np.sin(np.pi * x[1] / 36.0)]
#deform_field = deform_field_space.element(disp_func)
#template = space.element(geometric_deform(
#    shepp_logan(space, modified=True), deform_field))

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'nmp', 'mp' means mass-preserving method,
# 'nmp' means non-mass-preserving method
impl1 = 'mp'

# Implementation method for image matching or image reconstruction,
# impl chooses 'matching' or 'reconstruction', 'matching' means image matching,
# 'reconstruction' means image reconstruction
impl2 = 'reconstruction'

# Implementation method with Klas Modin or rkhs
# impl chooses 'poisson' or 'rkhs', 'poisson' means using poisson solver,
# 'rkhs' means using V-gradient
impl3 = 'rkhs'

# Normalize the template's density as the same as the ground truth if consider
# mass preserving method
if impl1 == 'mp':
#    template *= np.sum(ground_truth) / np.sum(template)
    template *= np.linalg.norm(ground_truth, 'fro')/ \
        np.linalg.norm(template, 'fro')


ground_truth.show('Ground truth')
template.show('Template')

# For image reconstruction
if impl2 == 'reconstruction':
    # Give step size for solver
    eps = 0.005

    # Give regularization parameter
    lamb = 0.05

    # Fix the sigma parameter in the kernel
    sigma = 2.0

    # Create the forward operator for image reconstruction
    op = ray_trafo

    # Create the gradient operator for the L2 functional
    gradS = op.adjoint * odl.ResidualOperator(op, noise_proj_data)

    padded_size = 2 * gradS.domain.shape[0]
    padded_ft_op = padded_ft_op(gradS.domain, padded_size)
    vectorial_ft_op = odl.DiagonalOperator(padded_ft_op, gradS.domain.ndim)

    # Compute Fourier trasform of the kernel function in data matching term
    ft_kernel_fitting = fitting_kernel_ft(kernel)

    # Compute Fourier trasform of the inverse Laplacian
    poisson_kernel_ft = poisson_kernel_ft()

    # Implement different gradient (poisson or RKHS)
    inv_inertia_op = inverse_inertia_op(impl3)

    # Compute by optimal information transport solver
    rec_result, E = optimal_information_transport_solver(
        gradS, template, niter, eps, lamb, inv_inertia_op, impl1, callback)

    # Show result
    rec_proj_data = op(rec_result)
    # Plot the results of interest
    plt.figure(1, figsize=(20, 10))
    plt.clf()

    plt.subplot(2, 3, 1)
    plt.imshow(np.rot90(template), cmap='bone'), plt.axis('off')
    plt.title('template')

    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(rec_result), cmap='bone'), plt.axis('off')
    plt.title('rec_result')
    
    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(ground_truth), cmap='bone'), plt.axis('off')
    plt.title('Ground truth')
    
    plt.subplot(2, 3, 4)
    plt.plot(np.asarray(proj_data)[0], 'b', np.asarray(noise_proj_data)[0],
             'r', np.asarray(rec_proj_data)[0], 'g'), plt.axis([0, 191, -3, 10]), plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(np.asarray(proj_data)[5], 'b', np.asarray(noise_proj_data)[5],
             'r', np.asarray(rec_proj_data)[5], 'g'), plt.axis([0, 191, -3, 10]), plt.grid(True)
    plt.subplot(2, 3, 6)
    plt.plot(np.asarray(proj_data)[10], 'b', np.asarray(noise_proj_data)[10],
             'r', np.asarray(rec_proj_data)[10], 'g'), plt.axis([0, 191, -3, 10]), plt.grid(True)


# For image matching
if impl2 == 'matching':
    # Give step size for solver
    eps = 0.0025

    # Give regularization parameter
    lamb = 0.05

    # Fix the sigma parameter in the kernel
    sigma = 5.0

    # Create the forward operator for image matching
    op = odl.IdentityOperator(space)

    # Create the gradient operator for the L2 functional
    gradS = op.adjoint * odl.ResidualOperator(op, ground_truth)

    padded_size = 2 * space.shape[0]
    padded_ft_op = padded_ft_op(space, padded_size)
    vectorial_ft_op = odl.DiagonalOperator(padded_ft_op, space.ndim)

    # Compute Fourier trasform of the kernel function in data matching term
    ft_kernel_fitting = fitting_kernel_ft(kernel)
    
    # Compute Fourier trasform of the inverse Laplacian
    poisson_kernel_ft = poisson_kernel_ft()

    # Implement different gradient (poisson or RKHS)
    inv_inertia_op = inverse_inertia_op(impl3)

    # Compute by OIT solver based on L2 norm distance
    rec_result, E = optimal_information_transport_solver(
        gradS, template, niter, eps, lamb, inv_inertia_op, impl1, callback)

#    # Compute by OIT solver based on Fisher-Rao distance
#    rec_result, E = optimal_information_transport_solver2(
#        template, ground_truth, niter, eps, lamb, inv_inertia_op, callback)

    plt.figure(1, figsize=(20, 10))
    plt.clf()
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.rot90(template), cmap='bone', vmin=I1.min(), vmax=I1.max()), plt.axis('off')
    plt.title('template')

    plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(rec_result), cmap='bone', vmin=I1.min(), vmax=I1.max()), plt.axis('off')
    plt.title('match_result')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.rot90(ground_truth), cmap='bone', vmin=I1.min(), vmax=I1.max()), plt.axis('off')
    plt.title('Ground truth')

    plt.figure(2, figsize=(8, 1.5))
    plt.clf()
    plt.plot(E)
    plt.ylabel('Energy')
    # plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.grid(True)
