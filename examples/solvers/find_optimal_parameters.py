"""
Example of how to use optimization in order to pick reconstruction parameters.

In this example, we solve the tomographic inversion problem with different
regularizers (fbp, huber and tv) and pick the "best" regularization parameter
for each method w.r.t. a set of reference data.

To find the "best" parameter we use Powells method to optimize a figure of
merit, here the L2-distance to the true result.
"""

import numpy as np
import odl
import scipy


def optimal_parameters(reconstruction, fom, phantoms, data,
                       initial_param=0):
    """Find the optimal parameters for a reconstruction method.

    Notes
    -----
    For a forward operator :math:`A : X \to Y`, a reconstruction operator
     parametrized by :math:`\theta` is some operator :math:`R_\theta : Y \to X`
     such that

    .. math::
        R_\theta(A(x)) \approx x.

    The optimal choice of :math:`\theta` is given by

    .. math::
        \theta = \argmin_\theta fom(R(A(x) + noise), x)

    where :math:`fom : X \times X \to \mathbb{R}` is a Figure of Merit.


    Parameters
    ----------
    reconstruction : callable
        Function that takes two parameters:

            * data : The data to be reconstructed
            * parameters : Parameters of the reconstruction method

        The function should return the reconstructed image.
    fom : callable
        Function that takes two parameters:

            * reconstructed_image
            * true_image

        and returns a scalar Figure of Merit.
    phantoms : sequence
        True images.
    data : sequence
        The data to be reconstructed.
    initial_param : array-like
        Initial guess for the parameters.
    """

    def func(lam):
        # Function to be minimized by scipy
        return sum(fom(reconstruction(datai, lam), phantomi)
                   for phantomi, datai in zip(phantoms, data))

    # Pick resolution to fit the one used by the space
    tol= np.finfo(phantoms[0].space.dtype).resolution * 10

    initial_param = np.asarray(initial_param)

    if initial_param.size == 1:
        bracket = [initial_param - tol, initial_param + tol]
        result = scipy.optimize.minimize_scalar(func,
                                                bracket=bracket,
                                                tol=tol,
                                                bounds=None,
                                                options={'disp': False})
        return result.x
    else:
        # Use a gradient free method to find the best parameters
        parameters = scipy.optimize.fmin_powell(func, initial_param,
                                                xtol=tol,
                                                ftol=tol,
                                                disp=False)
        return parameters





# USER INPUT. Pick reconstruction: 'fbp', 'huber' or 'tv'
# 'fbp' is fast, 'huber' and 'tv' takes some time.

reconstruction_method = 'fbp'
signal_to_noise = 5.0

# Reconstruction space
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[128, 128],
    dtype='float32')

# Define forward operator
geometry = odl.tomo.parallel_beam_geometry(space)
ray_trafo = odl.tomo.RayTransform(space, geometry)

# Define true phantoms
phantoms = [odl.phantom.shepp_logan(space, modified=True),
            odl.phantom.derenzo_sources(space)]

# Define noisy data
data = []
for phantom in phantoms:
    noiseless_data = ray_trafo(phantom)
    noise_scale = (1 / signal_to_noise) * np.mean(noiseless_data)
    noise = noise_scale * odl.phantom.white_noise(ray_trafo.range)
    noisy_data = noiseless_data + noise
    data.append(noisy_data)

# Define the reconstruction method to use
if reconstruction_method == 'fbp':
    # Define the reconstruction operator for FBP reconstruction
    # The filter type here is 'Hann' and the parameter to optimize is the
    # frequency scaling (cut off frequency)
    def reconstruction(proj_data, lam):

        print('lam = {}'.format(lam))

        fbp_op = odl.tomo.fbp_op(ray_trafo,
                                 filter_type='Hann', frequency_scaling=1 / lam)
        return fbp_op(proj_data)

    initial_param = 1.0

elif reconstruction_method == 'huber':
    # Define the reconstruction operator for Huber regularized reconstruction
    # the parameter to optimize is (lam, sigma) with
    # sigma the size of the quadratic region
    #   (parametrizing the moreau envelope of smoothed l1 norm)
    # lam the regularization strength
    #   (multiplying the smoothed l1 in the objectiv function)
    # See lbfgs_tomograhpy_tv.py for more information.
    def reconstruction(proj_data, parameters):
        # Extract the separate parameters
        lam, sigma = parameters

        print('lam = {}, sigma = {}'.format(lam, sigma))

        # We do not allow negative paramters, so return a bogus result
        if lam <= 0 or sigma <= 0:
            return np.inf *  space.one()

        # Create data term ||Ax - b||_2^2
        l2_norm = odl.solvers.L2NormSquared(ray_trafo.range)
        data_discrepancy = l2_norm * (ray_trafo - proj_data)

        # Create regularizing functional huber(|grad(x)|)
        gradient = odl.Gradient(space)
        l1_norm = odl.solvers.GroupL1Norm(gradient.range)
        smoothed_l1 = odl.solvers.MoreauEnvelope(l1_norm, sigma=sigma)
        regularizer = smoothed_l1 * gradient

        # Create full objective functional
        obj_fun = data_discrepancy + lam * regularizer

        # Pick parameters
        maxiter = 30
        num_store = 5

        # Run the algorithm
        x = ray_trafo.domain.zero()
        odl.solvers.bfgs_method(
            obj_fun, x, maxiter=maxiter, num_store=num_store,
            hessinv_estimate=odl.ScalingOperator(
                    space, 1 / odl.power_method_opnorm(ray_trafo) ** 2))

        return x

    initial_param = [0.1, 0.05]

elif reconstruction_method == 'tv':
    # Define the reconstruction operator for TV regularized reconstruction
    # The parameter to optimize is the regularization strength
    # See chambolle_pock_tomography.py for more information.

    def reconstruction(proj_data, lam):
        lam = float(lam)

        print('lam = {}'.format(lam))

        # We do not allow negative paramters, so return a bogus result
        if lam <= 0:
            return np.inf * space.one()

        # Construct operators and functionals
        gradient = odl.Gradient(space)
        op = odl.BroadcastOperator(ray_trafo, gradient)

        g = odl.solvers.ZeroFunctional(op.domain)

        l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(proj_data)
        l1_norm = lam * odl.solvers.GroupL1Norm(gradient.range)
        f = odl.solvers.SeparableSum(l2_norm, l1_norm)

        # Select solver parameters
        op_norm = 1.1 * odl.power_method_opnorm(op)

        # Run the algorithm
        x = op.domain.zero()
        odl.solvers.chambolle_pock_solver(
            x, f, g, op, tau=1.0 / op_norm, sigma=1.0 / op_norm, niter= 200, gamma= 0.3)

        return x

    initial_param = 0.1
else:
    raise RuntimeError('unknown reconstruction method')


def fom(I0,I1):
    gradient = odl.Gradient(I0.space)
    return gradient(I0-I1).norm() + I0.space.dist(I0,I1)


# Find optimal lambda
optimal_parameters = optimal_parameters(reconstruction,  fom,
                                        phantoms, data,
                                        initial_param=initial_param)

reconstruction(data[0], initial_param).show(reconstruction_method+' , initial parameter')
reconstruction(data[1], initial_param).show(reconstruction_method+' , initial parameter')

reconstruction(data[0], optimal_parameters).show(reconstruction_method+' , optimal parameter')
reconstruction(data[1], optimal_parameters).show(reconstruction_method+' , optimal parameter')
