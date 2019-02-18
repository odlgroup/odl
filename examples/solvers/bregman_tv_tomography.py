"""Example of Bregman-TV in tomography.

The example solves an inverse problem in tomography using Bregman iterations
for a total variation (TV) regularized problem. To this end, let ``A`` denote
the forward operator, ``g`` denote the data, ``D(Ax, g)`` denote data
discrepancy functional, and ``alpha * R(x)`` denote the regularizing
functional. In this particular example, the variational problem considered is
the TV-problem

    min_x  1/2 ||A(x) - g||_2^2 + lam || |grad(x)| ||_1.

For a functional ``F`` the Bregman functional is defined as

    D_F^p(x, z) = F(x) - F(z) - <p, x-z>,

where ``p`` is a subgradient at the point ``x``, i.e., ``p in dF(x)``.

The Bregman iterations can be written as

    x_{k+1} in argmin D(Ax, g) + lam * D_R^{p_k}(x, x_k)
    p_{k+1} - p_k in -(1/lam) * A^* dD(Ax_k, g).

Note that the first step involves solving an optimization problem, which is
(normally) done using an iterative method. This means that the solution method
contains a set of inner and outer iterations.

The Bregman iterations need to be stopped after a finite number of outer
iterations. Otherwise the noise from ``g`` will start to deteriorate the
result. In this example this is illustrated by doing a few outer iterations to
many, and displaying the result after each outer iteration.

For more details on Bregman iterations, see, e.g., [Bur2016].

Referenses
----------
[Bur2016] Burger, M. *Bregman distances in inverse problems and partial
differential equations*. In: Hiriart-Urruty, J B, Korytowski A, Maurer H,
Szymkat M. Advances in Mathematical Modeling, Optimization and Optimal Control.
Springer, 2016, pp 3--33.
https://doi.org/10.1007/978-3-319-30785-5_2
https://arxiv.org/abs/1505.05191
"""

import odl

# Reconstruction space
reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                               shape=[128, 128], dtype='float32')

# Make a parallel beam geometry with flat detector, and create ray transform
geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=100)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Create phantom, forward project to create sinograms, and add 10% noise
discr_phantom = odl.phantom.shepp_logan(reco_space, modified=True)
noise_free_data = ray_trafo(discr_phantom)
noise = odl.phantom.white_noise(ray_trafo.range)
noise *= 0.10 / noise.norm() * noise_free_data.norm()
data = noise_free_data + noise

# Components for variational problem: l2-squared data matching and isotropic
# TV-regularization
l2_norm = 0.5 * odl.solvers.L2NormSquared(ray_trafo.range).translated(data)
gradient = odl.Gradient(reco_space)
reg_param = 0.3
l12_norm = reg_param * odl.solvers.GroupL1Norm(gradient.range)

# Assemble functionals and operators for the optimization algorithm
f = odl.solvers.ZeroFunctional(reco_space)  # No f functional used, set to zero
g = [l2_norm, l12_norm]
L = [ray_trafo, gradient]

# Estimated operator norms, which are used to ensure that we fulfill the
# convergence criteria of the optimization algorithm
ray_trafo_norm = odl.power_method_opnorm(ray_trafo, maxiter=20)
gradient_norm = odl.power_method_opnorm(gradient, maxiter=20)

# Parameters for the optimization method; tuned in order to reduce the number
# of inner iterations needed to solve the first step in the Bregman iterations
niter_inner = 200
tau = 0.01  # Step size for the primal variable
sigma_ray_trafo = 45.0 / ray_trafo_norm ** 2  # Step size for dual variable
sigma_gradient = 45.0 / gradient_norm ** 2  # Step size for dual variable
sigma = [sigma_ray_trafo, sigma_gradient]

# The reconstruction looks nice after about 5 outer iterations; set total
# number of outer iterations to 7 to show what happens if one does to many
niter_bregman = 7

# Create initial guess and initial subgradient
x = reco_space.zero()
p = reco_space.zero()

# This defines the outer Bregman iterations
for breg_iter in range(niter_bregman):
    print('Outer Bregman Iteration: {}'.format(breg_iter))

    # Create the affine part of the Bregman functional
    constant = l12_norm(gradient(x))
    linear_part = reg_param * odl.solvers.QuadraticForm(vector=-p,
                                                        constant=constant)

    callback_inner = odl.solvers.CallbackPrintIteration(step=50)

    # Inner iterations; x is updated in-place with the consecutive iterates
    odl.solvers.forward_backward_pd(
        x=x, f=f, g=g, L=L, h=linear_part, tau=tau, sigma=sigma,
        niter=niter_inner, callback=callback_inner)

    # Update the subgradient
    p -= (1 / reg_param) * ray_trafo.adjoint(l2_norm.gradient(ray_trafo(x)))

    # Display the result after this iteration
    x.show(title='Outer Bregman Iteration {}'.format(breg_iter),
           force_show=True)

# Create an FBP-reconstruction to compare with
fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.4)
fbp_reco = fbp_op(data)
fbp_reco.show(title='FBP Reconstruction')

# Finally, also display phantom and sinograms
discr_phantom.show(title='Phantom')
data.show(title='Simulated Data (Sinogram)')
