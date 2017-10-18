"""Linearly convergent total variation denoising using PDHG.

This exhaustive example solve the L2-HuberTV problem

    .. math::
        \\min_{x >= 0}  1/2 ||x - d||_2^2
            + \\lambda \\sum_i \\eta_\\gamma(||grad(x)_i||_2)

where ``grad`` the spatial gradient and ``d`` is given noisy data. Here
``\\eta_\\gamma`` denotes the Huber function defined as

    .. math::
        \\eta_\\gamma(x) =
        \\begin{cases}
            \\frac{1}{2 \\gamma} x^2 + \\frac{gamma}{2}
                & \\text{if } |x| \leq \\gamma \\\\
            |x|
                & \\text{if } |x| > \\gamma,
        \\end{cases}.

We compare two different step size rules as described in

Chambolle, A., & Pock, T. (2011). *A First-Order Primal-Dual Algorithm for
Convex Problems with Applications to Imaging. Journal of Mathematical Imaging
and Vision, 40(1), 120–145. http://doi.org/10.1007/s10851-010-0251-1

Chambolle, A., Ehrhardt, M. J., Richtárik, P., & Schönlieb, C.-B. (2017).
Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling and
Imaging Applications. Retrieved from http://arxiv.org/abs/1706.04957

and show their convergence rates.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import scipy
import odl
import matplotlib.pyplot as plt

# --- define setting --- #

# Read test image: use only every second pixel, convert integer to float
image = np.rot90(scipy.misc.ascent()[::2, ::2].astype('float'), 3)
shape = image.shape

# Rescale max to 1
image /= image.max()

# Discretized space
space = odl.uniform_discr([0, 0], shape, shape)

# Create space element of ground truth
orig = space.element(image.copy())

# Create noisy observation
noisy = odl.phantom.white_noise(space, orig, 0.1)

# Gradient operator
gradient = odl.Gradient(space)

# The operator norm of the gradient with forward differences is well-known
gradient.norm = np.sqrt(8) + 1e-4

# regularization parameter
reg_param = 0.1

# l2 data matching
l2_norm = (1 / (2 * reg_param) *
           odl.solvers.L2NormSquared(space).translated(noisy))

# HuberTV-regularization
huber = odl.solvers.Huber(gradient.range, gamma=.01)

# define objective
obj_fun = l2_norm + huber * gradient


# define callback to store function values
class CallbackStore(odl.solvers.Callback):
    def __init__(self):
        self.iteration_count = 0
        self.iteration_counts = []
        self.obj_function_values = []

    def __call__(self, x):
        self.iteration_count += 1
        self.iteration_counts.append(self.iteration_count)
        self.obj_function_values.append(obj_fun(x))

    def reset(self):
        self.iteration_count = 0
        self.iteration_counts = []
        self.obj_function_values = []


callback = (odl.solvers.CallbackPrintIteration() & CallbackStore())

# number of iterations
niter = 200

# Assign operator and functionals
op = gradient
f = huber
g = l2_norm

# strong convexity of "f*" and "g"
mu_g = 1 / reg_param
mu_f = 1 / huber.grad_lipschitz

# parameters for algorithm 1
# slightly smaller than condition number of the problem
kappa1 = 0.999 * gradient.norm**2 / (mu_g * mu_f)

tau1 = 1 / (mu_g * (np.sqrt(1 + kappa1) - 1))  # Primal step size
sigma1 = 1 / (mu_f * (np.sqrt(1 + kappa1) - 1))  # Dual step size
theta1 = 1 - 2 / (1 + np.sqrt(1 + kappa1))  # Extrapolation constant

# parameters for algorithm 2
# square root of usual condition number
kappa2 = gradient.norm / np.sqrt(mu_f * mu_g)

tau2 = 1 / gradient.norm * np.sqrt(mu_f / mu_g)  # Primal step size
sigma2 = 1 / gradient.norm * np.sqrt(mu_g / mu_f)  # Dual step size
theta2 = 1 - 1 / (1 + 0.5 * kappa2)  # Extrapolation constant

# Run linearly convergent algorithm 1
x1 = space.zero()
callback(x1)
odl.solvers.pdhg(x1, f, g, op, tau1, sigma1, niter, theta=theta1,
                 callback=callback)
obj1 = callback.callbacks[1].obj_function_values

# Run linearly convergent algorithm 2
callback.reset()
x2 = space.zero()
callback(x2)
odl.solvers.pdhg(x2, f, g, op, tau2, sigma2, niter, theta=theta2,
                 callback=callback)
obj2 = callback.callbacks[1].obj_function_values

# %% Display results
# show images
clim = [0, 1]
cmap = 'gray'

orig.show('original', clim=clim, cmap=cmap)
noisy.show('noisy', clim=clim, cmap=cmap)
x1.show('denoised, alg1', clim=clim, cmap=cmap)
x2.show('denoised, alg2', clim=clim, cmap=cmap)

# show convergence rate
min_obj = min(obj1 + obj2)


def rel_fun(x):
    x = np.array(x)
    return (x - min_obj) / (x[0] - min_obj)


i = np.array(callback.callbacks[1].iteration_counts)

plt.figure(1)
plt.clf()
plt.semilogy(i, rel_fun(obj1), color='red',
             label='alg1, Chambolle et al 2017')
plt.semilogy(i, rel_fun(obj2), color='blue',
             label='alg2, Chambolle and Pock 2011')
rho = theta1
plt.semilogy(i[1:], rho**i[1:], '--', color='red',
             label='$O(\\rho_1^k), \\rho_1={:3.2f}$'.format(rho))
rho = theta2
plt.semilogy(i[1:], rho**i[1:], '--', color='blue',
             label='$O(\\rho_2^k), \\rho_2={:3.2f}$'.format(rho))
plt.title('Function values + theoretical upper bounds')
plt.ylim((1e-16, 1))
plt.legend()
