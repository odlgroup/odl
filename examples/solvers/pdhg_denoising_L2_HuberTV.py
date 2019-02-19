"""Linearly convergent total variation denoising using PDHG.

This exhaustive example solves the L2-HuberTV problem

        min_{x >= 0}  1/2 ||x - d||_2^2
            + lam * sum_i eta_gamma(||grad(x)_i||_2)

where ``grad`` is the spatial gradient and ``d`` is given noisy data. Here
``eta_gamma`` denotes the Huber function. For more details, see the Huber
documentation.

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
import scipy.misc
import odl
import matplotlib.pyplot as plt

# Define ground truth, space and noisy data
image = np.rot90(scipy.misc.ascent()[::2, ::2].astype('float'), 3)
shape = image.shape
image /= image.max()
space = odl.uniform_discr([0, 0], shape, shape)
orig = space.element(image.copy())
d = odl.phantom.white_noise(space, orig, 0.1)

# Define objective functional
op = odl.Gradient(space)  # operator
norm_op = np.sqrt(8) + 1e-4  # norm with forward differences is well-known
lam = 0.1  # Regularization parameter
f = 1 / (2 * lam) * odl.solvers.L2NormSquared(space).translated(d)  # data fit
g = odl.solvers.Huber(op.range, gamma=.01)  # regularization
obj_fun = f + g * op  # combined functional
mu_g = 1 / lam  # strong convexity of "g"
mu_f = 1 / f.grad_lipschitz  # strong convexity of "f*"

# Define algorithm parameters


class CallbackStore(odl.solvers.Callback):  # Callback to store function values
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


callback = odl.solvers.CallbackPrintIteration(step=10) & CallbackStore()
niter = 200  # number of iterations


# Parameters for algorithm 1
# Related to the root of the problem condition number
kappa1 = np.sqrt(1 + 0.999 * norm_op ** 2 / (mu_g * mu_f))
tau1 = 1 / (mu_g * (kappa1 - 1))  # Primal step size
sigma1 = 1 / (mu_f * (kappa1 - 1))  # Dual step size
theta1 = 1 - 2 / (1 + kappa1)  # Extrapolation constant

# Parameters for algorithm 2
# Square root of the problem condition number
kappa2 = norm_op / np.sqrt(mu_f * mu_g)
tau2 = 1 / norm_op * np.sqrt(mu_f / mu_g)  # Primal step size
sigma2 = 1 / norm_op * np.sqrt(mu_g / mu_f)  # Dual step size
theta2 = 1 - 2 / (2 + kappa2)  # Extrapolation constant

# Run linearly convergent algorithm 1
x1 = space.zero()
callback(x1)  # store values for initialization
odl.solvers.pdhg(x1, f, g, op, niter, tau1, sigma1, theta=theta1,
                 callback=callback)
obj1 = callback.callbacks[1].obj_function_values

# Run linearly convergent algorithm 2
callback.reset()
x2 = space.zero()
callback(x2)  # store values for initialization
odl.solvers.pdhg(x2, f, g, op, niter, tau2, sigma2, theta=theta2,
                 callback=callback)
obj2 = callback.callbacks[1].obj_function_values

# %% Display results
# Show images
clim = [0, 1]
cmap = 'gray'

orig.show('Original', clim=clim, cmap=cmap)
d.show('Noisy', clim=clim, cmap=cmap)
x1.show('Denoised, Algo 1', clim=clim, cmap=cmap)
x2.show('Denoised, Algo 2', clim=clim, cmap=cmap)

# Show convergence rate
min_obj = min(obj1 + obj2)


def rel_fun(x):
    x = np.array(x)
    return (x - min_obj) / (x[0] - min_obj)


iters = np.array(callback.callbacks[1].iteration_counts)

plt.figure()
plt.semilogy(iters, rel_fun(obj1), color='red',
             label='Algo 1, Chambolle et al 2017')
plt.semilogy(iters, rel_fun(obj2), color='blue',
             label='Algo 2, Chambolle and Pock 2011')
rho = theta1
plt.semilogy(iters[1:], rho ** iters[1:], '--', color='red',
             label=r'$O(\rho_1^k), \rho_1={:3.2f}$'.format(rho))
rho = theta2
plt.semilogy(iters[1:], rho ** iters[1:], '--', color='blue',
             label=r'$O(\rho_2^k), \rho_2={:3.2f}$'.format(rho))
plt.title('Function Values + Theoretical Upper Bounds')
plt.ylim((1e-16, 1))
plt.legend()
