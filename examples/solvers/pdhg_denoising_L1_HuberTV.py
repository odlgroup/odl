"""Total variation denoising using PDHG.

This exhaustive example solves the L1-HuberTV problem

        min_{x >= 0} ||x - d||_1
            + lam * sum_i eta_gamma(||grad(x)_i||_2)

where grad the spatial gradient and d is given noisy data. Here eta_gamma
denotes the Huber function. For more details, see the Huber documentation.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import scipy
import odl
import matplotlib.pyplot as plt

# --- define setting --- #

# Define ground truth, space and noisy data
image = np.rot90(scipy.misc.ascent()[::2, ::2].astype('float'), 3)
shape = image.shape
image /= image.max()
space = odl.uniform_discr([0, 0], shape, shape)
orig = space.element(image.copy())
d = odl.phantom.salt_pepper_noise(orig)

# Define objective functional
op = odl.Gradient(space)  # operator
op.norm = np.sqrt(8) + 1e-4  # norm with forward differences is well-known
lam = 1  # Regularization parameter
g = 1 / lam * odl.solvers.L1Norm(space).translated(d)  # data fit
f = odl.solvers.Huber(op.range, gamma=.1)  # regularization
obj_fun = f * op + g  # combined functional
mu_f = 1 / f.grad_lipschitz  # Strong convexity of "f*"

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


callback = (odl.solvers.CallbackPrintIteration(step=10) & CallbackStore())
niter = 500  # Number of iterations
tau = 1.0 / op.norm  # Step size for primal variable
sigma = 1.0 / op.norm  # Step size for dual variable

# Run algorithm
x = space.zero()
callback(x)  # store values for initialization
odl.solvers.pdhg(x, f, g, op, tau, sigma, niter, gamma_dual=mu_f,
                 callback=callback)
obj = callback.callbacks[1].obj_function_values

# %% Display results
# Show images
clim = [0, 1]
cmap = 'gray'

orig.show('original', clim=clim, cmap=cmap)
d.show('noisy', clim=clim, cmap=cmap)
x.show('denoised', clim=clim, cmap=cmap)


# Show convergence rate
def rel_fun(x):
    x = np.array(x)
    return (x - min(x)) / (x[0] - min(x))


i = np.array(callback.callbacks[1].iteration_counts)

plt.figure()
plt.loglog(i, rel_fun(obj), label='pdhg')
plt.loglog(i[1:], 1. / i[1:], '--', label='$O(1/k)$')
plt.loglog(i[1:], 4. / i[1:] ** 2, ':', label='$O(1/k^2)$')
rho = 0.97
plt.loglog(i[1:], rho ** i[1:], '-',
           label='$O(\\rho^k), \\rho={:3.2f}$'.format(rho))
plt.title('Function values')
plt.legend()
