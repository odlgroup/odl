"""Total variation denoising using PDHG.

This example solves the L1-HuberTV problem

        min_{x >= 0} ||x - d||_1
            + lam * sum_i eta_gamma(||grad(x)_i||_2)

where ``grad`` is the spatial gradient and ``d`` is given noisy data. Here
``eta_gamma`` denotes the Huber function. For more details, see the Huber
documentation.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import matplotlib.pyplot as plt
import numpy as np

import odl

# Define ground truth, space and noisy data
shape = [100, 100]
space = odl.uniform_discr([0, 0], shape, shape)
orig = odl.phantom.smooth_cuboid(space)
d = odl.phantom.salt_pepper_noise(orig, fraction=0.2)

# Define objective functional
op = odl.Gradient(space)  # operator
norm_op = np.sqrt(8) + 1e-4  # norm with forward differences is well-known
lam = 2  # Regularization parameter
const = 0.5
g = const / lam * odl.solvers.L1Norm(space).translated(d)  # data fit
f = const * odl.solvers.Huber(op.range, gamma=.01)  # regularization
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


callback = odl.solvers.CallbackPrintIteration(step=10) & CallbackStore()
niter = 500  # Number of iterations
tau = 1.0 / norm_op  # Step size for primal variable
sigma = 1.0 / norm_op  # Step size for dual variable

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
plt.loglog(i[1:], 20. / i[1:] ** 2, ':', label='$O(1/k^2)$')
plt.title('Function values')
plt.legend()
