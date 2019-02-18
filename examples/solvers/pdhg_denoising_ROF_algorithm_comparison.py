"""Total variation denoising using PDHG.

Three different algorithms (or variants of PDHG) are compared to solve the
ROF (Rudin-Osher-Fatemi) problem / L2-TV

  (ROF)    min_{x >= 0}  1/2 ||x - d||_2^2 + lam || |grad(x)| ||_1

Where ``grad`` the spatial gradient and ``d`` is given noisy data.

Algorithms 1 and 2 are two different assignments of the functional parts of ROF
to the functions f and g of PDHG. Algorithm 3 improves upon algorithm 2 by
making use of the strong convexity of the problem.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import scipy.misc
import odl
import matplotlib.pyplot as plt

# --- define setting --- #

# Read test image: use only every second pixel, convert integer to float
image = scipy.misc.ascent()[::2, ::2].astype('float')
shape = image.shape

# Rescale max to 1
image /= image.max()

# Discretized spaces
space = odl.uniform_discr([0, 0], shape, shape)

# Create space element of ground truth
orig = space.element(image.copy())

# Add noise and convert to space element
noisy = orig + 0.1 * odl.phantom.white_noise(space)

# Gradient operator
gradient = odl.Gradient(space, method='forward')

# regularization parameter
reg_param = 0.3

# l2-squared data matching
factr = 0.5 / reg_param
l2_norm = factr * odl.solvers.L2NormSquared(space).translated(noisy)

# Isotropic TV-regularization: l1-norm of grad(x)
l1_norm = odl.solvers.GroupL1Norm(gradient.range, 2)

# characteristic function
char_fun = odl.solvers.IndicatorNonnegativity(space)

# define objective
obj = l2_norm + l1_norm * gradient + char_fun

# strong convexity of "g"
strong_convexity = 1 / reg_param


# define callback to store function values
class CallbackStore(odl.solvers.util.callback.Callback):
    def __init__(self):
        self.iteration_count = 0
        self.iteration_counts = []
        self.ergodic_iterate = 0
        self.obj_function_values = []
        self.obj_function_values_ergodic = []

    def __call__(self, x):
        self.iteration_count += 1
        k = self.iteration_count

        self.iteration_counts.append(self.iteration_count)
        self.ergodic_iterate = (k - 1) / k * self.ergodic_iterate + 1 / k * x
        self.obj_function_values.append(obj(x))
        self.obj_function_values_ergodic.append(obj(self.ergodic_iterate))

    def reset(self):
        self.iteration_count = 0
        self.iteration_counts = []
        self.ergodic_iterate = 0
        self.obj_function_values = []
        self.obj_function_values_ergodic = []


callback = (odl.solvers.CallbackPrintIteration() & CallbackStore())

# number of iterations
niter = 500

# %% Run Algorithms

# --- Algorithm 1 --- #

# Operator assignment
op = odl.BroadcastOperator(odl.IdentityOperator(space), gradient)

# Make separable sum of functionals, order must correspond to the operator K
g = odl.solvers.SeparableSum(l2_norm, l1_norm)

# Non-negativity constraint
f = char_fun

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op, xstart=noisy)

tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Starting point
x_start = op.domain.zero()

# Run algorithm 1
x_alg1 = x_start.copy()
callback.reset()
odl.solvers.pdhg(x_alg1, f, g, op, niter=niter, tau=tau, sigma=sigma,
                 callback=callback)
obj_alg1 = callback.callbacks[1].obj_function_values
obj_ergodic_alg1 = callback.callbacks[1].obj_function_values_ergodic

# --- algorithm 2 and 3 --- #

# Operator assignment
op = gradient

# Assign functional f
g = l1_norm

# Create new functional that combines data fit and characteritic function
f = odl.solvers.FunctionalQuadraticPerturb(char_fun, factr, -2 * factr * noisy)

# The operator norm of the gradient with forward differences is well-known
op_norm = np.sqrt(8) + 1e-4

tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Run algorithms 2 and 3
x_alg2 = x_start.copy()
callback.reset()
odl.solvers.pdhg(x_alg2, f, g, op, niter=niter, tau=tau, sigma=sigma,
                 gamma_primal=0, callback=callback)
obj_alg2 = callback.callbacks[1].obj_function_values
obj_ergodic_alg2 = callback.callbacks[1].obj_function_values_ergodic

x_alg3 = x_start.copy()
callback.reset()
odl.solvers.pdhg(x_alg3, f, g, op, niter=niter, tau=tau, sigma=sigma,
                 gamma_primal=strong_convexity, callback=callback)
obj_alg3 = callback.callbacks[1].obj_function_values
obj_ergodic_alg3 = callback.callbacks[1].obj_function_values_ergodic

# %% Display results
# show images
plt.figure(0)
ax1 = plt.subplot(231)
ax1.imshow(orig, clim=[0, 1], cmap='gray')
ax1.title.set_text('Original Image')

ax2 = plt.subplot(232)
ax2.imshow(noisy, clim=[0, 1], cmap='gray')
ax2.title.set_text('Noisy Image')

ax3 = plt.subplot(234)
ax3.imshow(x_alg1, clim=[0, 1], cmap='gray')
ax3.title.set_text('Algo 1')

ax4 = plt.subplot(235)
ax4.imshow(x_alg2, clim=[0, 1], cmap='gray')
ax4.title.set_text('Algo 2')

ax5 = plt.subplot(236)
ax5.imshow(x_alg3, clim=[0, 1], cmap='gray')
ax5.title.set_text('Algo 3')

# show function values
i = np.array(callback.callbacks[1].iteration_counts)

plt.figure(1)
plt.clf()
plt.loglog(i, obj_alg1, label='Algo 1')
plt.loglog(i, obj_alg2, label='Algo 2')
plt.loglog(i, obj_alg3, label='Algo 3')
plt.title('Function Values')
plt.legend()

# show convergence rates
plt.figure(2)
plt.clf()
obj_opt = min(obj_alg1 + obj_alg2 + obj_alg3)


def rel_fun(x):
    return (np.array(x) - obj_opt) / (x[0] - obj_opt)


plt.loglog(i, rel_fun(obj_alg1), label='Algo 1')
plt.loglog(i, rel_fun(obj_alg2), label='Algo 2')
plt.loglog(i, rel_fun(obj_alg3), label='Algo 3')
plt.loglog(i[1:], 1. / i[1:], '--', label=r'$1/k$')
plt.loglog(i[1:], 1. / i[1:]**2, ':', label=r'$1/k^2$')
plt.title('Relative Function Values')
plt.legend()

# show ergodic convergence rates
plt.figure(3)
plt.clf()

plt.loglog(i, rel_fun(obj_ergodic_alg1), label='Algo 1')
plt.loglog(i, rel_fun(obj_ergodic_alg2), label='Algo 2')
plt.loglog(i[1:], 4. / i[1:], '--', label=r'$O(1/k)$')
plt.title('Relative Ergodic Function Values')
plt.legend()
