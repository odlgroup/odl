"""Example of a deconvolution problem with different solvers (CPU)."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import odl


class Convolution(odl.Operator):
    def __init__(self, kernel, adjkernel=None):
        self.kernel = kernel
        self.adjkernel = (adjkernel if adjkernel is not None
                          else kernel.space.element(kernel[::-1].copy()))
        self.norm = float(np.sum(np.abs(self.kernel)))
        super(Convolution, self).__init__(
            domain=kernel.space, range=kernel.space, linear=True)

    def _call(self, x):
        return scipy.signal.convolve(x, self.kernel, mode='same')

    @property
    def adjoint(self):
        return Convolution(self.adjkernel, self.kernel)

    def opnorm(self):
        return self.norm


# Discretization
discr_space = odl.uniform_discr(-5, 5, 500, impl='numpy')

# Complicated functions to check performance
kernel = discr_space.element(lambda x: np.exp(-x**2 * 2) * np.cos(x * 1.172))

# phantom = discr_space.element(lambda x: (x+5) ** 2 * np.sin(x+5) ** 2 * (x > 0))
phantom = discr_space.element(lambda x: np.cos(0*x) * (x > -1) * (x < 1))

# Create operator
conv = Convolution(kernel)

# Dampening parameter for landweber
iterations = 100
omega = 1 / conv.opnorm() ** 2


def test_with_plot(conv, phantom, solver, **extra_args):
    fig, axs = plt.subplots(2)
    fig.suptitle("CGN")
    axs[0].set_title("x")
    axs[1].set_title("k*x")
    axs[0].plot(phantom)
    axs[1].plot(conv(phantom))
    def plot_callback(x):
        axs[0].plot(conv(x), '--')
        axs[1].plot(conv(x), '--')
    solver(conv, discr_space.zero(), phantom, iterations, callback=plot_callback, **extra_args)

# Test CGN
test_with_plot(conv, phantom, odl.solvers.conjugate_gradient_normal)

# test_with_plot(conv, phantom, odl.solvers.landweber, omega=omega)

# # Landweber
# lw_fig, lw_axs = plt.subplots(1)
# lw_fig.suptitle("Landweber")
# lw_axs.plot(phantom)
# odl.solvers.landweber(conv, discr_space.zero(), phantom,
#                       iterations, omega, lambda x: lw_axs.plot(conv(x)))
# 
plt.show()
