"""Example of a deconvolution problem with different solvers (CPU)."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal
import odl


class Convolution(odl.Operator):
    def __init__(self, kernel, domain, range, adjkernel=None):
        self.kernel = kernel
        self.adjkernel = torch.flip(kernel, dims=(0,)) if adjkernel is None else adjkernel
        self.norm = float(torch.sum(torch.abs(self.kernel)))
        super(Convolution, self).__init__(
            domain=domain, range=range, linear=True)

    def _call(self, x):
        return self.range.element(
                  torch.conv1d( input=x.data.unsqueeze(0)
                              , weight=self.kernel.unsqueeze(0).unsqueeze(0)
                              , stride=1
                              , padding="same"
                              ).squeeze(0)
                           )

    @property
    def adjoint(self):
        return Convolution( self.adjkernel
                          , domain=self.range, range=self.domain
                          , adjkernel = self.kernel
                          )

    def opnorm(self):
        return self.norm


resolution = 50

# Discretization
discr_space = odl.uniform_discr(-5, 5, resolution*10, impl='pytorch', dtype=np.float32)

# Complicated functions to check performance
def mk_kernel():
    q = 1.172
    # Select main lobe and one side lobe on each side
    r = np.ceil(3*np.pi/(2*q))
    # Quantised to resolution
    nr = int(np.ceil(r*resolution))
    r = nr / resolution
    x = torch.linspace(-r, r, nr*2 + 1)
    return torch.exp(-x**2 * 2) * np.cos(x * q)
kernel = mk_kernel()

phantom = discr_space.element(lambda x: np.ones_like(x) ** 2 * (x > -1) * (x < 1))
# phantom = discr_space.element(lambda x: x ** 2 * np.sin(x) ** 2 * (x > 5))

# Create operator
conv = Convolution(kernel, domain=discr_space, range=discr_space)

# Dampening parameter for landweber
iterations = 100
omega = 1 / conv.opnorm() ** 2



def test_with_plot(conv, phantom, solver, **extra_args):
    fig, axs = plt.subplots(2)
    fig.suptitle("CGN")
    def plot_fn(ax_id, fn, *plot_args, **plot_kwargs):
        axs[ax_id].plot(fn, *plot_args, **plot_kwargs)
    axs[0].set_title("x")
    axs[1].set_title("k*x")
    plot_fn(0, phantom)
    plot_fn(1, conv(phantom))
    def plot_callback(x):
        plot_fn(0, conv(x), '--')
        plot_fn(1, conv(x), '--')
    solver(conv, discr_space.zero(), phantom, iterations, callback=plot_callback, **extra_args)

# Test CGN
test_with_plot(conv, phantom, odl.solvers.conjugate_gradient_normal)

# # Landweber
# lw_fig, lw_axs = plt.subplots(1)
# lw_fig.suptitle("Landweber")
# lw_axs.plot(phantom)
# odl.solvers.landweber(conv, discr_space.zero(), phantom,
#                       iterations, omega, lambda x: lw_axs.plot(conv(x)))

plt.show()
