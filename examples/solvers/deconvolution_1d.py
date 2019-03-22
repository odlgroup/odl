"""Example of a deconvolution problem with different solvers (CPU)."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import odl


class Convolution(odl.Operator):
    def __init__(self, space, kernel, adjkernel=None):
        super(Convolution, self).__init__(
            domain=space, range=space, linear=True
        )
        self.kernel = np.asarray(kernel)
        if adjkernel is None:
            self.adjkernel = kernel[::-1]
        else:
            self.adjkernel = np.asarray(adjkernel)
        self._norm = float(np.sum(np.abs(self.kernel)))

    def _call(self, x):
        return scipy.signal.convolve(x, self.kernel, mode='same')

    @property
    def adjoint(self):
        return Convolution(self.adjkernel, self.kernel)

    def norm(self):
        return self._norm


# Discretization
discr_space = odl.uniform_discr(0, 10, 500, impl='numpy')

# Complicated functions to check performance
kernel = discr_space.element(lambda x: np.exp(x / 2) * np.cos(x * 1.172))
phantom = discr_space.element(lambda x: x ** 2 * np.sin(x) ** 2 * (x > 5))

# Create operator
conv = Convolution(kernel)

# Dampening parameter for landweber
iterations = 100
omega = 1 / conv.norm() ** 2


# Display callback
def callback(x):
    plt.plot(conv(x))


# Test CGN
plt.figure()
plt.plot(phantom)
odl.solvers.conjugate_gradient_normal(conv, discr_space.zero(), phantom,
                                      iterations, callback)

# Landweber
plt.figure()
plt.plot(phantom)
odl.solvers.landweber(conv, discr_space.zero(), phantom,
                      iterations, omega, callback)

plt.show()
