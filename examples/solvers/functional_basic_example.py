"""Basic example on how to write a functional.

This is an example of how to implement the functional ``||x||_2^2``. For more
information on functionals, see `the ODL functional guide
<http://odlgroup.github.io/odl/guide/in_depth/functional_guide.html>`_
"""

import numpy as np
import odl


# Here we define the functional
class MyFunctional(odl.solvers.Functional):

    """This is my functional: ||x||_2^2."""

    def __init__(self, space):
        # This comand calls the init of Functional and sets a number of
        # parameters associated with a functional. All but domain have default
        # values if not set.
        super(MyFunctional, self).__init__(
            space=space, linear=False, grad_lipschitz=2)

    def _call(self, x):
        # This is what is returned when calling my_func(x)
        return self.domain.norm(x) ** 2

    @property
    def gradient(self):
        # The gradient is given by 2 * x
        return 2.0 * odl.IdentityOperator(self.domain)

    @property
    def convex_conj(self):
        # Calculations give that this funtional has the analytic expression
        # f^*(x) = 1/4 * ||x||_2^2.
        return 1.0 / 4.0 * MyFunctional(self.domain)


# Create an instance of the functional and test some basic parts of it.
n = 10
space = odl.rn(n)
my_func = MyFunctional(space=space)

# Functional evaluation
x = space.element(np.random.randn(n))
print(
    'f(x) == ||x||^2                             ?',
    my_func(x) == space.norm(x) ** 2,
)

# Gradient
my_grad = my_func.gradient
print(
    'grad f(x) == 2 * x                          ?',
    all(my_grad(x) == 2.0 * x),
)

# Derivative (implemented via gradient)
p = space.element(np.random.randn(n))
my_deriv = my_func.derivative(x)
print(
    'Df(p)(x) == <grad f(x), p>                  ?',
    my_deriv(p) == space.inner(my_grad(x), p),
)

# Convex conjugate
my_func_conj = my_func.convex_conj
print(
    'f*(x) == ||x||^2 / 4                        ?',
    my_func_conj(x) == space.norm(x) ** 2 / 4,
)

# Scaling and translating a functional, checking the gradient
scal = np.random.rand()
transl = space.element(np.random.randn(n))
scal_transl_grad = (scal * my_func.translated(transl)).gradient
print(
    'grad [s * f(. - t)](x) == s * grad f(x - t) ?',
    all(scal_transl_grad(x) == scal * my_func.gradient(x - transl)),
)
