# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Basic example on how to write a functional.

When defining a new functional, there are a few standard methods and properties
that can be implemeted. These are:
- ``__init__``. This method intialize the functional
- ``_call``. The actual function call ``functional(x)``
- ``gradient`` (property). This gives the gradient operator of the functional,
    i.e., the operator that corresponds to the mapping ``x -> grad_f(x)``
- ``proximal``. This returns the proximal operator. If called only as
    ``functional.proximal``, it corresponds to a `Proximal factory`.
- ``conjugate_functional`` (property). This gives the convex conjugate
    functional
- ``derivative``. This returns the (directional) derivative operator in a point
    y, such that when called with a point x it corresponds to the linear
    operator ``x --> <x, grad_f(y)>``. Note that this has a default
    implemetation that uses the gradient in order to achieve this.

Below follows an example of implementing the functional ``||x||_2^2 + <x,y>``,
for some parameter ``y.``
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import odl

# Here we define the functional
class MyFunctional(odl.solvers.Functional):
    """This is my functional: ||x||_2^2 + <x, y>."""

    # Defining the __init__ function
    def __init__(self, domain, y):
        # This comand calls the init of Functional and sets a number of
        # parameters associated with a functional. All but domain have default
        # values if not set.
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=True, grad_lipschitz=2)

        # We need to check that y is in the domain. Then we store the value of
        # y for future use.
        if y not in domain:
            raise TypeError('y is not in the domain!')
        self._y = y

    # Now we define a propert which returns y, so that the user can see which
    # value is used in a particular instance of the class.
    @property
    def y(self):
        return self._y

    # Defining the _call function
    def _call(self, x):
        return x.norm()**2 + x.inner(self.y)

    # Next we define the gradient. Note that this is a property.
    @property
    def gradient(self):
        # Inside this property, we define the gradient operator. This can be
        # defined anywhere and just returned here, but in this example we will
        # also define it here.

        class MyGradientOperator(odl.Operator):

            # Define an __init__ method for this operator
            def __init__(self, functional):
                super().__init__(domain=functional.domain,
                                 range=functional.domain)

                self._functional = functional

            # Define a _call method for this operator
            def _call(self, x):
                return 2.0 * x + self._functional.y

        return MyGradientOperator(functional=self)

    # Next we define the convex conjugate functional
    @property
    def conjugate_functional(self):
        # This functional is implemented below
        return MyFunctionalConjugate(domain=self.domain, y=self.y)


# Here is the conjugate functional
class MyFunctionalConjugate(odl.solvers.Functional):
    """Calculations give that this funtional has the analytic expression
    f^*(x) = ||x||^2/2 - ||x-y||^2/4 + ||y||^2/2 - <x,y>.
    """
    def __init__(self, domain, y):
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=True, grad_lipschitz=2)

        if y not in domain:
            raise TypeError('y is not in the domain!')
        self._y = y

    @property
    def y(self):
        return self._y

    def _call(self, x):
        return (x.norm()**2 / 2.0 - (x - self.y).norm()**2 / 4.0 +
                self.y.norm()**2 / 2.0 - x.inner(self.y))


# Now we test the functional. First we create an instance of the functional
n = 10
space = odl.rn(n)
y = space.element(np.random.randn(n))
my_func = MyFunctional(domain=space, y=y)

# Now we evaluate it, and see that it returns the expected value
x = space.element(np.random.randn(n))

if my_func(x) == x.norm()**2 + x.inner(y):
    print('My functional evaluates corretly.')
else:
    print('There is a bug in the evaluation of my functional.')

# Next we create the gradient
my_gradient = my_func.gradient

# Frist we test that it is indeed an odl Operator
if isinstance(my_gradient, odl.Operator):
    print('The gradient is an operator, as it should be.')
else:
    print('There is an error in the gradient; it is not an operator.')

# Second, we test that it evaluates correctly
if my_gradient(x) == 2.0 * x + y:
    print('The gradient evaluates correctly.')
else:
    print('There is an error in the evaluation of the gradient.')

# Since we have not implemented the (directional) derivative, but we have
# implemeted the gradient, the default implementation will use this in order to
# evaluate the derivative. We test this behaviour.
p = space.element(np.random.randn(n))
my_deriv = my_func.derivative(x)

if my_deriv(p) == my_gradient(x).inner(p):
    print('The default implementation of the derivative works as intended.')
else:
    print('There is a bug in the implementation of the derivative')

# Since the proximal operator was not implemented it will raise a
# NotImplementedError
try:
    my_func.proximal()
except NotImplementedError:
    print('As expected we caught a NotImplementedError when trying to create '
          'the proximal operator')
else:
    print('There should have been an error, but it did not occure.')

# We now create the conjugate functional and test a call to it
my_func_conj = my_func.conjugate_functional

if my_func_conj(x) == (x.norm()**2 / 2.0 - (x - my_func.y).norm()**2 / 4.0 +
                       my_func.y.norm()**2 / 2.0 - x.inner(my_func.y)):
    print('The conjugate functional evaluates correctly.')
else:
    print('There is an error in the evaluation of the conjugate functional.')

# Nothing else has been implemented in the conjugate functional. For example,
# there is no gradient.
try:
    my_func_conj.gradient
except NotImplementedError:
    print('As expected we caught a NotImplementedError when trying to access '
          'the gradient operator.')
else:
    print('There should have been an error, but it did not occure.')

# There is no derivative either.
try:
    my_func_conj.derivative(x)(p)
except NotImplementedError:
    print('As expected we caught a NotImplementedError when trying to '
          'evaluate the derivative.')
else:
    print('There should have been an error, but it did not occure.')

# We now test some general properties that exists for all functionals. We can
# add two functioanls, scale it by multiplying with a scalar from the left,
# scale the argument by multiplying with a scalar from the right, and also
# translate the argument. Except for the sum of two functional, the other
# operations will apply corrections in order to evaluate gradients, etc.,
# in a correct way.

# Scaling the functional
func_scal = np.random.rand()
my_func_scaled = func_scal * my_func

if my_func_scaled(x) == func_scal * (my_func(x)):
    print('Scaling of functional works.')
else:
    print('There is an error in the scaling of functionals.')

my_func_scaled_grad = my_func_scaled.gradient
if my_func_scaled_grad(x) == func_scal * (my_func.gradient(x)):
    print('Scaling of functional evaluates gradient correctly.')
else:
    print('There is an error in evaluating the gradient in the scaling of '
          'functionals.')

# Scaling of the argument
arg_scal = np.random.rand()
my_func_arg_scaled = my_func * arg_scal

if my_func_arg_scaled(x) == my_func(arg_scal * x):
    print('Scaling of functional argument works.')
else:
    print('There is an error in the scaling of functional argument.')

# Sum of two functionals
y_2 = space.element(np.random.randn(n))
my_func_2 = MyFunctional(domain=space, y=y_2)
my_funcs_sum = my_func + my_func_2

if my_funcs_sum(x) == my_func(x) + my_func_2(x):
    print('Summing two functionals works.')
else:
    print('There is an error in the summation of functionals.')

# Translation of the functional, i.e., creating the functional f(. - y).
my_func_translated = my_func.translate(y_2)
if my_func_translated(x) == my_func(x - y_2):
    print('Translation of functional works.')
else:
    print('There is an error in the translation of functional.')
