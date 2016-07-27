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
    """This is my functional: ||x||_2^2 + <x, linear_term>."""

    def __init__(self, domain, linear_term):
        # This comand calls the init of Functional and sets a number of
        # parameters associated with a functional. All but domain have default
        # values if not set.
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=True, grad_lipschitz=2)

        # We need to check that y is in the domain. Then we store the value of
        # y for future use.
        if linear_term not in domain:
            raise TypeError('y is not in the domain!')
        self._linear_term = linear_term

    # Property that returns the linear term.
    @property
    def linear_term(self):
        return self._linear_term

    # Defining the _call function
    def _call(self, x):
        return x.norm()**2 + x.inner(self.linear_term)

    # Defining the gradient. Note that this is a property.
    @property
    def gradient(self):

        # The class corresponding to the gradient operator.
        class MyGradientOperator(odl.Operator):
            """Class that implements the gradient operator of the functional
            ``||x||_2^2 + <x,y>``.
            """

            def __init__(self, functional):
                super().__init__(domain=functional.domain,
                                 range=functional.domain)

                self._functional = functional

            def _call(self, x):
                return 2.0 * x + self._functional.linear_term

        return MyGradientOperator(functional=self)

    # Next we define the convex conjugate functional.
    @property
    def conjugate_functional(self):
        # This functional is implemented below.
        return MyFunctionalConjugate(domain=self.domain,
                                     linear_term=self.linear_term)


# Here is the conjugate functional.
class MyFunctionalConjugate(odl.solvers.Functional):
    """Conjugate functional to ``||x||_2^2 + <x,linear_term>``.

    Calculations give that this funtional has the analytic expression
    f^*(x) = ||x||^2/2 - ||x-linear_term||^2/4 + ||linear_term||^2/2 -
    <x,linear_term>.
    """
    def __init__(self, domain, linear_term):
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=True, grad_lipschitz=2)

        if linear_term not in domain:
            raise TypeError('y is not in the domain!')
        self._linear_term = linear_term

    @property
    def linear_term(self):
        return self._linear_term

    def _call(self, x):
        return (x.norm()**2 / 2.0 - (x - self.linear_term).norm()**2 / 4.0 +
                self.linear_term.norm()**2 / 2.0 - x.inner(self.linear_term))


# Now we create an instance of the functional and test some basic parts of it.
n = 10
space = odl.rn(n)
linear_term = space.element(np.random.randn(n))
my_func = MyFunctional(domain=space, linear_term=linear_term)

# The functional evaluates correctly
x = space.element(np.random.randn(n))
print(my_func(x) == x.norm()**2 + x.inner(linear_term))

# The gradient works
my_gradient = my_func.gradient
print(my_gradient(x) == 2.0 * x + linear_term)

# The standard implementation of the directional derivative works
p = space.element(np.random.randn(n))
my_deriv = my_func.derivative(x)
print(my_deriv(p) == my_gradient(x).inner(p))

# The conjugate functional works
my_func_conj = my_func.conjugate_functional
print(my_func_conj(x) == (x.norm()**2 / 2.0 -
                          (x - my_func.linear_term).norm()**2 / 4.0 +
                          my_func.linear_term.norm()**2 / 2.0 -
                          x.inner(my_func.linear_term)))

# As a final test, we check that the a scaled and translated version of the
# functional evalutes the gradient correctly
scal = np.random.rand()
transl = space.element(np.random.randn(n))
scal_and_transl_func_gradient = (scal * my_func.translate(transl)).gradient
print(scal_and_transl_func_gradient(x) == scal * my_func.gradient(x - transl))
