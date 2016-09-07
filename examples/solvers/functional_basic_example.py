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

This is an example of how to implemente the functional ``||x||_2^2``. For more
information on functionals, see
http://odl.readthedocs.io/guide/in_depth/functional_guide.html
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

    """This is my functional: ||x||_2^2."""

    def __init__(self, domain):
        # This comand calls the init of Functional and sets a number of
        # parameters associated with a functional. All but domain have default
        # values if not set.
        super().__init__(domain=domain, linear=False, grad_lipschitz=2)

    def _call(self, x):
        # This is what is returned when calling my_func(x)
        return x.norm()**2

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
my_func = MyFunctional(domain=space)

# The functional evaluates correctly
x = space.element(np.random.randn(n))
print(my_func(x) == x.norm()**2)

# The gradient works
my_gradient = my_func.gradient
print(my_gradient(x) == 2.0 * x)

# The standard implementation of the directional derivative works
p = space.element(np.random.randn(n))
my_deriv = my_func.derivative(x)
print(my_deriv(p) == my_gradient(x).inner(p))

# The conjugate functional works
my_func_conj = my_func.convex_conj
print(my_func_conj(x) == 1.0 / 4.0 * x.norm()**2)

# As a final, a bit more advanced, test, this check that the a scaled and
# translated version of the functional evalutes the gradient correctly
scal = np.random.rand()
transl = space.element(np.random.randn(n))
scal_and_transl_func_gradient = (scal * my_func.translate(transl)).gradient
print(scal_and_transl_func_gradient(x) == scal * my_func.gradient(x - transl))
