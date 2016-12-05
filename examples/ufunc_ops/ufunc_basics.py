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

"""Basic examples of using the UFunc functionals in ODL."""

from __future__ import print_function
import odl


# Trigonometric functions can be computed, along with their gradients.


cos = odl.ufunc_ops.cos()
sin = odl.ufunc_ops.sin()

# Compute cosine and its gradient

print('cos(0)={}, cos.gradient(0.2)={}, -sin(0.2)={}'.format(
    cos(0), cos.gradient(0.2), -sin(0.2)))


# Other functions include the square, exponential, etc
# Higher order derivatives are obtained via the gradient of the gradient, etc.

square = odl.ufunc_ops.square()

print('[x^2](3) = {}, [d/dx x^2](3) = {}, '
      '[d^2/dx^2 x^2](3) = {}, [d^3/dx^3 x^2](3) = {}'.format(
    square(3), square.gradient(3),
    square.gradient.gradient(3), square.gradient.gradient.gradient(3)))


# Can also define UFuncs on vector-spaces, then they act pointwise.

r3 = odl.rn(3)
exp_r3 = odl.ufunc_ops.exp(r3)
print('e^[1, 2, 3] = {}'.format(exp_r3([1, 2, 3])))
