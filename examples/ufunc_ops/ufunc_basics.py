"""Basic examples of using the ufunc functionals in ODL."""

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


# Can also define ufuncs on vector-spaces, then they act pointwise.

r3 = odl.rn(3)
exp_r3 = odl.ufunc_ops.exp(r3)
print('e^[1, 2, 3] = {}'.format(exp_r3([1, 2, 3])))
