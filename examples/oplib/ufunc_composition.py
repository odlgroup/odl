"""Examples of composing Ufuncs.

Here we demonstrate how the functionals can be composed with other operators
and functionals in order to achieve more complicated functions.

We create the L2-norm squared in two ways, first using the built in
L2NormSquared functional, and also by composing the square ufunc with the
L2Norm functional.

We also demonstrate that we can do this pointwise.
"""

import odl

# Create square functional. It's domain is by default the real numbers.
square = odl.oplib.square()

# Create L2 norm functionals
space = odl.rn(3)
l2_norm = odl.solvers.L2Norm(space)
l2_norm_squared_comp = square * odl.solvers.L2Norm(space)
l2_norm_squared_raw = odl.solvers.L2NormSquared(space)

# Evaluate in a point and see that the results are equal
x = space.element([1, 2, 3])

print('composed      ||x||^2 = {}'.format(l2_norm_squared_comp(x)))
print('raw           ||x||^2 = {}'.format(l2_norm_squared_raw(x)))

# The usual properties like gradients follow as expected
print('composed grad ||x||^2 = {}'.format(l2_norm_squared_comp.gradient(x)))
print('raw      grad ||x||^2 = {}'.format(l2_norm_squared_raw.gradient(x)))


# The above can also be done using pointwise UFuncs
square = odl.ufunc_ops.square(space)

l2_norm_composed_pointwise = l2_norm * square
print('composed      ||x^2|| = {}'.format(l2_norm_composed_pointwise(x)))
print('raw           ||x^2|| = {}'.format(l2_norm(x ** 2)))
