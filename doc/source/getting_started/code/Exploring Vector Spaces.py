
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


# # Exploring vector spaces

# The first tutorial showed some very basic functionality about vectors
# and spaces containing them. This tutorial explores the topic a bit
# further and introduces some more interesting and useful functionality.
# As usual, we start by importing the `odl` module, and this time we also
# add `numpy` for later.

from __future__ import print_function  # For Python2 compatibility
import numpy as np
import odl


# As seen before, one way to create a space of 3-element vectors with a
# certain data type is to use the `odl.fn` constructor:

odl.fn(3, 'int')


odl.fn(3, 'float')


odl.fn(3, 'complex')


# The spaces with real or complex floating point data types have own
# constructors `rn` and `cn`, mostly because they are most heavily used.
# Mathematically, they stand for $n$-dimensional Euclidean spaces
# $\mathbb{R}^n$ or $\mathbb{C}^n$, respectively, where in our case, $n =
# 3$. In general, we usually write $\mathbb{F}$ for "$\mathbb{R}$ or
# $\mathbb{C}$" and $\mathbb{F}^n$ for the $n$-dimensional Euclidean
# vector space over the *field* $\mathbb{F}$.
#
# Besides the arithmetic operations in Tutorial 1, Euclidean spaces have
# further structure, for example
# - an [inner product](https://en.wikipedia.org/wiki/Inner_product_space)
# that allows measuring angles between vectors,
# - a <a href="https://en.wikipedia.org/wiki/Norm_(mathematics)">norm</a>
# for measuring the length of a vector, and
# - a <a
# href="https://en.wikipedia.org/wiki/Metric_(mathematics)">metric</a> to
# determine distances between points in space.
#
# For simplicity, we look at the space $\mathbb{R}^n$ -- everything can be
# generalized for the complex case.

# ## Inner products

# An inner product takes two vectors in the space $\mathbb{R}^n$ and
# produces a real number:
#
# $$
#     \langle\cdot, \cdot\rangle : \mathbb{R}^n \times \mathbb{R}^n \to
# \mathbb{R}.
# $$
#
# It is *linear in the first argument*, *conjuagte-symmetric* and
# *positive definite*. Check
# [here](https://en.wikipedia.org/wiki/Inner_product_space#Definition) for
# the exact definitions. The standard inner product on $\mathbb{R}^n$ is
# also known as "dot product" and defined as
#
# $$
#     \langle x, y\rangle_{\mathbb{R}^n} := \sum_{k=1}^n x_k y_k.
# $$
#
# In ODL, it is available as `inner` method both on the space, i.e.,
# `space.inner(vec1, vec2)`, or on the space element, `vec1.inner(vec2)`.
# Let's try it out:

space = odl.rn(3)
vec1 = space.element([1, 2, 3])
vec2 = space.element([1, -1, 1])


space.inner(vec1, vec2)  # 1*1 + 2*(-1) + 3*1 = 2


vec1.inner(vec2)


# We can calculate angles by the formula
#
# $$
#     \cos \angle(x, y) = \frac{\langle x, y \rangle}{\sqrt{\langle x, x
# \rangle \langle y, y \rangle}}
# $$
#
# For example, the angle between $x = (1, 1, 0)$ and $(1, 0, 0)$ should be
# 45 degrees:

RAD2DEG = 180 / np.pi  # conversion factor to degrees
x = space.element([1, 1, 0])
y = space.element([1, 0, 0])

cos_angle_rad = x.inner(y) / np.sqrt(x.inner(x) * y.inner(y))
angle_rad = np.arccos(cos_angle_rad)
angle_deg = angle_rad * RAD2DEG
print('The angle between {} and {} is {} degrees.'.format(x, y, angle_deg))


# ## Norms

# A norm takes a vector in $\mathbb{R}^n$ and maps it to a postive number:
#
# $$
#     \|\cdot\| : \mathbb{R}^n \to [0, \infty)
# $$
#
# It is *positively one-homogeneous*, *positive definite* and satisfies
# the *triangle inequality*, see <a
# href="https://en.wikipedia.org/wiki/Norm_(mathematics)">here</a> for
# details. The standard norm on $\mathbb{R}^n$ is the root of the sum of
# squares of all components:
#
# $$
#     \|x\|_2 = \sqrt{\sum_{k=1}^n x_k^2}
# $$
#
# This is just the same as $\sqrt{\langle x, x\rangle}$ -- the norm is
# *induced* by the inner product.
#
# Again, this function is available as a space method `space.norm(vec)` or
# as method on the element itself, `vec.norm()`:

x = space.element([2, 3, 6])  # has norm sqrt(2*2 + 3*3 + 6*6) = 7


space.norm(x)


x.norm()


# There is a variety of norms on $\mathbb{R}^n$ that are not induced by an
# inner product. They are parametrized by a real number $p \in [1,
# \infty]$ and are defined as
#
# $$
#     \|x\|_p = \left( \sum_{k=1}^n |x_k|^p \right)^{1/p}\quad \text{for }
# p < \infty \text{ and}\quad \|x\|_\infty = \max_k |x_k|\quad \text{for }
# p = \infty.
# $$
#
# (In principle, also $p < 1$ is allowed, but the resulting function is no
# longer a norm)
# In NumPy, norms are implemented in the function [`numpy.linalg.norm`](ht
# tps://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.htm
# l), and the $p$ parameter can be chosen via the `ord` argument.
#
# To use these norm variants in ODL, the space must be initialized with
# the `exponent` argument:

space = odl.rn(3, exponent=1)
space


# Since the value of the exponent deviates from the default 2, it is also
# printed in the representation. Now we can go back to the example above
# and compute the norm in this space:

x = space.element([2, 3, 6])  # has norm 2 + 3 + 6 = 11
x.norm()


# **Important:**
# - Two otherwise equal spaces with differing exponent are considered
# different in ODL. The main reason for this is that only for
# `exponent=2.0`, the norm is induced by an inner product. For all other
# choices, `space.inner` is not defined!
# - A vector "knows" to which space it belongs, i.e. if `x` was created in
# a space with one exponent, it is not considered and element of another
# space with a different exponent.

odl.rn(3, exponent=1) == odl.rn(3, exponent=2)


odl.rn(3, exponent=1).element([1, 2, 3]) in odl.rn(3)


# If we try to call the `inner` method on either a space or an element of
# it, we get an error if `exponent != 2`:

space = odl.rn(3, exponent=1)
x = space.element([1, 2, 3])
y = space.element([1, 0, -1])


# We don't want to see a long traceback, only the error message.
# Don't worry too much about the details in the print statement :-).
try:
    space.inner(x, y)
except Exception as exc:
    print('{}: {}'.format(exc.__class__.__name__, exc))


try:
    x.inner(y)
except Exception as exc:
    print('{}: {}'.format(exc.__class__.__name__, exc))


# ## Metrics

# A metric is a measure of distance between points in space, the
# "endpoints" of vectors. It takes two vectors and produces a positive
# real number:
#
# $$
#     d : \mathbb{R}^n \times \mathbb{R}^n \to [0, \infty)
# $$
#
# A metric is *symmetric*, *sub-additive* and *positive definite* -- check
# the details <a
# href="https://en.wikipedia.org/wiki/Metric_(mathematics)">here</a>. Most
# of the time, metrics are induced by norms in the form
#
# $$
#     d(x, y) = \|x - y\|.
# $$
#
# On a simple space like $\mathbb{R}^n$, metrics that are not of this form
# are unusual, but exist. We go with the default case here.
#
# Using again the norm with exponent $p = 1$, we can compute distances
# with `space.dist(vec1, vec2)` or `vec1.dist(vec2)`:

space = odl.rn(3, exponent=1)
x = space.element([1, 2, 3])
y = space.element([1, 0, -1])
# Distance is: (1-1) + (2-0) + (3-(-1)) = 6
space.dist(x, y)


zero = space.element([0, 0, 0])
x.dist(zero) == x.norm()


# ## Some random but relevant details

# - Calling `space.element()` without arguments allocates memory but does
# not initialize. It is similar to `numpy.empty`:

odl.rn(3).element()  # No guarantee for values


# - Elements with all zeros or all ones can be created with `space.zero()`
# or `space.one()`, respectively:

odl.rn(3).zero()


odl.rn(3).one()


# - The `rn` and `cn` also accept a `dtype` argument. This parameter also
# makes for distinct spaces:

space_f32 = odl.rn(3, dtype='float32')
space_f32


space_f64 = odl.rn(3, dtype='float64')
space_f32 == space_f64


space_f32.element() in space_f64


# - The data container of a space element (by default a Numpy array) can
# be retrieved by the `vec.asarray()` method. Note, however, that
# *mutating this array is not guaranteed to mutate the vector*. To be
# sure, the result must be re-assigned to the vector.

x = odl.rn(3).one()
x_arr = x.asarray()
x_arr[:] = 2
x[:] = x_arr  # no-op in this case
x


# - NumPy arrays are merely wrapped if possible, i.e., if the default
# `'numpy'` back-end is used and if the data types of array and space are
# the same:

array_f32 = np.array([4, 5, 6], dtype='float32')
space_f32 = odl.rn(3, dtype='float32')
x_f32 = space_f32.element(array_f32)
x_f32[:] = -1
array_f32  # modified


space_f64 = odl.rn(3, dtype='float64')
x_f64 = space_f64.element(array_f32)
x_f64[:] = 17
array_f32  # same as before


# ## Wrap-up
#
# We have covered these topics:
# - real and complex Euclidean spaces,
# - inner products,
# - norms,
# - metrics,
# - how to create an empty element,
# - how to create an element of all zeros or ones,
# - how to use different data types,
# - under which conditions Numpy arrays are wrapped rather than copied.
