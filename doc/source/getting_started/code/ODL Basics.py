
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


# # ODL Basics

# Welcome to your first tutorial. Here you will learn the basics of linear
# spaces, vectors and related functionality as used throughout ODL. The
# main purpose is to get you accustomed to working with the library, the
# "look and feel" of typical code and some building blocks and patterns
# that repeat themselves in other parts. We will explain them as we go
# along.
#
# At this point, you should have a working installation of ODL on your
# machine. If that's not the case, check out [the installation instruction
# s](https://odlgroup.github.io/odl/getting_started/installing.html) and
# come back when you're finished!

# We import this for Python 2 compatibility
from __future__ import print_function
import odl


# Everything starts with importing the package. If you want to explore it,
# you can type `odl.` and hit `<TAB>` in an interactive shell (for example
# [IPython](https://ipython.org/)) to see all its top-level contents. The
# `odl` package is organized into several sub-packages, but the most
# important core functions and classes are available at the top level, so
# you don't have to type `odl.subpackage1.subpackage2.module.function` all
# the time.
#
# Now the simplest kind of mathematical object you can create in ODL is a
# *vector*. If you're familiar with [NumPy](http://www.numpy.org/), you
# can simply think of a `numpy.ndarray`.

x = odl.vector([1, 2, 3])
print(x)


# In fact, the default implementation of a `vector` is just a thin wrapper
# around `numpy.ndarray`. However, this back-end can be switched for
# something else, e.g., a CUDA-based data container. On the surface it
# will look exactly the same, and the differences will be entirely under
# the hood. More on that later.
#
# Let's look at the representation (`repr`) of `x` to get some additional
# information beyond the pretty-printing above:

x  # or print(repr(x))


# Hm, this is interesting. Obviously, there's more to this vector than
# just its values. The line printed above suggests that it is an `element`
# of something called `fn`. Let's see if we can just look at the first
# part of that expression, of course prepending the package name and a
# dot, a trick we know from NumPy:

odl.fn(3, 'int')


# Okay, the code works, so we can try to run the whole line in the
# representation of `x`.

odl.fn(3, 'int').element([1, 2, 3])


odl.fn(3, 'int').element([1, 2, 3]) == x


odl.fn(3, 'int').element([1, 2, 3]) is x


# So we can recreate `x` with its `repr` code, but it's a new vector, not
# identical to `x`. Of course, we should be able to create other vectors,
# too, and add them, subtract, multiply and so on.

y = odl.vector([4, 5, 6])


x + y


x - y


x * y


# So that works exactly as in NumPy, too. Let's go a step further and
# check out what this first part `fn(3, 'int')` really means. We already
# saw that the code runs and produces an output, so we assign it to a
# variable and look at its properties.

s = odl.fn(3, 'int')


print(s)


type(s)


# So `s` is an object of type `NumpyFn`. That gives at least a hint that
# it uses NumPy internally, but what is it? More experienced Python users
# know that we can print the
# [docstring](https://www.python.org/dev/peps/pep-0257/#what-is-a-
# docstring) of any Python object to get more information. Docstrings are
# strings stored in the object as `object.__doc__` attribute:

print(s.__doc__)


# This text actually gives quite a lot of information. It tells us that
# our object `s` created by the command `odl.fn(3, 'int')`
# - is a [vector space](https://en.wikipedia.org/wiki/Vector_space),
# - contains vectors of a fixed length `n` (that's the "n" in `Fn`),
# - is constructed over the <a
# href="https://en.wikipedia.org/wiki/Field_(mathematics)">field</a> (the
# "F" in `Fn`) of real or complex numbers (usually) and
# - represents its elements as `NumpyFnVector` instances.
#
# The last part is easiest to check since we already have an element `x`
# to test:

type(x)


# Indeed, the type is the same as the one mentioned in the docstring. We
# should be able to probe also the other properties.

s.field


s.size  # the equivalent of "length", NumPy users will recognize it


# Also the word "contains" resonates with experienced Python users. They
# think of the `__contains__` method and want to probe it with the `object
# in container` expression. Here, the vector should be in the space:

x in s


# So we can check member ship of vectors in spaces. Further up, we created
# another vector `y` from the same space, so it should be contained, too.

y in s


# This raises, of course, the question which properties of a space are
# important such that they contain certain types of vectors. As it turns
# out, pretty much all of them. For example, if we change the `size`
# parameter (was 3) to anything else, no matter if smaller or larger, `x`
# won't be contained any longer:

x in odl.fn(2, 'int')


x in odl.fn(4, 'int')


# The second argument, currently `'int'`, looks like a data type, so we
# can check here, too, if it's important for membership.

x in odl.fn(3, 'uint')  # unsigned integers


x in odl.fn(3, 'float')  # same as float64, i.e. double precision floats


# We see that membership really only holds for the exact same data type.
# There are even more, currently less obvious, parameters which can be
# tuned. We will cover them in later lessons. Just as a preview: *all of
# them* are important for space membership.

# ## Wrap-up

# In summary, we have covered the following topics:
# - How to create a vector
# - What vectors actually represent (members of vector spaces)
# - How to create a vector space of a certain size and data type
# - Properties of vector spaces
# - When a vector is contained in a space and how to check it
