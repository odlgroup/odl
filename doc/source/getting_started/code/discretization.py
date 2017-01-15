
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


# ## Discretization example
#
# We create a discretized space of functions in $\mathbb{R}^2$ and do some
# simple calculations.

import odl


# First we create a discretization of the space $L^2(\Omega)$ with a
# rectangular domain $\Omega \subset \mathbb{R}^2$. The simplest way to do
# this is to use the `uniform_discr` function:

l2_discr = odl.uniform_discr([-1, -1], [1, 1], nsamples=[200, 200])


# This function returns a Lebesgue $L^p$ space with default exponent $p=2$
# and nearest neighbor interpolation, which can be changed by parameters.
# We can check its attributes:

l2_discr.exponent  # The p in L^p


l2_discr.domain  # The domain Omega of the discretized functions


l2_discr.interp  # Interpolation scheme per dimension


# Now we can create elements in this space and do some calculations with
# them. All these operations use [Numpy](http://www.numpy.org/) for fast
# array computations:

get_ipython().magic('matplotlib inline')
# Create a Shepp-Logan phantom in the chosen space
phantom = odl.util.phantom.shepp_logan(l2_discr, modified=True)
# Storing the figure to suppress (double) instant plotting
fig = phantom.show('The famous Shepp-Logan phantom')


phantom.norm()  # Approximation to the true function norm


one = l2_discr.one()  # The constant one function
one.norm()  # Square root of the domain area


lincomb = 2 * phantom + 3 * one  # This creates a new element 'lincomb'
lincomb.norm()


# We can also do computations in place to avoid allocation of new memory.
# The `element()` method can be used to simply allocate memory for a re-
# usable temporary element which can hold intermediate values:

buffer = l2_discr.element()  # Uninitialized memory
# Same linear combination as above, but the result is stored in 'buffer'
buffer.lincomb(2, phantom, 3, one)
buffer.norm()
