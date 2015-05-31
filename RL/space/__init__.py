# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""
Core Spaces and set support

Core classes, sets
==========================================================================
Abstract sets (set)
==========================================================================
Set             A mathematical set, has method 'contains'
ComplexNumbers  The set of complex numbers
RealNumbers     The set of real numbers
Integers        The set of integers
IntervalProd    The Cartesian product of n intervals
Interval        IntervalProd specialization in 1-D
Rectanggle      IntervalProd specialization in 2-D
==========================================================================

Contains several abstract space definitions as 'Set' implementations
==========================================================================
General Spaces (space)
==========================================================================
LinearSpace     Vector space with addition, scalar multiplication
MetricSpace     A LinearSpace with a metric
NormedSpace     A MetricSpace where the metric is induced by a norm
HilbertSpace    A NormedSpace where the norm is induced by a inner product
==========================================================================

==========================================================================
Product Spaces (product)
==========================================================================
LinearProductSpace      Vector space created by the Cartesian product
                        of other LinearSpaces
MetricProductSpace      A LinearProductSpace with a metric
NormedProductSpace      A MetricProductSpace where the metric is induced
                        by a norm
HilbertProductSpace     A HilbertProductSpace where the norm is induced by
                        a inner product
==========================================================================

Also has a set of concerete implemenation of standard spaces
==========================================================================
R^n type spaces (euclidean, cuda)
==========================================================================
Rn                  Basic space of n-tuples of real numbers, uses numpy.
NormedRn            R^n with some norm
EuclidRn            R^n with the usual Euclidean norm and inner product
CudaRn              EuclidRn implemented in CUDA
==========================================================================

==========================================================================
Function spaces (function)
==========================================================================
FunctionSpace       The space of functions over some domain
L2                  FunctionSpace with the usual L2-norm
==========================================================================

==========================================================================
Discretizations of function spaces (discretizations)
==========================================================================
uniform_discretization  Discretization of an Interval using some Rn
pixel_discretization    Discretization of an Rectangle using some Rn
==========================================================================
"""

__all__ = ['cuda', 'discretizations', 'euclidean', 'function', 'product',
           'sequence', 'set', 'space']
