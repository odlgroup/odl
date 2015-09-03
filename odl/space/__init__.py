# Copyright 2014, 2015 The ODL development group
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

"""Core Spaces and set support.

Abstract and concrete sets (module 'set')
=========================================

==============  ===========
Name            Description
==============  ===========
Set             Base class for mathematical sets
ComplexNumbers  Set of complex numbers
RealNumbers     Set of real numbers
Integers        Set of integers
IntervalProd    Cartesian product of n intervals
Interval        IntervalProd specialization in 1-D
Rectangle       IntervalProd specialization in 2-D
Cube            IntervalProd specialization in 3-D
==============  ===========

Abstract vector spaces (modules 'space' and 'product')
======================================================

General Spaces (module 'space')
-------------------------------

==============  ===========
Name            Description
==============  ===========
LinearSpace     Vector space with addition and scalar multiplication
MetricSpace     A LinearSpace with a metric
NormedSpace     A MetricSpace with a norm and induced metric
HilbertSpace    A NormedSpace with an inner product and induced norm
==============  ===========

Product Spaces (module 'product')
---------------------------------
===================  ===========
Name                 Description
===================  ===========
LinearProductSpace   Cartesian product of linear spaces
MetricProductSpace   Cartesian product of metric spaces
NormedProductSpace   Cartesian product of normed spaces
HilbertProductSpace  Cartesian product of Hilbert spaces
===================  ===========

Concrete vector spaces (modules 'cartesian', 'cuda', 'function')
================================================================

R^n type spaces, CPU implementation (module 'cartesian')
--------------------------------------------------------
===========  ===========
Name         Description
===========  ===========
NTuples      Any set of n elements represented using numpy
Fn           NTuples where the elements are a field
Cn           Basic space of n-tuples of complex numbers
Rn           Basic space of n-tuples of real numbers
===========  ===========

R^n type spaces, CUDA implementation (module 'cuda')
----------------------------------------------------

Requires the compiled extension 'odlpp'

===========  ===========
Name         Description
===========  ===========
CudaFn       Any set of n elements with field operations
CudaRn       Basic space of n-tuples of real numbers in CUDA
===========  ===========

Function spaces (module 'function')
-----------------------------------

=============  ===========
Name           Description
=============  ===========
FunctionSpace  Vector space of functions over some domain
L2             FunctionSpace with the usual integral 2-norm
=============  ===========
"""

__all__ = ('cuda', 'cartesian', 'function', 'product',
           'sequence', 'set', 'space')
