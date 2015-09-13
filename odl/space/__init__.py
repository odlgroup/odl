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

Abstract and concrete sets (modules 'set' and 'domain')
=======================================================

Simple sets (module 'set')
--------------------------

+-------------------+-------------------+------------------------------+
|Name               |Direct             |Description                   |
|                   |ancestors          |                              |
+===================+===================+==============================+
|``Set``            |`object`           |Base class for mathematical   |
|                   |                   |sets                          |
+-------------------+-------------------+------------------------------+
|``Integers``       |``Set``            |Set of integers               |
+-------------------+-------------------+------------------------------+
|``RealNumbers``    |``Set``            |Set of real numbers           |
+-------------------+-------------------+------------------------------+
|``ComplexNumbers`` |``Set``            |Set of complex numbers        |
+-------------------+-------------------+------------------------------+
|``Strings``        |``Set``            |Set of fixed-length strings   |
+-------------------+-------------------+------------------------------+

More complex sets intended as function domains (module 'domain')
----------------------------------------------------------------

+-------------------+-------------------+------------------------------+
|Name               |Direct             |Description                   |
|                   |ancestors          |                              |
+===================+===================+==============================+
|``IntervalProd``   |``Set``            |Cartesian product of intervals|
+-------------------+-------------------+------------------------------+
|``Interval``       |``IntervalProd``   |1-D special case              |
+-------------------+-------------------+------------------------------+
|``Rectangle``      |``IntervalProd``   |1-D special case              |
+-------------------+-------------------+------------------------------+
|``Cube``           |``IntervalProd``   |1-D special case              |
+-------------------+-------------------+------------------------------+


Abstract vector spaces (modules 'space', 'product', 'function')
===============================================================

+----------------------+----------------+------------------------------+
|Name                  |Direct          |Description                   |
|                      |ancestors       |                              |
+======================+================+==============================+
|``LinearSpace``       |``Set``         |Abstract vector space with    |
|                      |                |addition and scalar           |
|                      |                |multiplication                |
+----------------------+----------------+------------------------------+
|``LinearProductSpace``|``LinearSpace`` |Cartesian product of linear   |
|                      |                |spaces                        |
+----------------------+----------------+------------------------------+
|``FunctionSet``       |``Set``         |Set of functions with common  |
|                      |                |domain and range              |
+----------------------+----------------+------------------------------+
|``FunctionSpace``     |``FunctionSet``,|Function set where the range  |
|                      |``LinearSpace`` |is a field                    |
+----------------------+----------------+------------------------------+

Concrete vector spaces (modules 'cartesian', 'cuda', 'default')
===============================================================

:math:`R^n` type spaces, NumPy implementation (module 'cartesian')
------------------------------------------------------------------

+----------------------+----------------+------------------------------+
|Name                  |Direct          |Description                   |
|                      |ancestors       |                              |
+======================+================+==============================+
|``NTuplesBase``       |``Set``         |Abstract base class for sets  |
|                      |                |of n-tuples of various types  |
+----------------------+----------------+------------------------------+
|``Ntuples``           |``NTuplesBase`` |Set of n-tuples of almost     |
|                      |                |arbitrary type                |
+----------------------+----------------+------------------------------+
|``FnBase``            |``NTuplesBase`` |Abstract base class for spaces|
|                      |                |of n-tuples over a field      |
+----------------------+----------------+------------------------------+
|``Fn``                |``FnBase``      |Space of n-tuples over a      |
|                      |                |field allowing any scalar     |
|                      |                |data type                     |
+----------------------+----------------+------------------------------+
|``Cn``                |``Fn``          |Space of n-tuples of complex  |
|                      |                |numbers                       |
+----------------------+----------------+------------------------------+
|``Rn``                |``Fn``          |Space of n-tuples of real     |
|                      |                |numbers                       |
+----------------------+----------------+------------------------------+

:math:`R^n` type spaces, CUDA implementation (module 'cuda')
------------------------------------------------------------

Requires the compiled extension 'odlpp'

+----------------------+----------------+------------------------------+
|Name                  |Direct          |Description                   |
|                      |ancestors       |                              |
+======================+================+==============================+
|``CudaNtuples``       |``NTuplesBase`` |Set of n-tuples of almost     |
|                      |                |arbitrary type                |
+----------------------+----------------+------------------------------+
|``CudaFn``            |``FnBase``      |Space of n-tuples over a      |
|                      |                |field allowing any scalar     |
|                      |                |data type                     |
+----------------------+----------------+------------------------------+
|``CudaCn``            |TODO            |(Space of n-tuples of complex |
|                      |                |numbers)                      |
+----------------------+----------------+------------------------------+
|``CudaRn``            |``CudaFn``      |Space of n-tuples of real     |
|                      |                |numbers                       |
+----------------------+----------------+------------------------------+

Function spaces (module 'default')
-----------------------------------

+----------------------+-----------------+-----------------------------+
|Name                  |Direct           |Description                  |
|                      |ancestors        |                             |
+======================+=================+=============================+
|``L2``                |``FunctionSpace``|Square-integrable functions  |
|                      |                 |taking real or complex values|
+----------------------+-----------------+-----------------------------+
"""

from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals

__all__ = ('cartesian', 'cuda', 'default', 'domain',  'function',
           'product', 'sequence', 'set', 'space')
