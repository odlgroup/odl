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

Abstract and concrete sets (modules :mod:`~odl.set.sets` and :mod:`~odl.set.domain`)
====================================================================================

Simple sets (module `set`)
--------------------------

+--------------------+-------------------------------------------------+
|Name                |Description                                      |
+====================+=================================================+
|`Set`               |**Abstract** base class for mathematical sets    |
+--------------------+-------------------------------------------------+
|`EmptySet`          |Empty set, contains only `None`                  |
+--------------------+-------------------------------------------------+
|`UniversalSet`      |Contains everything                              |
+--------------------+-------------------------------------------------+
|`Integers`          |Set of integers                                  |
+--------------------+-------------------------------------------------+
|`RealNumbers`       |Set of real numbers                              |
+--------------------+-------------------------------------------------+
|`ComplexNumbers`    |Set of `complex` numbers                         |
+--------------------+-------------------------------------------------+
|`Strings`           |Set of fixed-length strings                      |
+--------------------+-------------------------------------------------+
|`CartesianProduct`  |Set of tuples with the i-th entry being an       |
|                    |element of the i-th factor (set)                 |
+--------------------+-------------------------------------------------+

More `complex` sets intended as function domains (module :mod:`~odl.set.domain`)
--------------------------------------------------------------------------------

+-------------------+--------------------------------------------------+
|Name               |Description                                       |
+===================+==================================================+
|`IntervalProd`     |n-dimensional Cartesian product of intervals      |
|                   |forming a rectangular box in :math_`R^n`          |
+-------------------+--------------------------------------------------+
|`Interval`         |1-D special case                                  |
+-------------------+--------------------------------------------------+
|`Rectangle`        |2-D special case                                  |
+-------------------+--------------------------------------------------+
|`Cuboid`           |3-D special case                                  |
+-------------------+--------------------------------------------------+


Abstract vector spaces (modules :mod:`~odl.set.space`, :mod:`~odl.set.pspace`)
==============================================================================

+----------------------+-----------------------------------------------+
|Name                  |Description                                    |
+======================+===============================================+
|`LinearSpace`         |**Abstract** base class for vector spaces over |
|                      |the real or `complex` numbers with addition and|
|                      |scalar multiplication                          |
+----------------------+-----------------------------------------------+
|`LinearProductSpace`  |Cartesian product of linear spaces             |
+----------------------+----------------+------------------------------+
"""

from __future__ import absolute_import

__all__ = ()

from . import domain
from .domain import *
__all__ += domain.__all__

from . import pspace
from .pspace import *
__all__ += pspace.__all__

from . import sets
from .sets import *
__all__ += sets.__all__

from . import space
from .space import *
__all__ += space.__all__
