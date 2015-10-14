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

"""Core Operator support for ODL.

Operators (module `operator`)
=============================

Core operators
--------------

+-----------------------+----------------------------------------------+
|Class name             |Description                                   |
+=======================+==============================================+
|`Operator`             |**Abstract** basic class for (mathematical)   |
|                       |operators                                     |
+-----------------------+----------------------------------------------+

Operator compositions, sums etc.
--------------------------------

+----------------------------+-----------------------------------------+
|Class name                  |Description                              |
+============================+=========================================+
|`OperatorSum`               |Sum of two operators, `S = A + B`,       |
|                            |defined as                               |
|                            |`x` --> `(A + B)(x) = A(x) + B(x)`       |
+----------------------------+-----------------------------------------+
|`OperatorComp`              |Composition of two operators,            |
|                            |`C = A o B` defined as                   |
|                            |`x` --> `(A o B)(x) = A(B(x))`           |
+----------------------------+-----------------------------------------+
|`OperatorPointwiseProduct`  |Product of two operators,`P = A * B`,    |
|                            |defined as                               |
|                            |`x --> (A * B)(x) = A(x) * B(x)`.        |
+----------------------------+-----------------------------------------+
|`OperatorLeftScalarMult`    |Multiplication of an operator from left  |
|                            |with a scalar, `L = c * A`, defined as   |
|                            |`x --> (c * A)(x) = c * A(x)`            |
+----------------------------+-----------------------------------------+
|`OperatorRightScalarMult`   |Multiplication of an operator from right |
|                            |with a scalar, `S = A * c`, defined by   |
|                            |`x --> (A * c)(x) =  A(c * x)`           |
+----------------------------+-----------------------------------------+

Factory functions
-----------------

+-------------------+--------------------------------------------------+
|Name               |Description                                       |
+===================+==================================================+
|`operator()`       |Create an `Operator` by specifying either a       |
|                   |`call` or an `apply` method (or both) for         |
|                   |evaluation.                                       |
+-------------------+--------------------------------------------------+

Default (standard) operators (modlule `default_ops`)
====================================================

+--------------------+-------------------------------------------------+
|Class name          |Description                                      |
+====================+=================================================+
|`ScalingOperator`   |Multiplication with a scalar `s`, defined as     |
|                    |`x` --> `s * x`                                  |
+--------------------+-------------------------------------------------+
|`ZeroOperator`      |Multiplication with 0, defined as                |
|                    |`x` --> `0 * x`                                  |
+--------------------+-------------------------------------------------+
|`IdentityOperator`  |Multiplication with 1, defined as                |
|                    |`x` --> `1 * x`                                  |
+--------------------+-------------------------------------------------+
|`LinCombOperator`   |Linear combination of two space elements with two|
|                    |fixed scalars `a` and `b`, defined as            |
|                    |`(x, y)` --> `a * x + b * y`                     |
+--------------------+-------------------------------------------------+
|`MultiplyOperator`  |Multiplication of two space elements, defined as |
|                    |`(x, y)` --> `x * y`                             |
+--------------------+-------------------------------------------------+

Equation system solvers (module `solvers`)
==========================================

+----------------------+-----------------------------------------------+
|Name                  |Description                                    |
+======================+===============================================+
|`landweber()`         |Landweber's iterative method                   |
+----------------------+-----------------------------------------------+
|`conjugate_gradient()`|Conjugate gradient method for the normal       |
|                      |equation                                       |
+----------------------+-----------------------------------------------+
|`gauss_newton()`      |Gauss-Newton iterative method                  |
+----------------------+-----------------------------------------------+
"""

from __future__ import absolute_import

__all__ = ()

from . import default_ops
from .default_ops import *
__all__ += default_ops.__all__

from . import operator
from .operator import *
__all__ += operator.__all__

from . import solvers
