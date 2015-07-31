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

"""
Core Operator support for ODL.

Operators (module `operator`)
=============================

Core operators
--------------

=================== ===========
Name                Description
=================== ===========
Operator            Basic operator class
LinearOperator      Basic linear operator class
SelfAdjointOperator Class of linear operators whose adjoint is itself
=================== ===========


Operator compositions, sums etc.
--------------------------------

======================== ===========
Name                     Description
======================== ===========
OperatorSum              x --> A(x) + B(x)
OperatorComp             x --> A(B(x))
OperatorPointwiseProduct x --> A(x) * B(x)
OperatorLeftScalarMult   x --> scalar * A(x)
OperatorRightScalarMult  x --> A(scalar * x)
======================== ===========


Linear Operator compositions, sums etc.
---------------------------------------

======================== ===========
Name                     Description
======================== ===========
LinearOperatorSum        x --> A(x) + B(x)
LinearOperatorComp       x --> A(B(x))
LinearOperatorScalarMult x --> scalar * A(x)
======================== ===========

Default (standard) operators (modlule 'default')
================================================

=================== ===========
Name                Description
=================== ===========
ScalingOperator     x --> scalar * x
IdentityOperator    x --> x
=================== ===========

Equation system solvers (module 'solvers')
==========================================

=================== ===========
Name                Description
=================== ===========
landweber           Landweber's method
conjugate_gradient  Conjugate gradient method for the normal equation
gauss_newton        Gauss-Newton method
=================== ===========
"""

from __future__ import absolute_import

__all__ = ('default', 'operator', 'solvers')
