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
Core Operator support for ODL
=============================

================================================================================
Core operators (operator)
================================================================================
Operator                Has method __call__() and/or apply()
LinearOperator          Operator whose operation is linear.
SelfAdjointOperator     Linear operator whose adjoint is itself
================================================================================

================================================================================
Operator compositions, sums etc (operator)
================================================================================
OperatorSum                         A(x) + B(x)
OperatorComp                        A(B(x))
OperatorPointwiseProduct            A(x)*B(x)
OperatorLeftScalarMult              s*A(x)
OperatorRightScalarMult             A(s*x)
================================================================================

================================================================================
Linear Operator compositions, sums etc (operator)
================================================================================
LinearOperatorSum                   A(x) + B(x)
LinearOperatorComp                  A(B(x))
LinearOperatorScalarMult            s*A(x)
================================================================================

================================================================================
Default (standard) operators (default)
================================================================================
ScalingOperator         Scales a vector by a scalar
IdentityOperator        Identity operator
================================================================================

================================================================================
Equation system solvers (solvers)
================================================================================
landweber               The landweber method
conjugate_gradient      The Conjugate gradient Noraml Equations method
gauss_newton            The Gauss Newton method
================================================================================

"""

from __future__ import absolute_import

__all__ = ['default', 'operator', 'solvers']
