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
Core Operator support for RL
============================

======================================================================================
Core operators (operator)
======================================================================================
Operator                Has method __call__() and/or apply()
LinearOperator          Operator whose operation is linear.
SelfAdjointOperator     Linear operator whose adjoint is itself
======================================================================================

======================================================================================
Operator compositions, sums etc (operator)
======================================================================================
OperatorSum                         OperatorSum(A,B)(x) = A(x) + B(x)
OperatorComposition                 OperatorComposition(A,B)(x) = A(B(x))
OperatorPointwiseProduct            OperatorPointwiseProduct(A,B)(x) = A(x)*B(x)
OperatorLeftScalarMultiplication    OperatorLeftScalarMultiplication(A,s)(x) = s*A(x)
OperatorRightScalarMultiplication   OperatorRightScalarMultiplication(A,s)(x) = A(s*x)
======================================================================================

======================================================================================
Linear Operator compositions, sums etc (operator)
======================================================================================
LinearOperatorSum                   LinearOperatorSum(A,B)(x) = A(x) + B(x)
LinearOperatorComposition           LinearOperatorComposition(A,B)(x) = A(B(x))
LinearOperatorScalarMultiplication  LinearOperatorScalarMultiplication(A,s)(x) = s*A(x)
======================================================================================

======================================================================================
Default (standard) operators (default_operators)
======================================================================================
ScalingOperator         Scales a vector by a scalar
IdentityOperator        Identity operator
======================================================================================

======================================================================================
Equation system solvers (solvers)
======================================================================================
landweber               The landweber method
conjugate_gradient      The Conjugate gradient Noraml Equations method
gauss_newton            The Gauss Newton method
======================================================================================

"""

from __future__ import absolute_import

__all__ = ['default_operators', 'function', 'functional', 'operator',
           'solvers']
