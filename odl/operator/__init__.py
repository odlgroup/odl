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

"""Mathematical operators in ODL.

Operators in ODL are represented by the abstract :class:`Operator`
class. As an *abstract class*, it cannot be used directly but must be
subclassed for concrete implementation. To define your own operator,
you start by writing::

    class MyOperator(odl.Operator):
        ...

:class:`Operator` has a couple of *abstract methods* which need to be
explicitly overridden by any subclass, namely

:attr:`domain`: :class:`Set`
    Set of elements to which the operator can be applied
:attr:`range`: :class:`Set`
    Set in which the operator takes values

As a simple example, you can implement the matrix multiplication
operator

    :math:`\mathcal{A}: \mathbb{R^m} \\to \mathbb{R^n}, \quad
    \mathcal{A}(x) = Ax`

for a matrix :math:`A\\in \mathbb{R}^{n\\times m}` as follows::

    class MatVecOperator(odl.Operator):
        def __init__(self, matrix):
            assert isinstance(matrix, np.ndarray)
            self.matrix = matrix
            self.domain = odl.Rn(matrix.shape[1])
            self.range = odl.Rn(matrix.shape[0])

In addition, an :class:`Operator` needs at least one way of evaluation,
*in-place* or *out-of-place*.

- In-place evaluation means that the operator is evaluated on a
  ``domain`` element, and the result is written to an *already existing*
  ``range`` element. To implement this behavior, create the (private)
  :attr:`_apply()` method, in the above example::

    class MatVecOperator(odl.Operator):
        ...
        def _apply(x, out):
            self.matrix.dot(x, out=out)

  In-place evaluation is usually more efficient and should be used
  *whenever possible*.

- Out-of-place evaluation means that the
  operator is evaluated on a ``domain`` element, and the result is
  written to a *newly allocated* ``range`` element. To implement this
  behavior, create the (private) :obj:`_call()` method, in the above
  example::

    class MatVecOperator(odl.Operator):
        ...
        def _call(x):
            return self.matrix.dot(x)

  Out-of-place evaluation is usually less efficient since it requires
  allocation of an array and a full copy and should be *generally
  avoided*.

**Important:** Do not call these methods directly. Use the call pattern
``operator(x)`` or ``operator(x, out=y)``, e.g.::

    matrix = 2 * np.eye(3)
    operator = MatVecOperator(matrix)
    x = odl.Rn(3).one()
    y = odl.Rn(3).element()

    # Out-of-place evaluation
    y = operator(x)

    # In-place evaluation
    operator(x, out=y)

This public calling interface is type-checked, so the private methods
do not need to implement type checks.
"""

from __future__ import absolute_import

__all__ = ()

from . import default_ops
from .default_ops import *
__all__ += default_ops.__all__

from . import operator
from .operator import *
__all__ += operator.__all__
