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

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()


def landweberBase(operator, x, rhs, iterations=1, omega=1):
    """ Straightforward implementation of Landweber iteration
    """
    for _ in range(iterations):
        x = x - omega * operator.T(operator(x)-rhs)

    return x


def conjugate_gradient_base(op, x, rhs, iterations=1):
    """ Non-optimized CGN
    """
    d = rhs - op(x)
    p = op.T(d)
    s = p.copy()

    for _ in range(iterations):
        q = op(p)
        norms2 = s.norm()**2
        a = norms2 / q.norm()**2
        x = x + a*p
        d = d - a*q
        s = op.T(d)
        b = s.norm()**2/norms2
        p = s + b*p