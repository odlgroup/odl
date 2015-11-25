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

"""Usefull utility functions on discrete spaces (i.e., either Rn/Cn or
discretized function spaces), for example obtaining a matrix representation of
an operator. """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np

# Internal
from odl.space.base_ntuples import FnBase


def matrix_representation(op):
    """Returns a matrix representation of a linear operator.

    Parameters
    ----------
    op : :class:`~odl.Operator`
        The linear operator of which one wants a matrix representation.

    Returns
    ----------
    matrix : `numpy.ndarray`
        The matrix representation of the operator.

    Notes
    ----------
    The algorithm works by letting the operator act on all unit vectors, and
    stacking the output as a matrix.

    """

    if not op.is_linear:
        print('WARNING: The operator is not linear; cannot produce matrix',
              'representation of it.')
        return

    if not isinstance(op.domain, FnBase):
        print('WARNING: The operator domain is not discrete or produc space;',
              'cannot produce matrix representation of it.')
        return

    if not isinstance(op.range, FnBase):
        print('WARNING: The operator range is not discrete; cannot produce',
              'matrix representation of it.')
        return

    n = op.range.size
    m = op.domain.size

    matrix = np.zeros([n, m])
    v = op.domain.element()
    tmp = op.range.element()
    for i in range(m):
        v.set_zero()
        v[i] = 1.0
        matrix[:, i] = op(v, out=tmp)

    return matrix
