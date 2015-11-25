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
from odl.set.pspace import ProductSpace


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

    if not (isinstance(op.domain, FnBase) or
            isinstance(op.domain, ProductSpace)):
        print('WARNING: The operator domain is not discrete or produc space;',
              'cannot produce matrix representation of it.')
        return

    if not (isinstance(op.range, FnBase) or
            isinstance(op.range, ProductSpace)):
        print('WARNING: The operator range is not discrete; cannot produce',
              'matrix representation of it.')

    # Get the size of the range, and handle ProductSpace
    op_ran = op.range
    op_ran_is_prod_space = isinstance(op_ran, ProductSpace)
    if op_ran_is_prod_space:
        num_ran = op_ran.size
        n = [op_ran[i].size for i in range(num_ran)]
    else:
        num_ran = 1
        n = [op_ran.size]

    # Get the size of the domain, and handle ProductSpace
    op_dom = op.domain
    op_dom_is_prod_space = isinstance(op_dom, ProductSpace)
    if op_dom_is_prod_space:
        num_dom = op_dom.size
        m = [op_dom[i].size for i in range(num_dom)]
    else:
        num_dom = 1
        m = [op_dom.size]

    # Generate the matrix
    matrix = np.zeros([np.sum(n), np.sum(m)])
    tmp_1 = op_ran.element()
    v = op_dom.element()
    index = 0

    for i in range(num_dom):
        for j in range(m[i]):
            v.set_zero()
            if op_dom_is_prod_space:
                v[i][j] = 1.0
            else:
                v[j] = 1.0
            tmp_2 = op(v, out=tmp_1)
            tmp_3 = []
            if op_ran_is_prod_space:
                for k in range(num_ran):
                    tmp_3 = np.concatenate((tmp_3, tmp_2[k].asarray()))
            else:
                tmp_3 = tmp_2.asarray()
            matrix[:, index] = tmp_3
            index += 1

    return matrix
