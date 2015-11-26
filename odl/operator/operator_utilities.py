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
        raise ValueError('The operator is not linear')

    if not (isinstance(op.domain, FnBase) or
            (isinstance(op.domain, ProductSpace) and
            all(isinstance(spc, FnBase) for spc in op.domain))):
        raise TypeError('Operator domain {} is not FnBase, nor ProductSpace'
                        'with only FnBase components'.format(op.domain))

    if not (isinstance(op.range, FnBase) or
            (isinstance(op.range, ProductSpace) and
            all(isinstance(spc, FnBase) for spc in op.range))):
        raise TypeError('Operator range {} is not FnBase, nor ProductSpace'
                        'with only FnBase components'.format(op.range))

    # Get the size of the range, and handle ProductSpace
    # Store for reuse in loop
    op_ran_is_prod_space = isinstance(op.range, ProductSpace)
    if op_ran_is_prod_space:
        num_ran = op.range.size
        n = [ran.size for ran in op.range]
    else:
        num_ran = 1
        n = [op.range.size]

    # Get the size of the domain, and handle ProductSpace
    # Store for reuse in loop
    op_dom_is_prod_space = isinstance(op.domain, ProductSpace)
    if op_dom_is_prod_space:
        num_dom = op.domain.size
        m = [dom.size for dom in op.domain]
    else:
        num_dom = 1
        m = [op.domain.size]

    # Generate the matrix
    matrix = np.zeros([np.sum(n), np.sum(m)])
    tmp_ran = op.range.element()  # Store for reuse in loop
    tmp_dom = op.domain.element()  # Store for reuse in loop
    index = 0

    for i in range(num_dom):
        for j in range(m[i]):
            tmp_dom.set_zero()
            if op_dom_is_prod_space:
                tmp_dom[i][j] = 1.0
            else:
                tmp_dom[j] = 1.0
            op(tmp_dom, out=tmp_ran)
            if op_ran_is_prod_space:
                tmp_result = np.empty(np.sum(n))
                tmp_idx = 0
                for k in range(num_ran):
                    tmp_result[tmp_idx: tmp_idx + op.range[k].size] = \
                        tmp_ran[k].asarray()
                    tmp_idx += op.range[k].size
            else:
                tmp_result = tmp_ran.asarray()
            matrix[:, index] = tmp_result
            index += 1

    return matrix
