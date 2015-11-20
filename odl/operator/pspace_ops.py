# Copyright 2014, 2015 Jonas Adler
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

"""Default operators defined on any (reasonable) space."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy as sp

# ODL imports
from odl.operator.operator import Operator
from odl.set.pspace import ProductSpace


__all__ = ('ProductSpaceOperator', 'ComponentProjection')


class ProductSpaceOperator(Operator):
    """A separable operator on product spaces.

    This is intended for the case where a operator can be decomposed
    as a linear combination of "sub" operators. For example:

    ```
    |A, B, 0| |x|   |Ax + By|
    |0, C, 0| |y| = |  Cy   |
    |0, 0, D| |z|   |  Dz   |
    ```

    """

    def __init__(self, operators, dom=None, ran=None):
        """ TODO
        """

        # Validate input data
        if dom is not None and not isinstance(dom, ProductSpace):
            raise TypeError('space {!r} not a ProductSpace instance.'
                            ''.format(dom))
        if ran is not None and not isinstance(ran, ProductSpace):
            raise TypeError('space {!r} not a ProductSpace instance.'
                            ''.format(dom))

        # Convert ops to sparse representation
        self.ops = sp.sparse.coo_matrix(operators)

        if not all(isinstance(op, Operator) for op in self.ops.data):
            raise TypeError('operators {!r} must be a matrix of operators.'
                            ''.format(operators))

        # Set domain and range (or verify if given)
        if dom is None:
            domains = [None] * self.ops.shape[1]

        if ran is None:
            ranges = [None] * self.ops.shape[0]

        for row, col, op in zip(self.ops.row, self.ops.col, self.ops.data):
            if domains[col] is None:
                domains[col] = op.domain
            elif domains[col] != op.domain:
                raise ValueError('Column {}, has inconsistent domains,'
                                 'got {} and {}'
                                 ''.format(col, domains[col], op.domain))

            if ranges[row] is None:
                ranges[row] = op.range
            elif ranges[row] != op.range:
                raise ValueError('Row {}, has inconsistent ranges,'
                                 'got {} and {}'
                                 ''.format(row, ranges[row], op.range))

        if dom is None:
            for col in range(len(domains)):
                if domains[col] is None:
                    raise ValueError('Col {} empty, unable to determine '
                                     'domain, please use `dom` parameter'
                                     ''.format(col, domains[col]))

            dom = ProductSpace(*domains)

        if ran is None:
            for row in range(len(ranges)):
                if ranges[row] is None:
                    raise ValueError('Row {} empty, unable to determine '
                                     'range, please use `ran` parameter'
                                     ''.format(row, ranges[row]))

            ran = ProductSpace(*ranges)

        # Set linearity
        linear = all(op.is_linear for op in self.ops.data)

        super().__init__(domain=dom, range=ran, linear=linear)

    def _apply(self, x, out):
        """ TODO

        Parameters
        ----------
        todo

        Returns
        -------
        None

        Examples
        --------
        todo
        """
        has_evaluated_row = np.zeros(self.range.size, dtype=bool)
        for i, j, op in zip(self.ops.row, self.ops.col, self.ops.data):
            if not has_evaluated_row[i]:
                op(x[j], out=out[i])
            else:
                # TODO: optimize
                out[i] += op(x[j])

            has_evaluated_row[i] = True

        for i, evaluated in enumerate(has_evaluated_row):
            if not evaluated:
                out[i].set_zero()

    def _call(self, x):
        """ TODO

        Parameters
        ----------
        todo

        Returns
        -------
        todo

        Examples
        --------
        todo

        """
        out = self.range.zero()
        for i, j, op in zip(self.ops.row, self.ops.col, self.ops.data):
            out[i] += op(x[j])
        return out

    @property
    def adjoint(self):
        """ The adjoint is given by taking the conjugate of the scalar
        """
        # TODO: implement
        raise NotImplementedError()

    def __repr__(self):
        """op.__repr__() <==> repr(op)."""
        return 'ProductSpaceOperator({!r})'.format(self.ops)


class ComponentProjection(Operator):
    def __init__(self, space, index):
        self.index = index
        super().__init__(space, space[index], linear=True)

    def _apply(self, x, out):
        out.assign(x[self.index])

    def _call(self, x):
        return x[self.index].copy()

    @property
    def adjoint(self):
        """ The adjoint is given by extending along indices, and setting
        zero along the others
        """
        return ComponentProjectionAdjoint(self.domain, self.index)


class ComponentProjectionAdjoint(Operator):
    def __init__(self, space, index):
        self.index = index
        super().__init__(space[index], space, linear=True)

    def _apply(self, x, out):
        out.set_zero()
        out[self.index] = x

    def _call(self, x):
        """ Extend x into the superspace
        """
        out = self.range.zero()
        out[self.index] = x
        return out

    @property
    def adjoint(self):
        """ The adjoint is given by the projection onto space[index]
        """
        return ComponentProjection(self.range, self.index)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
