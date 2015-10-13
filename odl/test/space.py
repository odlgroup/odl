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

import warnings
import numpy as np
from itertools import product

from odl.set.pspace import ProductSpace
from odl.operator.operator import LinearOperator
from odl.space.base_ntuples import FnBase, NtuplesBase
from odl.discr.l2_discr import DiscreteL2
from odl.test.examples import scalar_examples, vector_examples

__all__ = ('SpaceTest',)

class SpaceTest(object):
    def __init__(self, space):
        self.space = space

    def _norm_positive(self):
        print('\ntesting positivity, ||x|| >= 0\n')
        print('error = -||x||')

        num_failed = 0
        for [name, vec] in vector_examples(self.space):
            norm = vec.norm()

            if norm < 0:
                print('x={:25s} : ||x||={}'
                        ''.format(name, error))
                num_failed += 1

        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:
            print('*** FAILED {} TEST CASES ***'.format(num_failed))

    def _norm_subadditive(self):
        print('\ntesting subadditivity, ||x+y|| <= ||x|| + ||y||\n')
        print('error = ||x+y|| - ||x|| + ||y||')
        
        num_failed = 0
        for [name_x, vec_x] in vector_examples(self.space):
            norm_x = vec_x.norm()
            for [name_y, vec_y] in vector_examples(self.space):
                norm_xy = (vec_x + vec_y).norm()
                norm_y = vec_y.norm()

            error = norm_xy-norm_x-norm_y

            if error > 0:
                print('x={:25s} x={:25s}: error={}'
                        ''.format(name, error))
                num_failed += 1

        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:
            print('*** FAILED {} TEST CASES ***'.format(num_failed))

    def _norm_homogeneity(self):
        print('\ntesting homogeneity, ||a*x|| = |a| ||x||\n')
        print('error = abs(||a*x|| - |a| ||x||)')
        
        num_failed = 0
        for [name, vec] in vector_examples(self.space):
            for scalar in scalar_examples(self.space):
                error = abs((scalar * vec).norm() - abs(scalar) * vec.norm())
                if error > 0.00001:
                    print('x={:25s} a={}: ||x||={}'
                            ''.format(name, scalar, error))
                    num_failed += 1

        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:
            print('*** FAILED {} TEST CASES ***'.format(num_failed))

    def norm(self):
        print('\n== Verifying norm ==\n')

        try:
            self.space.zero().norm()
        except NotImplementedError:
            print('Space is not normed')
            return
            
        print('testing ||0|| = {}. Expected 0.0'.format(self.space.zero().norm()))
        self._norm_positive()
        self._norm_subadditive()
        self._norm_homogeneity()

    def run_tests(self):
        """Runs all tests on this operator
        """
        print('\n== RUNNING ALL TESTS ==\n')
        print('Space = {}'.format(self.space))

        self.norm()

    def __str__(self):
        return 'SpaceTest({})'.format(self.space)

    def __repr__(self):
        return 'SpaceTest({!r})'.format(self.space)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
