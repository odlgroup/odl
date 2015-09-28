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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

# External module imports
import unittest
import numpy as np
import scipy as sp
from math import sqrt, ceil
from textwrap import dedent

# ODL imports
from odl import Rn, Cn, Fn, ConstWeightedInner, MatrixWeightedInner
from odl.util.testutils import ODLTestCase

# TODO: add tests for:
# * Ntuples (different data types)
# * metric, normed, Hilbert space variants
# * Cn
# * Rn, Cn with non-standard data types
# * vector multiplication
# * MatVecOperator


class RnTest(ODLTestCase):
    @staticmethod
    def _vectors(rn):
        # Generate numpy vectors
        y = np.random.rand(rn.size)
        x = np.random.rand(rn.size)
        z = np.random.rand(rn.size)

        # Make rn vectors
        yVec = rn.element(y)
        xVec = rn.element(x)
        zVec = rn.element(z)
        return x, y, z, xVec, yVec, zVec

    def _test_lincomb(self, a, b, n=10):
        # Validates lincomb against the result on host with randomized
        # data and given a,b
        rn = Rn(n)

        # Unaliased arguments
        x, y, z, xVec, yVec, zVec = self._vectors(rn)

        z[:] = a*x + b*y
        rn.lincomb(zVec, a, xVec, b, yVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # First argument aliased with output
        x, y, z, xVec, yVec, zVec = self._vectors(rn)

        z[:] = a*z + b*y
        rn.lincomb(zVec, a, zVec, b, yVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # Second argument aliased with output
        x, y, z, xVec, yVec, zVec = self._vectors(rn)

        z[:] = a*x + b*z
        rn.lincomb(zVec, a, xVec, b, zVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # Both arguments aliased with each other
        x, y, z, xVec, yVec, zVec = self._vectors(rn)

        z[:] = a*x + b*x
        rn.lincomb(zVec, a, xVec, b, xVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # All aliased
        x, y, z, xVec, yVec, zVec = self._vectors(rn)
        z[:] = a*z + b*z
        rn.lincomb(zVec, a, zVec, b, zVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

    def test_lincomb(self):
        scalar_values = [0, 1, -1, 3.41]
        for a in scalar_values:
            for b in scalar_values:
                self._test_lincomb(a, b)


class OperatorOverloadTest(ODLTestCase):
    def _test_unary_operator(self, function, n=10):
        """ Verifies that the statement y=function(x) gives equivalent
        results to Numpy.
        """
        rn = Rn(n)

        x_arr = np.random.rand(n)
        y_arr = function(x_arr)

        x = rn.element(x_arr)
        y = function(x)

        self.assertAllAlmostEquals(x, x_arr)
        self.assertAllAlmostEquals(y, y_arr)

    def _test_binary_operator(self, function, n=10):
        """ Verifies that the statement z=function(x,y) gives equivalent
        results to Numpy.
        """
        rn = Rn(n)

        x_arr = np.random.rand(n)
        y_arr = np.random.rand(n)
        z_arr = function(x_arr, y_arr)

        x = rn.element(x_arr)
        y = rn.element(y_arr)
        z = function(x, y)

        self.assertAllAlmostEquals(x, x_arr)
        self.assertAllAlmostEquals(y, y_arr)
        self.assertAllAlmostEquals(z, z_arr)

    def test_operators(self):
        """ Test of all operator overloads against the corresponding
        Numpy implementation
        """
        # Unary operators
        self._test_unary_operator(lambda x: +x)
        self._test_unary_operator(lambda x: -x)

        # Scalar multiplication
        for scalar in [-31.2, -1, 0, 1, 2.13]:
            def imul(x):
                x *= scalar
            self._test_unary_operator(imul)
            self._test_unary_operator(lambda x: x*scalar)

        # Scalar division
        for scalar in [-31.2, -1, 1, 2.13]:
            def idiv(x):
                x /= scalar
            self._test_unary_operator(idiv)
            self._test_unary_operator(lambda x: x/scalar)

        # Incremental operations
        def iadd(x, y):
            x += y

        def isub(x, y):
            x -= y

        self._test_binary_operator(iadd)
        self._test_binary_operator(isub)

        # Incremental operators with aliased inputs
        def iadd_aliased(x):
            x += x

        def isub_aliased(x):
            x -= x
        self._test_unary_operator(iadd_aliased)
        self._test_unary_operator(isub_aliased)

        # Binary operators
        self._test_binary_operator(lambda x, y: x + y)
        self._test_binary_operator(lambda x, y: x - y)

        # Binary with aliased inputs
        self._test_unary_operator(lambda x: x + x)
        self._test_unary_operator(lambda x: x - x)


class MethodTest(ODLTestCase):
    def test_norm(self):
        r3 = Rn(3)
        xd = r3.element([1, 2, 3])

        correct_norm = sqrt(1**2 + 2**2 + 3**2)
        self.assertAlmostEquals(r3.norm(xd), correct_norm)

    def test_inner(self):
        r3 = Rn(3)
        xd = r3.element([1, 2, 3])
        yd = r3.element([5, -3, 9])

        correct_inner = 1*5 + 2*(-3) + 3*9
        self.assertAlmostEquals(r3.inner(xd, yd), correct_inner)


class GetSetTest(ODLTestCase):
    def test_setitem(self):
        r3 = Rn(3)
        x = r3.element([42, 42, 42])

        for index in [0, 1, 2, -1, -2, -3]:
            x[index] = index
            self.assertAlmostEquals(x[index], index)

    def test_setitem_index_error(self):
        r3 = Rn(3)
        x = r3.element([1, 2, 3])

        with self.assertRaises(IndexError):
            x[-4] = 0

        with self.assertRaises(IndexError):
            x[3] = 0

    def _test_getslice(self, slice):
        # Validate get against python list behaviour
        r6 = Rn(6)
        y = [0, 1, 2, 3, 4, 5]
        x = r6.element(y)

        self.assertAllAlmostEquals(x[slice].data, y[slice])

    def test_getslice(self):
        # Tests getting all combinations of slices
        steps = [None, -2, -1, 1, 2]
        starts = [None, -1, -3, 0, 2, 5]
        ends = [None, -1, -3, 0, 2, 5]

        for start in starts:
            for end in ends:
                for step in steps:
                    self._test_getslice(slice(start, end, step))

    def _test_setslice(self, slice):
        # Validate set against python list behaviour
        r6 = Rn(6)
        z = [7, 8, 9, 10, 11, 10]
        y = [0, 1, 2, 3, 4, 5]
        x = r6.element(y)

        x[slice] = z[slice]
        y[slice] = z[slice]
        self.assertAllAlmostEquals(x, y)

    def test_setslice(self):
        # Tests a range of combination of slices
        steps = [None, -2, -1, 1, 2]
        starts = [None, -1, -3, 0, 2, 5, 10]
        ends = [None, -1, -3, 0, 2, 5, 10]

        for start in starts:
            for end in ends:
                for step in steps:
                    self._test_setslice(slice(start, end, step))

    def test_setslice_index_error(self):
        r3 = Rn(3)
        xd = r3.element([1, 2, 3])

        # Bad slice
        with self.assertRaises(ValueError):
            xd[10:13] = [1, 2, 3]

        # Bad size of rhs
        with self.assertRaises(ValueError):
            xd[:] = []

        with self.assertRaises(ValueError):
            xd[:] = [1, 2]

        with self.assertRaises(ValueError):
            xd[:] = [1, 2, 3, 4]


class MatrixWeightedInnerTest(ODLTestCase):
    @staticmethod
    def _vectors(fn):
        # Generate numpy vectors, real or complex
        if isinstance(fn, Rn):
            xarr = np.random.rand(fn.size)
            yarr = np.random.rand(fn.size)
        else:
            xarr = np.random.rand(fn.size) + 1j * np.random.rand(fn.size)
            yarr = np.random.rand(fn.size) + 1j * np.random.rand(fn.size)

        # Make Fn vectors
        x = fn.element(xarr)
        y = fn.element(yarr)
        return xarr, yarr, x, y

    @staticmethod
    def _sparse_matrix(fn):
        nnz = np.random.randint(0, int(ceil(fn.size**2/2)))
        coo_r = np.random.randint(0, fn.size, size=nnz)
        coo_c = np.random.randint(0, fn.size, size=nnz)
        values = np.random.rand(nnz)
        return sp.sparse.coo_matrix((values, (coo_r, coo_c)),
                                    shape=(fn.size, fn.size))

    @staticmethod
    def _dense_matrix(fn):
        return np.asmatrix(np.random.rand(fn.size, fn.size), dtype=float)

    def test_init(self):
        rn = Rn(10)
        sparse_mat = self._sparse_matrix(rn)
        dense_mat = self._dense_matrix(rn)

        # Just test if the code runs
        inner = MatrixWeightedInner(rn, sparse_mat)
        inner = MatrixWeightedInner(rn, dense_mat)

        nonsquare_mat = np.eye(10, 5)
        with self.assertRaises(ValueError):
            MatrixWeightedInner(rn, nonsquare_mat)

    def _test_equals(self, n):
        rn = Rn(n)
        sparse_mat = self._sparse_matrix(rn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(rn)
        different_dense_mat = dense_mat.copy()
        different_dense_mat[0, 0] = -10

        inner_sparse = MatrixWeightedInner(rn, sparse_mat)
        inner_sparse2 = MatrixWeightedInner(rn, sparse_mat)
        inner_sparse_as_dense = MatrixWeightedInner(rn, sparse_mat_as_dense)
        inner_dense = MatrixWeightedInner(rn, dense_mat)
        inner_dense2 = MatrixWeightedInner(rn, dense_mat)
        inner_different_dense = MatrixWeightedInner(rn, different_dense_mat)

        # Identical objects -> True
        self.assertEquals(inner_sparse, inner_sparse)
        # Identical matrices -> True
        self.assertEquals(inner_sparse, inner_sparse2)
        self.assertEquals(inner_dense, inner_dense2)
        # Equivalent but not identical matrices -> False
        self.assertNotEquals(inner_sparse, inner_sparse_as_dense)
        self.assertNotEquals(inner_sparse_as_dense, inner_sparse)
        # Not equivalent -> False
        self.assertNotEquals(inner_dense, inner_different_dense)

    def test_equals(self):
        for n in range(1, 20):
            self._test_equals(n)

    def _test_equiv(self, n):
        rn = Rn(n)
        sparse_mat = self._sparse_matrix(rn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(rn)
        different_dense_mat = dense_mat.copy()
        different_dense_mat[0, 0] = -10

        inner_sparse = MatrixWeightedInner(rn, sparse_mat)
        inner_sparse2 = MatrixWeightedInner(rn, sparse_mat)
        inner_sparse_as_dense = MatrixWeightedInner(rn, sparse_mat_as_dense)
        inner_dense = MatrixWeightedInner(rn, dense_mat)
        inner_dense_copy = MatrixWeightedInner(rn, dense_mat.copy())
        inner_different_dense = MatrixWeightedInner(rn, different_dense_mat)

        # Equal -> True
        self.assertTrue(inner_sparse.equiv(inner_sparse))
        self.assertTrue(inner_sparse.equiv(inner_sparse2))
        # Equivalent matrices -> True
        self.assertTrue(inner_sparse.equiv(inner_sparse_as_dense))
        self.assertTrue(inner_dense.equiv(inner_dense_copy))
        # Different matrices -> False
        self.assertFalse(inner_dense.equiv(inner_different_dense))

    def _test_call_real(self, n):
        rn = Rn(n)
        xarr, yarr, x, y = self._vectors(rn)
        sparse_mat = self._sparse_matrix(rn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(rn)

        inner_sparse = MatrixWeightedInner(rn, sparse_mat)
        inner_dense = MatrixWeightedInner(rn, dense_mat)

        result_sparse = inner_sparse(x, y)
        result_dense = inner_dense(x, y)

        true_result_sparse = np.dot(
            yarr, np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze())
        true_result_dense = np.dot(
            yarr, np.asarray(np.dot(dense_mat, xarr)).squeeze())

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def _test_call_complex(self, n):
        cn = Cn(n)
        xarr, yarr, x, y = self._vectors(cn)
        sparse_mat = self._sparse_matrix(cn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(cn)

        inner_sparse = MatrixWeightedInner(cn, sparse_mat)
        inner_dense = MatrixWeightedInner(cn, dense_mat)

        result_sparse = inner_sparse(x, y)
        result_dense = inner_dense(x, y)

        true_result_sparse = np.dot(
            yarr.conj(),
            np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze())
        true_result_dense = np.dot(
            yarr.conj(), np.asarray(np.dot(dense_mat, xarr)).squeeze())

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def test_call(self):
        for _ in range(20):
            self._test_call_real(10)
            self._test_call_complex(10)

    def test_repr(self):
        n = 5
        rn = Rn(n)
        sparse_mat = sp.sparse.dia_matrix((np.arange(n, dtype=float), [0]),
                                          shape=(n, n))
        dense_mat = sparse_mat.todense()

        inner_sparse = MatrixWeightedInner(rn, sparse_mat)
        inner_dense = MatrixWeightedInner(rn, dense_mat)

        mat_str_sparse = ("<(5, 5) sparse matrix, format 'dia', "
                          "5 stored entries>")
        mat_str_dense = dedent('''
        matrix([[ 0.,  0.,  0.,  0.,  0.],
                [ 0.,  1.,  0.,  0.,  0.],
                [ 0.,  0.,  2.,  0.,  0.],
                [ 0.,  0.,  0.,  3.,  0.],
                [ 0.,  0.,  0.,  0.,  4.]])
        ''')

        repr_str_sparse = 'MatrixWeightedInner({})'.format(mat_str_sparse)
        repr_str_dense = 'MatrixWeightedInner({})'.format(mat_str_dense)
        self.assertEquals(repr(inner_sparse), repr_str_sparse)
        self.assertEquals(repr(inner_dense), repr_str_dense)

    def test_str(self):
        n = 5
        rn = Rn(n)
        sparse_mat = sp.sparse.dia_matrix((np.arange(n, dtype=float), [0]),
                                          shape=(n, n))
        dense_mat = sparse_mat.todense()

        inner_sparse = MatrixWeightedInner(rn, sparse_mat)
        inner_dense = MatrixWeightedInner(rn, dense_mat)

        mat_str_sparse = '''
  (1, 1)\t1.0
  (2, 2)\t2.0
  (3, 3)\t3.0
  (4, 4)\t4.0'''
        mat_str_dense = dedent('''
        [[ 0.  0.  0.  0.  0.]
         [ 0.  1.  0.  0.  0.]
         [ 0.  0.  2.  0.  0.]
         [ 0.  0.  0.  3.  0.]
         [ 0.  0.  0.  0.  4.]]''')

        print_str_sparse = ('(x, y) --> y^H G x,  G ={}'
                            ''.format(mat_str_sparse))
        self.assertEquals(str(inner_sparse), print_str_sparse)

        print_str_dense = ('(x, y) --> y^H G x,  G ={}'
                           ''.format(mat_str_dense))
        self.assertEquals(str(inner_dense), print_str_dense)


class ConstWeightedInnerTest(ODLTestCase):
    @staticmethod
    def _vectors(fn):
        # Generate numpy vectors, real or complex
        if isinstance(fn, Rn):
            xarr = np.random.rand(fn.size)
            yarr = np.random.rand(fn.size)
        else:
            xarr = np.random.rand(fn.size) + 1j * np.random.rand(fn.size)
            yarr = np.random.rand(fn.size) + 1j * np.random.rand(fn.size)

        # Make Fn vectors
        x = fn.element(xarr)
        y = fn.element(yarr)
        return xarr, yarr, x, y

    def test_init(self):
        n = 10
        rn = Rn(n)
        constant = 1.5

        # Just test if the code runs
        inner = ConstWeightedInner(rn, constant)

    def test_equals(self):
        n = 10
        rn = Rn(n)
        constant = 1.5

        inner_const = ConstWeightedInner(rn, constant)
        inner_const2 = ConstWeightedInner(rn, constant)

        const_sparse_mat = sp.sparse.dia_matrix(([constant]*n, [0]),
                                                shape=(n, n))
        const_dense_mat = constant * np.eye(n)
        inner_matrix_sp = MatrixWeightedInner(rn, const_sparse_mat)
        inner_matrix_de = MatrixWeightedInner(rn, const_dense_mat)

        self.assertEquals(inner_const, inner_const)
        self.assertEquals(inner_const, inner_const2)
        self.assertEquals(inner_const2, inner_const)
        # Equivalent but not equal -> False
        self.assertNotEquals(inner_const, inner_matrix_sp)
        self.assertNotEquals(inner_const, inner_matrix_de)

        rn_single = Rn(n, dtype='single')
        inner_const_single = ConstWeightedInner(rn_single, constant)
        self.assertNotEquals(inner_const, inner_const_single)

        inner_different_const = ConstWeightedInner(rn, 2.5)
        self.assertNotEquals(inner_const, inner_different_const)

    def test_equiv(self):
        n = 10
        rn = Rn(n)
        constant = 1.5

        inner_const = ConstWeightedInner(rn, constant)
        inner_const2 = ConstWeightedInner(rn, constant)

        const_sparse_mat = sp.sparse.dia_matrix(([constant]*n, [0]),
                                                shape=(n, n))
        const_dense_mat = constant * np.eye(n)
        inner_matrix_sp = MatrixWeightedInner(rn, const_sparse_mat)
        inner_matrix_de = MatrixWeightedInner(rn, const_dense_mat)

        # Equal -> True
        self.assertTrue(inner_const.equiv(inner_const))
        self.assertTrue(inner_const.equiv(inner_const2))
        # Equivalent matrix representation -> True
        self.assertTrue(inner_const.equiv(inner_matrix_sp))
        self.assertTrue(inner_const.equiv(inner_matrix_de))

        rn_single = Rn(n, dtype='single')
        inner_const_single = ConstWeightedInner(rn_single, constant)
        self.assertFalse(inner_const.equiv(inner_const_single))

        inner_different_const = ConstWeightedInner(rn, 2.5)
        self.assertFalse(inner_const.equiv(inner_different_const))

    def _test_call_real(self, n):
        rn = Rn(n)
        xarr, yarr, x, y = self._vectors(rn)

        constant = 1.5
        inner_const = ConstWeightedInner(rn, constant)

        result_const = inner_const(x, y)
        true_result_const = constant * np.dot(yarr, xarr)

        self.assertAlmostEquals(result_const, true_result_const)

    def _test_call_complex(self, n):
        cn = Cn(n)
        xarr, yarr, x, y = self._vectors(cn)

        constant = 1.5
        inner_const = ConstWeightedInner(cn, constant)

        result_const = inner_const(x, y)
        true_result_const = constant * np.dot(yarr.conj(), xarr)

        self.assertAlmostEquals(result_const, true_result_const)

    def test_call(self):
        for n in range(1, 20):
            self._test_call_real(n)
            self._test_call_complex(n)

    def test_repr(self):
        n = 10
        rn = Rn(n)
        constant = 1.5
        inner_const = ConstWeightedInner(rn, constant)

        repr_str = 'ConstWeightedInner(Rn(10), 1.5)'
        self.assertEquals(repr(inner_const), repr_str)

    def test_str(self):
        n = 10
        rn = Rn(n)
        constant = 1.5
        inner_const = ConstWeightedInner(rn, constant)

        print_str = '(x, y) --> 1.5 * y^H x'
        self.assertEquals(str(inner_const), print_str)


if __name__ == '__main__':
    unittest.main(exit=False)
