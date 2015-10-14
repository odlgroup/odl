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
from odl import Rn, Cn
from odl.space.ntuples import _FnConstWeighting, _FnMatrixWeighting
from odl.util.testutils import ODLTestCase

# TODO: add tests for:
# * Ntuples (different data types)
# * metric, normed, Hilbert space variants
# * Cn
# * Rn, Cn with non-standard data types
# * vector multiplication
# * MatVecOperator
# * Custom inner/norm/dist


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

class NumpyInteractionTest(ODLTestCase):
    def test_multiply_by_scalar(self):
        """Verifies that multiplying with numpy scalars
        does not change the type of the array
        """

        r3 = Rn(3)
        x = r3.zero()
        self.assertIn(x * 1.0, r3)
        self.assertIn(x * np.float32(1.0), r3)
        self.assertIn(1.0 * x, r3)
        self.assertIn(np.float32(1.0) * x, r3)

    def test_array_method(self):
        """ Verifies that the __array__ method works
        """
        r3 = Rn(3)
        x = r3.zero()

        arr = x.__array__()

        self.assertIsInstance(arr, np.ndarray)
        self.assertAllAlmostEquals(arr, [0.0, 0.0, 0.0])

    def test_array_wrap_method(self):
        """ Verifies that the __array_wrap__ method works
        This enables us to use numpy ufuncs on vectors
        """
        r3 = Rn(3)
        x_h = [0.0, 1.0, 2.0]
        x = r3.element([0.0, 1.0, 2.0])
        y_h = np.sin(x_h)
        y = np.sin(x)

        self.assertAllAlmostEquals(y, y_h)
        self.assertIn(y, r3)

        #Test with a non-standard dtype
        r3_f32 = Rn(3, dtype=np.float32)
        x_h = [0.0, 1.0, 2.0]
        x = r3_f32.element(x_h)
        y_h = np.sin(x_h)
        y = np.sin(x)

        self.assertAllAlmostEquals(y, y_h)
        self.assertIn(y, r3_f32)



class FnMatrixWeightingTest(ODLTestCase):
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
        mat = sp.sparse.coo_matrix((values, (coo_r, coo_c)),
                                   shape=(fn.size, fn.size))
        # Make symmetric and positive definite
        return mat + mat.T + fn.size * sp.sparse.eye(fn.size)

    @staticmethod
    def _dense_matrix(fn):
        mat = np.asmatrix(np.random.rand(fn.size, fn.size), dtype=float)
        # Make symmetric and positive definite
        return mat + mat.T + fn.size * np.eye(fn.size)

    def test_init(self):
        rn = Rn(10)
        sparse_mat = self._sparse_matrix(rn)
        dense_mat = self._dense_matrix(rn)

        # Just test if the code runs
        weighting = _FnMatrixWeighting(sparse_mat)
        weighting = _FnMatrixWeighting(dense_mat)

        nonsquare_mat = np.eye(10, 5)
        with self.assertRaises(ValueError):
            _FnMatrixWeighting(nonsquare_mat)

    def _test_equals(self, n):
        rn = Rn(n)
        sparse_mat = self._sparse_matrix(rn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(rn)
        different_dense_mat = dense_mat.copy()
        different_dense_mat[0, 0] = -10

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_sparse2 = _FnMatrixWeighting(sparse_mat)
        w_sparse_as_dense = _FnMatrixWeighting(sparse_mat_as_dense)
        w_dense = _FnMatrixWeighting(dense_mat)
        w_dense2 = _FnMatrixWeighting(dense_mat)
        w_different_dense = _FnMatrixWeighting(different_dense_mat)

        # Identical objects -> True
        self.assertEquals(w_sparse, w_sparse)
        # Identical matrices -> True
        self.assertEquals(w_sparse, w_sparse2)
        self.assertEquals(w_dense, w_dense2)
        # Equivalent but not identical matrices -> False
        self.assertNotEquals(w_sparse, w_sparse_as_dense)
        self.assertNotEquals(w_sparse_as_dense, w_sparse)
        # Not equivalent -> False
        self.assertNotEquals(w_dense, w_different_dense)

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

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_sparse2 = _FnMatrixWeighting(sparse_mat)
        w_sparse_as_dense = _FnMatrixWeighting(
            rn, sparse_mat_as_dense)
        w_dense = _FnMatrixWeighting(dense_mat)
        w_dense_copy = _FnMatrixWeighting(dense_mat.copy())
        w_different_dense = _FnMatrixWeighting(
            rn, different_dense_mat)

        # Equal -> True
        self.assertTrue(w_sparse.equiv(w_sparse))
        self.assertTrue(w_sparse.equiv(w_sparse2))
        # Equivalent matrices -> True
        self.assertTrue(w_sparse.equiv(w_sparse_as_dense))
        self.assertTrue(w_dense.equiv(w_dense_copy))
        # Different matrices -> False
        self.assertFalse(w_dense.equiv(w_different_dense))

    def _test_inner_real(self, n):
        rn = Rn(n)
        xarr, yarr, x, y = self._vectors(rn)
        sparse_mat = self._sparse_matrix(rn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(rn)

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_dense = _FnMatrixWeighting(dense_mat)

        result_sparse = w_sparse.inner(x, y)
        result_dense = w_dense.inner(x, y)

        true_result_sparse = np.dot(
            yarr, np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze())
        true_result_dense = np.dot(
            yarr, np.asarray(np.dot(dense_mat, xarr)).squeeze())

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def _test_norm_real(self, n):
        rn = Rn(n)
        xarr, yarr, x, y = self._vectors(rn)
        sparse_mat = self._sparse_matrix(rn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(rn)

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_dense = _FnMatrixWeighting(dense_mat)

        result_sparse = w_sparse.norm(x)
        result_dense = w_dense.norm(x)

        true_result_sparse = np.sqrt(np.dot(
            xarr, np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze()))
        true_result_dense = np.sqrt(np.dot(
            xarr, np.asarray(np.dot(dense_mat, xarr)).squeeze()))

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def _test_dist_real(self, n):
        rn = Rn(n)
        xarr, yarr, x, y = self._vectors(rn)
        sparse_mat = self._sparse_matrix(rn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(rn)

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_dense = _FnMatrixWeighting(dense_mat)

        result_sparse = w_sparse.dist(x, y)
        result_dense = w_dense.dist(x, y)

        true_result_sparse = np.sqrt(np.dot(
            xarr-yarr,
            np.asarray(np.dot(sparse_mat_as_dense, xarr-yarr)).squeeze()))
        true_result_dense = np.sqrt(np.dot(
            xarr-yarr, np.asarray(np.dot(dense_mat, xarr-yarr)).squeeze()))

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def _test_inner_complex(self, n):
        cn = Cn(n)
        xarr, yarr, x, y = self._vectors(cn)
        sparse_mat = self._sparse_matrix(cn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(cn)

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_dense = _FnMatrixWeighting(dense_mat)

        result_sparse = w_sparse.inner(x, y)
        result_dense = w_dense.inner(x, y)

        true_result_sparse = np.vdot(
            yarr,
            np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze())
        true_result_dense = np.vdot(
            yarr, np.asarray(np.dot(dense_mat, xarr)).squeeze())

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def _test_norm_complex(self, n):
        cn = Cn(n)
        xarr, yarr, x, y = self._vectors(cn)
        sparse_mat = self._sparse_matrix(cn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(cn)

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_dense = _FnMatrixWeighting(dense_mat)

        result_sparse = w_sparse.norm(x)
        result_dense = w_dense.norm(x)

        true_result_sparse = np.sqrt(np.vdot(
            xarr,
            np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze()))
        true_result_dense = np.sqrt(np.vdot(
            xarr, np.asarray(np.dot(dense_mat, xarr)).squeeze()))

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def _test_dist_complex(self, n):
        cn = Cn(n)
        xarr, yarr, x, y = self._vectors(cn)
        sparse_mat = self._sparse_matrix(cn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(cn)

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_dense = _FnMatrixWeighting(dense_mat)

        result_sparse = w_sparse.dist(x, y)
        result_dense = w_dense.dist(x, y)

        true_result_sparse = np.sqrt(np.vdot(
            xarr-yarr,
            np.asarray(np.dot(sparse_mat_as_dense, xarr-yarr)).squeeze()))
        true_result_dense = np.sqrt(np.vdot(
            xarr-yarr,
            np.asarray(np.dot(dense_mat, xarr-yarr)).squeeze()))

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def test_methods(self):
        for n in range(2, 20):
            self._test_inner_real(n)
            self._test_norm_real(n)
            self._test_dist_real(n)
            self._test_inner_complex(n)
            self._test_norm_complex(n)
            self._test_dist_complex(n)

    def test_repr(self):
        n = 5
        sparse_mat = sp.sparse.dia_matrix((np.arange(n, dtype=float), [0]),
                                          shape=(n, n))
        dense_mat = sparse_mat.todense()

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_dense = _FnMatrixWeighting(dense_mat)

        mat_str_sparse = ("<(5, 5) sparse matrix, format 'dia', "
                          "5 stored entries>")
        mat_str_dense = dedent('''
        matrix([[ 0.,  0.,  0.,  0.,  0.],
                [ 0.,  1.,  0.,  0.,  0.],
                [ 0.,  0.,  2.,  0.,  0.],
                [ 0.,  0.,  0.,  3.,  0.],
                [ 0.,  0.,  0.,  0.,  4.]])
        ''')

        repr_str_sparse = ('_FnMatrixWeighting({})'
                           ''.format(mat_str_sparse))
        repr_str_dense = '_FnMatrixWeighting({})'.format(mat_str_dense)
        self.assertEquals(repr(w_sparse), repr_str_sparse)
        self.assertEquals(repr(w_dense), repr_str_dense)

    def test_str(self):
        n = 5
        sparse_mat = sp.sparse.dia_matrix((np.arange(n, dtype=float), [0]),
                                          shape=(n, n))
        dense_mat = sparse_mat.todense()

        w_sparse = _FnMatrixWeighting(sparse_mat)
        w_dense = _FnMatrixWeighting(dense_mat)

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

        print_str_sparse = ('Weighting: matrix ={}'
                            ''.format(mat_str_sparse))
        self.assertEquals(str(w_sparse), print_str_sparse)

        print_str_dense = ('Weighting: matrix ={}'
                           ''.format(mat_str_dense))
        self.assertEquals(str(w_dense), print_str_dense)


class FnConstWeightingTest(ODLTestCase):
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
        constant = 1.5

        # Just test if the code runs
        weighting = _FnConstWeighting(constant)

    def test_equals(self):
        n = 10
        constant = 1.5

        w_const = _FnConstWeighting(constant)
        w_const2 = _FnConstWeighting(constant)

        const_sparse_mat = sp.sparse.dia_matrix(([constant]*n, [0]),
                                                shape=(n, n))
        const_dense_mat = constant * np.eye(n)
        w_matrix_sp = _FnMatrixWeighting(const_sparse_mat)
        w_matrix_de = _FnMatrixWeighting(const_dense_mat)

        self.assertEquals(w_const, w_const)
        self.assertEquals(w_const, w_const2)
        self.assertEquals(w_const2, w_const)
        # Equivalent but not equal -> False
        self.assertNotEquals(w_const, w_matrix_sp)
        self.assertNotEquals(w_const, w_matrix_de)

        w_different_const = _FnConstWeighting(2.5)
        self.assertNotEquals(w_const, w_different_const)

    def test_equiv(self):
        n = 10
        constant = 1.5

        w_const = _FnConstWeighting(constant)
        w_const2 = _FnConstWeighting(constant)

        const_sparse_mat = sp.sparse.dia_matrix(([constant]*n, [0]),
                                                shape=(n, n))
        const_dense_mat = constant * np.eye(n)
        w_matrix_sp = _FnMatrixWeighting(const_sparse_mat)
        w_matrix_de = _FnMatrixWeighting(const_dense_mat)

        # Equal -> True
        self.assertTrue(w_const.equiv(w_const))
        self.assertTrue(w_const.equiv(w_const2))
        # Equivalent matrix representation -> True
        self.assertTrue(w_const.equiv(w_matrix_sp))
        self.assertTrue(w_const.equiv(w_matrix_de))

        w_different_const = _FnConstWeighting(2.5)
        self.assertFalse(w_const.equiv(w_different_const))

    def _test_inner_real(self, n):
        rn = Rn(n)
        xarr, yarr, x, y = self._vectors(rn)

        constant = 1.5
        w_const = _FnConstWeighting(constant)

        result_const = w_const.inner(x, y)
        true_result_const = constant * np.dot(yarr, xarr)

        self.assertAlmostEquals(result_const, true_result_const)

    def _test_norm_real(self, n):
        rn = Rn(n)
        xarr, yarr, x, y = self._vectors(rn)

        constant = 1.5
        w_const = _FnConstWeighting(constant)

        result_const = w_const.norm(x)
        true_result_const = np.sqrt(constant * np.dot(xarr, xarr))

        self.assertAlmostEquals(result_const, true_result_const)

    def _test_dist_real(self, n):
        rn = Rn(n)
        xarr, yarr, x, y = self._vectors(rn)

        constant = 1.5
        w_const = _FnConstWeighting(constant)

        result_const = w_const.dist(x, y)
        true_result_const = np.sqrt(constant * np.dot(xarr-yarr, xarr-yarr))

        self.assertAlmostEquals(result_const, true_result_const)

    def _test_inner_complex(self, n):
        cn = Cn(n)
        xarr, yarr, x, y = self._vectors(cn)

        constant = 1.5
        w_const = _FnConstWeighting(constant)

        result_const = w_const.inner(x, y)
        true_result_const = constant * np.vdot(yarr, xarr)

        self.assertAlmostEquals(result_const, true_result_const)

    def _test_norm_complex(self, n):
        cn = Cn(n)
        xarr, yarr, x, y = self._vectors(cn)

        constant = 1.5
        w_const = _FnConstWeighting(constant)

        result_const = w_const.norm(x)
        true_result_const = np.sqrt(constant * np.vdot(xarr, xarr))

        self.assertAlmostEquals(result_const, true_result_const)

    def _test_dist_complex(self, n):
        cn = Cn(n)
        xarr, yarr, x, y = self._vectors(cn)

        constant = 1.5
        w_const = _FnConstWeighting(constant)

        result_const = w_const.dist(x, y)
        true_result_const = np.sqrt(constant * np.vdot(xarr-yarr, xarr-yarr))

        self.assertAlmostEquals(result_const, true_result_const)

    def test_call(self):
        for n in range(2, 20):
            self._test_inner_real(n)
            self._test_norm_real(n)
            self._test_dist_real(n)
            self._test_inner_complex(n)
            self._test_norm_complex(n)
            self._test_dist_complex(n)

    def test_repr(self):
        constant = 1.5
        w_const = _FnConstWeighting(constant)

        repr_str = '_FnConstWeighting(1.5)'
        self.assertEquals(repr(w_const), repr_str)

    def test_str(self):
        constant = 1.5
        w_const = _FnConstWeighting(constant)

        print_str = 'Weighting: const = 1.5'
        self.assertEquals(str(w_const), print_str)


if __name__ == '__main__':
    unittest.main(exit=False)
