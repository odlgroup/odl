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
from builtins import super
standard_library.install_aliases()

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
from odl import (Operator, OperatorSum, OperatorComp,
                 OperatorLeftScalarMult, OperatorRightScalarMult)
from odl.util.testutils import almost_equal, all_almost_equal


class MultiplyAndSquareOp(Operator):
    """ Example of a nonlinear operator, Calculates (A*x)**2
    """

    def __init__(self, matrix, domain=None, range=None):
        dom = (odl.Rn(matrix.shape[1])
               if domain is None else domain)
        ran = (odl.Rn(matrix.shape[0])
               if range is None else range)

        super().__init__(dom, ran)
        self.matrix = matrix

    def _apply(self, rhs, out):
        np.dot(self.matrix, rhs.data, out=out.data)
        out.data[:] **= 2

    def __str__(self):
        return "MaS: " + str(self.matrix) + "**2"


def mult_sq_np(A, x):
    # The same as MultiplyAndSquareOp but only using numpy
    return np.dot(A, x)**2


def test_nonlinear_op():
    # Verify that the operator does indeed work as expected
    A = np.random.rand(4, 3)
    x = np.random.rand(3)
    Aop = MultiplyAndSquareOp(A)
    xvec = Aop.domain.element(x)

    assert all_almost_equal(Aop(xvec), mult_sq_np(A, x))

def test_nonlinear_addition():
    # Test operator addition
    A = np.random.rand(4, 3)
    B = np.random.rand(4, 3)
    x = np.random.rand(3)

    Aop = MultiplyAndSquareOp(A)
    Bop = MultiplyAndSquareOp(B)
    xvec = Aop.domain.element(x)

    # Explicit instantiation
    C = OperatorSum(Aop, Bop)
    
    assert not C.is_linear
    
    assert all_almost_equal(C(xvec),
                               mult_sq_np(A, x) + mult_sq_np(B, x))

    # Using operator overloading
    assert all_almost_equal((Aop + Bop)(xvec),
                               mult_sq_np(A, x) + mult_sq_np(B, x))

    # Verify that unmatched operators domains fail
    C = np.random.rand(4, 4)
    Cop = MultiplyAndSquareOp(C)

    with pytest.raises(TypeError):
        C = OperatorSum(Aop, Cop)

def test_nonlinear_scale():
    A = np.random.rand(4, 3)
    x = np.random.rand(3)

    Aop = MultiplyAndSquareOp(A)
    xvec = Aop.domain.element(x)

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1)).
    scalars = [-1.432, -1, 0, 1, 3.14]
    for scale in scalars:
        lscaled = OperatorLeftScalarMult(Aop, scale)
        rscaled = OperatorRightScalarMult(Aop, scale)
        
        assert not lscaled.is_linear
        assert not rscaled.is_linear

        assert all_almost_equal(lscaled(xvec),
                                   scale * mult_sq_np(A, x))
        assert all_almost_equal(rscaled(xvec),
                                   mult_sq_np(A, scale*x))

        # Using operator overloading
        assert all_almost_equal((scale * Aop)(xvec),
                                   scale * mult_sq_np(A, x))
        assert all_almost_equal((Aop * scale)(xvec),
                                   mult_sq_np(A, scale*x))

    # Fail when scaling by wrong scalar type (A complex number)
    nonscalars = [1j, [1, 2]]
    for nonscalar in nonscalars:
        with pytest.raises(TypeError):
            OperatorLeftScalarMult(Aop, nonscalar)

        with pytest.raises(TypeError):
            OperatorRightScalarMult(Aop, nonscalar)

        with pytest.raises(TypeError):
            Aop * nonscalar

        with pytest.raises(TypeError):
            nonscalar * Aop

def test_composition():
    A = np.random.rand(5, 4)
    B = np.random.rand(4, 3)
    x = np.random.rand(3)

    Aop = MultiplyAndSquareOp(A)
    Bop = MultiplyAndSquareOp(B)
    xvec = Bop.domain.element(x)

    C = OperatorComp(Aop, Bop)
    
    assert not C.is_linear

    assert all_almost_equal(C(xvec), mult_sq_np(A, mult_sq_np(B, x)))

    # Verify that incorrect order fails
    with pytest.raises(TypeError):
        C = OperatorComp(Bop, Aop)


class MultiplyOp(Operator):
    """Multiply with matrix.
    """

    def __init__(self, matrix, domain=None, range=None):
        domain = (odl.Rn(matrix.shape[1])
                        if domain is None else domain)
        range = (odl.Rn(matrix.shape[0])
                       if range is None else range)
        self.matrix = matrix

        super().__init__(domain, range, linear=True)

    def _apply(self, rhs, out):
        np.dot(self.matrix, rhs.data, out=out.data)

    @property
    def adjoint(self):
        return MultiplyOp(self.matrix.T, self.range, self.domain)


def test_linear_Op():
    # Verify that the multiply op does indeed work as expected

    A = np.random.rand(3, 3)
    x = np.random.rand(3)
    out = np.random.rand(3)

    Aop = MultiplyOp(A)
    xvec = Aop.domain.element(x)
    outvec = Aop.range.element()

    # Using out parameter
    Aop(xvec, outvec)
    np.dot(A, x, out)
    assert all_almost_equal(out, outvec)

    # Using return value
    assert all_almost_equal(Aop(xvec), np.dot(A, x))

def test_linear_op_nonsquare():
    # Verify that the multiply op does indeed work as expected
    A = np.random.rand(4, 3)
    x = np.random.rand(3)
    out = np.random.rand(4)

    Aop = MultiplyOp(A)
    xvec = Aop.domain.element(x)
    outvec = Aop.range.element()

    # Using out parameter
    Aop(xvec, outvec)
    np.dot(A, x, out)
    assert all_almost_equal(out, outvec)

    # Using return value
    assert all_almost_equal(Aop(xvec), np.dot(A, x))

def test_linear_adjoint():
    A = np.random.rand(4, 3)
    x = np.random.rand(4)
    out = np.random.rand(3)

    Aop = MultiplyOp(A)
    xvec = Aop.range.element(x)
    outvec = Aop.domain.element()

    # Using adjoint
    Aop.adjoint(xvec, outvec)
    np.dot(A.T, x, out)
    assert all_almost_equal(out, outvec)

    # Using T method
    assert all_almost_equal(Aop.T(xvec), np.dot(A.T, x))

def test_linear_addition():
    A = np.random.rand(4, 3)
    B = np.random.rand(4, 3)
    x = np.random.rand(3)
    y = np.random.rand(4)

    Aop = MultiplyOp(A)
    Bop = MultiplyOp(B)
    xvec = Aop.domain.element(x)
    yvec = Aop.range.element(y)

    # Explicit instantiation
    C = OperatorSum(Aop, Bop)

    assert C.is_linear

    assert all_almost_equal(C(xvec), np.dot(A, x) + np.dot(B, x))
    assert all_almost_equal(C.T(yvec), np.dot(A.T, y) + np.dot(B.T, y))

    # Using operator overloading
    assert all_almost_equal((Aop + Bop)(xvec),
                               np.dot(A, x) + np.dot(B, x))
    assert all_almost_equal((Aop + Bop).T(yvec),
                               np.dot(A.T, y) + np.dot(B.T, y))

def test_linear_scale():
    A = np.random.rand(4, 3)
    x = np.random.rand(3)
    y = np.random.rand(4)

    Aop = MultiplyOp(A)
    xvec = Aop.domain.element(x)
    yvec = Aop.range.element(y)

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    scalars = [-1.432, -1, 0, 1, 3.14]
    for scale in scalars:
        C = OperatorRightScalarMult(Aop, scale)
        
        assert C.is_linear

        assert all_almost_equal(C(xvec), scale * np.dot(A, x))
        assert all_almost_equal(C.T(yvec), scale * np.dot(A.T, y))

        # Using operator overloading
        assert all_almost_equal((scale * Aop)(xvec),
                                   scale * np.dot(A, x))
        assert all_almost_equal((Aop * scale)(xvec),
                                   np.dot(A, scale * x))
        assert all_almost_equal((scale * Aop).T(yvec),
                                   scale * np.dot(A.T, y))
        assert all_almost_equal((Aop * scale).T(yvec),
                                   np.dot(A.T, scale * y))

def test_linear_composition():
    A = np.random.rand(5, 4)
    B = np.random.rand(4, 3)
    x = np.random.rand(3)
    y = np.random.rand(5)

    Aop = MultiplyOp(A)
    Bop = MultiplyOp(B)
    xvec = Bop.domain.element(x)
    yvec = Aop.range.element(y)

    C = OperatorComp(Aop, Bop)

    assert C.is_linear

    assert all_almost_equal(C(xvec), np.dot(A, np.dot(B, x)))
    assert all_almost_equal(C.T(yvec), np.dot(B.T, np.dot(A.T, y)))

def test_type_errors():
    r3 = odl.Rn(3)
    r4 = odl.Rn(4)

    Aop = MultiplyOp(np.random.rand(3, 3))
    r3Vec1 = r3.zero()
    r3Vec2 = r3.zero()
    r4Vec1 = r4.zero()
    r4Vec2 = r4.zero()

    # Verify that correct usage works
    Aop(r3Vec1, r3Vec2)
    Aop.adjoint(r3Vec1, r3Vec2)

    # Test that erroneous usage raises TypeError
    with pytest.raises(TypeError):
        Aop(r4Vec1)

    with pytest.raises(TypeError):
        Aop.T(r4Vec1)

    with pytest.raises(TypeError):
        Aop(r3Vec1, r4Vec1)

    with pytest.raises(TypeError):
        Aop.adjoint(r3Vec1, r4Vec1)

    with pytest.raises(TypeError):
        Aop(r4Vec1, r3Vec1)

    with pytest.raises(TypeError):
        Aop.adjoint(r4Vec1, r3Vec1)

    with pytest.raises(TypeError):
        Aop(r4Vec1, r4Vec2)

    with pytest.raises(TypeError):
        Aop.adjoint(r4Vec1, r4Vec2)

def test_test_left_vector_mult():
    r3 = odl.Rn(3)
    Aop = MultiplyOp(np.random.rand(3, 3))

    x = r3.element([1, 2, 3])
if __name__ == '__main__':
    pytest.main(str(__file__))
