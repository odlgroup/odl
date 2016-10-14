# Copyright 2014-2016 The ODL development group
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
from builtins import super

# External module imports
import pytest
import numpy as np
import sys

# ODL imports
import odl
from odl import (Operator, OperatorSum, OperatorComp,
                 OperatorLeftScalarMult, OperatorRightScalarMult,
                 FunctionalLeftVectorMult, OperatorRightVectorMult,
                 MatVecOperator, OperatorLeftVectorMult,
                 OpTypeError, OpDomainError, OpRangeError)
from odl.operator.operator import _signature_from_spec, _dispatch_call_args
from odl.util.testutils import almost_equal, all_almost_equal


class MultiplyAndSquareOp(Operator):
    """Example of a nonlinear operator, x --> (A*x)**2."""

    def __init__(self, matrix, domain=None, range=None):
        dom = (odl.rn(matrix.shape[1])
               if domain is None else domain)
        ran = (odl.rn(matrix.shape[0])
               if range is None else range)

        super().__init__(dom, ran)
        self.matrix = matrix

    def _call(self, rhs, out=None):
        if out is None:
            return np.dot(self.matrix, rhs.data) ** 2
        else:
            np.dot(self.matrix, rhs.data, out=out.data)
            out **= 2

    def __str__(self):
        return "MaS: " + str(self.matrix) + " ** 2"


def mult_sq_np(A, x):
    # The same as MultiplyAndSquareOp but only using numpy
    return np.dot(A, x) ** 2


def test_nonlinear_op():
    # Verify that the operator does indeed work as expected
    A = np.random.rand(4, 3)
    x = np.random.rand(3)
    Aop = MultiplyAndSquareOp(A)
    xvec = Aop.domain.element(x)

    assert all_almost_equal(Aop(xvec), mult_sq_np(A, x))


def evaluate(operator, point, expected):
    """Assert that operator(point) == expected."""

    assert all_almost_equal(operator(point), expected)

    out = operator.range.element()
    operator(point, out=out)

    assert all_almost_equal(out, expected)


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

    evaluate(C, xvec, mult_sq_np(A, x) + mult_sq_np(B, x))

    # Using operator overloading
    evaluate(Aop + Bop, xvec, mult_sq_np(A, x) + mult_sq_np(B, x))

    # Verify that unmatched operators domains fail
    C = np.random.rand(4, 4)
    Cop = MultiplyAndSquareOp(C)

    with pytest.raises(OpTypeError):
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

        # Explicit
        evaluate(lscaled, xvec, scale * mult_sq_np(A, x))
        evaluate(rscaled, xvec, mult_sq_np(A, scale * x))

        # Using operator overloading
        evaluate(scale * Aop, xvec, scale * mult_sq_np(A, x))
        evaluate(Aop * scale, xvec, mult_sq_np(A, scale * x))

    # Fail when scaling by wrong scalar type (A complex number)
    wrongscalars = [1j, [1, 2], (1, 2)]
    for wrongscalar in wrongscalars:
        with pytest.raises(TypeError):
            print(OperatorLeftScalarMult(Aop, wrongscalar))

        with pytest.raises(TypeError):
            print(OperatorRightScalarMult(Aop, wrongscalar))

        with pytest.raises(TypeError):
            print(Aop * wrongscalar)

        with pytest.raises(TypeError):
            print(wrongscalar * Aop)


def test_nonlinear_vector_mult():
    A = np.random.rand(4, 3)

    Aop = MultiplyAndSquareOp(A)
    rvec = Aop.domain.element([1, 2, 3])
    lvec = Aop.range.element([1, 2, 3, 4])
    x = Aop.domain.element([4, 5, 6])

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    rmult = OperatorRightVectorMult(Aop, rvec)
    lmult = OperatorLeftVectorMult(Aop, lvec)

    assert not rmult.is_linear
    assert not lmult.is_linear

    evaluate(rmult, x, mult_sq_np(A, rvec * x))
    evaluate(lmult, x, lvec * mult_sq_np(A, x))

    # Using operator overloading
    evaluate(Aop * rvec, x, mult_sq_np(A, rvec * x))
    evaluate(lvec * Aop, x, lvec * mult_sq_np(A, x))


def test_nonlinear_composition():
    A = np.random.rand(5, 4)
    B = np.random.rand(4, 3)
    x = np.random.rand(3)

    Aop = MultiplyAndSquareOp(A)
    Bop = MultiplyAndSquareOp(B)
    xvec = Bop.domain.element(x)

    C = OperatorComp(Aop, Bop)

    assert not C.is_linear

    evaluate(C, xvec, mult_sq_np(A, mult_sq_np(B, x)))

    # Verify that incorrect order fails
    with pytest.raises(OpTypeError):
        C = OperatorComp(Bop, Aop)


def test_linear_Op():
    # Verify that the multiply op does indeed work as expected

    A = np.random.rand(3, 3)
    x = np.random.rand(3)
    out = np.random.rand(3)

    Aop = MatVecOperator(A)
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

    Aop = MatVecOperator(A)

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

    Aop = MatVecOperator(A)
    xvec = Aop.range.element(x)
    outvec = Aop.domain.element()

    # Using in-place adjoint
    Aop.adjoint(xvec, outvec)
    np.dot(A.T, x, out)
    assert all_almost_equal(out, outvec)

    # Using out-of-place method
    assert all_almost_equal(Aop.adjoint(xvec), np.dot(A.T, x))


def test_linear_addition():
    A = np.random.rand(4, 3)
    B = np.random.rand(4, 3)
    x = np.random.rand(3)
    y = np.random.rand(4)

    Aop = MatVecOperator(A)
    Bop = MatVecOperator(B)
    xvec = Aop.domain.element(x)
    yvec = Aop.range.element(y)

    # Explicit instantiation
    C = OperatorSum(Aop, Bop)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert all_almost_equal(C(xvec), np.dot(A, x) + np.dot(B, x))
    assert all_almost_equal(C.adjoint(yvec), np.dot(A.T, y) + np.dot(B.T, y))

    # Using operator overloading
    assert all_almost_equal((Aop + Bop)(xvec),
                            np.dot(A, x) + np.dot(B, x))
    assert all_almost_equal((Aop + Bop).adjoint(yvec),
                            np.dot(A.T, y) + np.dot(B.T, y))


def test_linear_scale():
    A = np.random.rand(4, 3)
    x = np.random.rand(3)
    y = np.random.rand(4)

    Aop = MatVecOperator(A)
    xvec = Aop.domain.element(x)
    yvec = Aop.range.element(y)

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    scalars = [-1.432, -1, 0, 1, 3.14]
    for scale in scalars:
        C = OperatorRightScalarMult(Aop, scale)

        assert C.is_linear
        assert C.adjoint.is_linear

        assert all_almost_equal(C(xvec), scale * np.dot(A, x))
        assert all_almost_equal(C.adjoint(yvec), scale * np.dot(A.T, y))

        # Using operator overloading
        assert all_almost_equal((scale * Aop)(xvec),
                                scale * np.dot(A, x))
        assert all_almost_equal((Aop * scale)(xvec),
                                np.dot(A, scale * x))
        assert all_almost_equal((scale * Aop).adjoint(yvec),
                                scale * np.dot(A.T, y))
        assert all_almost_equal((Aop * scale).adjoint(yvec),
                                np.dot(A.T, scale * y))


def test_linear_right_vector_mult():
    A = np.random.rand(4, 3)

    Aop = MatVecOperator(A)
    vec = Aop.domain.element([1, 2, 3])
    x = Aop.domain.element([4, 5, 6])
    y = Aop.range.element([5, 6, 7, 8])

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    C = OperatorRightVectorMult(Aop, vec)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert all_almost_equal(C(x), np.dot(A, vec * x))
    assert all_almost_equal(C.adjoint(y), vec * np.dot(A.T, y))
    assert all_almost_equal(C.adjoint.adjoint(x), C(x))

    # Using operator overloading
    assert all_almost_equal((Aop * vec)(x),
                            np.dot(A, vec * x))
    assert all_almost_equal((Aop * vec).adjoint(y),
                            vec * np.dot(A.T, y))


def test_linear_composition():
    A = np.random.rand(5, 4)
    B = np.random.rand(4, 3)
    x = np.random.rand(3)
    y = np.random.rand(5)

    Aop = MatVecOperator(A)
    Bop = MatVecOperator(B)
    xvec = Bop.domain.element(x)
    yvec = Aop.range.element(y)

    C = OperatorComp(Aop, Bop)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert all_almost_equal(C(xvec), np.dot(A, np.dot(B, x)))
    assert all_almost_equal(C.adjoint(yvec), np.dot(B.T, np.dot(A.T, y)))


def test_type_errors():
    r3 = odl.rn(3)
    r4 = odl.rn(4)

    Aop = MatVecOperator(np.random.rand(3, 3))
    r3Vec1 = r3.zero()
    r3Vec2 = r3.zero()
    r4Vec1 = r4.zero()
    r4Vec2 = r4.zero()

    # Verify that correct usage works
    Aop(r3Vec1, r3Vec2)
    Aop.adjoint(r3Vec1, r3Vec2)

    # Test that erroneous usage raises
    with pytest.raises(OpDomainError):
        Aop(r4Vec1)

    with pytest.raises(OpDomainError):
        Aop.adjoint(r4Vec1)

    with pytest.raises(OpRangeError):
        Aop(r3Vec1, r4Vec1)

    with pytest.raises(OpRangeError):
        Aop.adjoint(r3Vec1, r4Vec1)

    with pytest.raises(OpDomainError):
        Aop(r4Vec1, r3Vec1)

    with pytest.raises(OpDomainError):
        Aop.adjoint(r4Vec1, r3Vec1)

    with pytest.raises(OpDomainError):
        Aop(r4Vec1, r4Vec2)

    with pytest.raises(OpDomainError):
        Aop.adjoint(r4Vec1, r4Vec2)


# FUNCTIONAL TEST
class SumFunctional(Operator):

    """Sum of elements."""

    def __init__(self, domain):
        super().__init__(domain, domain.field, linear=True)

    def _call(self, x):
        return np.sum(x)

    @property
    def adjoint(self):
        return ConstantVector(self.domain)


class ConstantVector(Operator):

    """Vector times a scalar."""

    def __init__(self, domain):
        super().__init__(domain.field, domain, linear=True)

    def _call(self, x):
        return self.range.element(np.ones(self.range.size) * x)

    @property
    def adjoint(self):
        return SumFunctional(self.range)


def test_functional():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    op = SumFunctional(r3)

    assert op(x) == 6


def test_functional_out():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    op = SumFunctional(r3)

    # No out parameter allowed with functionals
    out = 0
    with pytest.raises(TypeError):
        op(x, out=out)


def test_functional_adjoint():
    r3 = odl.rn(3)

    op = SumFunctional(r3)

    assert op.adjoint(3) == r3.element([3, 3, 3])

    x = r3.element([1, 2, 3])
    assert op.adjoint.adjoint(x) == op(x)


def test_functional_addition():
    r3 = odl.rn(3)

    Aop = SumFunctional(r3)
    Bop = SumFunctional(r3)
    x = r3.element([1, 2, 3])
    y = 1

    # Explicit instantiation
    C = OperatorSum(Aop, Bop)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert C(x) == 2 * np.sum(x)

    # Test adjoint
    assert all_almost_equal(C.adjoint(y), y * 2 * np.ones(3))
    assert all_almost_equal(C.adjoint.adjoint(x), C(x))

    # Using operator overloading
    assert (Aop + Bop)(x) == 2 * np.sum(x)
    assert all_almost_equal((Aop + Bop).adjoint(y), y * 2 * np.ones(3))


def test_functional_scale():
    r3 = odl.rn(3)

    Aop = SumFunctional(r3)
    x = r3.element([1, 2, 3])
    y = 1

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    scalars = [-1.432, -1, 0, 1, 3.14]
    for scale in scalars:
        C = OperatorRightScalarMult(Aop, scale)

        assert C.is_linear
        assert C.adjoint.is_linear

        assert C(x) == scale * np.sum(x)
        assert all_almost_equal(C.adjoint(y), scale * y * np.ones(3))
        assert all_almost_equal(C.adjoint.adjoint(x), C(x))

        # Using operator overloading
        assert (scale * Aop)(x) == scale * np.sum(x)
        assert (Aop * scale)(x) == scale * np.sum(x)
        assert all_almost_equal((scale * Aop).adjoint(y),
                                scale * y * np.ones(3))
        assert all_almost_equal((Aop * scale).adjoint(y),
                                scale * y * np.ones(3))


def test_functional_left_vector_mult():
    r3 = odl.rn(3)
    r4 = odl.rn(4)

    Aop = SumFunctional(r3)
    x = r3.element([1, 2, 3])
    y = r4.element([3, 2, 1, 5])

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    C = FunctionalLeftVectorMult(Aop, y)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert all_almost_equal(C(x), y * np.sum(x))
    assert all_almost_equal(C.adjoint(y), y.inner(y) * np.ones(3))
    assert all_almost_equal(C.adjoint.adjoint(x), C(x))

    # Using operator overloading
    assert all_almost_equal((y * Aop)(x),
                            y * np.sum(x))
    assert all_almost_equal((y * Aop).adjoint(y),
                            y.inner(y) * np.ones(3))


def test_functional_right_vector_mult():
    r3 = odl.rn(3)

    Aop = SumFunctional(r3)
    vec = r3.element([1, 2, 3])
    x = r3.element([4, 5, 6])
    y = 2.0

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    C = OperatorRightVectorMult(Aop, vec)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert all_almost_equal(C(x), np.sum(vec * x))
    assert all_almost_equal(C.adjoint(y), vec * y)
    assert all_almost_equal(C.adjoint.adjoint(x), C(x))

    # Using operator overloading
    assert all_almost_equal((Aop * vec)(x),
                            np.sum(vec * x))
    assert all_almost_equal((Aop * vec).adjoint(y),
                            vec * y)


def test_functional_composition():
    r3 = odl.rn(3)

    Aop = SumFunctional(r3)
    Bop = ConstantVector(r3)
    x = r3.element([1, 2, 3])
    y = 1

    C = OperatorComp(Bop, Aop)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert all_almost_equal(C(x), np.sum(x) * np.ones(3))
    assert all_almost_equal(C.adjoint(x), np.sum(x) * np.ones(3))
    assert all_almost_equal(C.adjoint.adjoint(x), C(x))

    # Using operator overloading
    assert (Aop * Bop)(y) == y * 3
    assert (Aop * Bop).adjoint(y) == y * 3
    assert all_almost_equal((Bop * Aop)(x),
                            np.sum(x) * np.ones(3))
    assert all_almost_equal((Bop * Aop).adjoint(x),
                            np.sum(x) * np.ones(3))


class SumSquaredFunctional(Operator):

    """Sum of the squared elements."""

    def __init__(self, domain):
        super().__init__(domain, domain.field, linear=False)

    def _call(self, x):
        return np.sum(x ** 2)


def test_nonlinear_functional():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    op = SumSquaredFunctional(r3)

    assert almost_equal(op(x), np.sum(x ** 2))


def test_nonlinear_functional_out():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    op = SumSquaredFunctional(r3)
    out = op.range.element()

    with pytest.raises(TypeError):
        print(op(x, out=out))


def test_nonlinear_functional_operators():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    A = SumSquaredFunctional(r3)
    B = SumFunctional(r3)

    # Sum
    C = A + B

    assert not C.is_linear
    assert almost_equal(C(x), A(x) + B(x))

    # Minus
    C = A - B

    assert not C.is_linear
    assert almost_equal(C(x), A(x) - B(x))

    # left mul
    C = 2.0 * A

    assert not C.is_linear
    assert almost_equal(C(x), 2.0 * A(x))

    # right mul
    C = A * 2.0

    assert not C.is_linear
    assert almost_equal(C(x), A(x * 2.0))

    # right divide
    C = A / 2.0

    assert not C.is_linear
    assert almost_equal(C(x), A(x / 2.0))


# test functions to dispatch
def f1(x):
    """f1(x)
    False, False
    good
    """


def f2(x, y):
    """f2(x, y)
    False, False
    bad
    """


def f3(x, y, out):
    """f3(x, y, out)
    False, True
    bad
    """


def f4(*args):
    """f4(*args)
    False, False
    bad
    """


def f5(out):
    """f5(out)
    True, False
    bad
    """


def f6(**kwargs):
    """f6(**kwargs)
    False, False
    bad
    """


def f7(*args, **kwargs):
    """f7(*args, **kwargs)
    False, False
    bad
    """


def f8(out=None, param=2):
    """f8(out=None, param=2)
    True, True
    bad
    """


def f9(x, out=None):
    """f9(x, out=None)
    True, True
    good
    """


def f10(x, out=None, param=2):
    """f10(x, out=None, param=2)
    True, True
    bad
    """


def f11(x, out, **kwargs):
    """f11(x, out, **kwargs)
    True, False
    good
    """


def f12(x, y, out=None, param=2, *args, **kwargs):
    """f12(x, y, out=None, param=2, *args, **kwargs)
    True, True
    bad
    """


py3 = sys.version_info.major > 2
if py3:
    exec('''
def f13(x, *, out=None, param=2, **kwargs):
    """f13(x, *, out=None, param=2, **kwargs)
    True, True
    good
    """
''')

    exec('''
def f14(x, *y, out=None, param=2, **kwargs):
    """f14(x, *y, out=None, param=2, **kwargs)
    True, True
    bad
    """
''')


func_params = [eval('f{}'.format(i)) for i in range(1, 13)]
if py3:
    func_params += [eval('f{}'.format(i)) for i in range(13, 15)]
func_ids = [' f{} '.format(i) for i in range(1, 13)]
if py3:
    func_ids += [' f{} '.format(i) for i in range(13, 15)]
func_fixture = pytest.fixture(scope="module", ids=func_ids, params=func_params)


@func_fixture
def func(request):
    return request.param


def test_signature_from_spec(func):

    true_sig = func.__doc__.splitlines()[0].strip()
    sig = _signature_from_spec(func)
    assert true_sig == sig


def test_dispatch_call_args(func):
    import inspect

    py3 = sys.version_info.major > 2
    getspec = inspect.getfullargspec if py3 else inspect.getargspec

    # Unbound functions
    true_has, true_opt = eval(func.__doc__.splitlines()[1].strip())
    good = func.__doc__.splitlines()[2].strip() == 'good'

    if good:
        truespec = getspec(func)
        truespec.args.insert(0, 'self')

        has, opt, spec = _dispatch_call_args(unbound_call=func)

        assert has == true_has
        assert opt == true_opt
        assert spec == truespec
    else:
        with pytest.raises(ValueError):
            _dispatch_call_args(unbound_call=func)


def test_dispatch_call_args_class():

    # Two sneaky classes whose _call method would pass the signature check
    class WithStaticMethod(object):
        @staticmethod
        def _call(x, y, out):
            pass

    class WithClassMethod(object):
        @classmethod
        def _call(cls, x, out=None):
            pass

    with pytest.raises(TypeError):
        _dispatch_call_args(cls=WithStaticMethod)

    with pytest.raises(TypeError):
        _dispatch_call_args(cls=WithClassMethod)

if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
