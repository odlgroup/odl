# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import numpy as np
from past.builtins import basestring
import pytest

import odl
from odl import vector
from odl.space.npy_tensors import NumpyTensor
from odl.space.space_utils import auto_weighting
from odl.util.testutils import all_equal, simple_fixture, noise_element


auto_weighting_optimize = simple_fixture('optimize', [True, False])
call_variant = simple_fixture('call_variant', ['oop', 'ip', 'dual'])
weighting = simple_fixture('weighting', [1.0, 2.0, [1.0, 2.0]])


def test_vector_numpy():

    # Rn
    inp = [[1.0, 2.0, 3.0],
           [4.0, 5.0, 6.0]]

    x = vector(inp)
    assert isinstance(x, NumpyTensor)
    assert x.dtype == np.dtype('float64')
    assert all_equal(x, inp)

    x = vector([1.0, 2.0, float('inf')])
    assert x.dtype == np.dtype('float64')
    assert isinstance(x, NumpyTensor)

    x = vector([1.0, 2.0, float('nan')])
    assert x.dtype == np.dtype('float64')
    assert isinstance(x, NumpyTensor)

    x = vector([1, 2, 3], dtype='float32')
    assert x.dtype == np.dtype('float32')
    assert isinstance(x, NumpyTensor)

    # Cn
    inp = [[1 + 1j, 2, 3 - 2j],
           [4 + 1j, 5, 6 - 1j]]

    x = vector(inp)
    assert isinstance(x, NumpyTensor)
    assert x.dtype == np.dtype('complex128')
    assert all_equal(x, inp)

    x = vector([1, 2, 3], dtype='complex64')
    assert isinstance(x, NumpyTensor)

    # Fn
    inp = [1, 2, 3]

    x = vector(inp)
    assert isinstance(x, NumpyTensor)
    assert x.dtype == np.dtype('int')
    assert all_equal(x, inp)

    # Tensors
    inp = ['a', 'b', 'c']
    x = vector(inp)
    assert isinstance(x, NumpyTensor)
    assert np.issubdtype(x.dtype, basestring)
    assert all_equal(x, inp)

    x = vector([1, 2, 'inf'])  # Becomes string type
    assert isinstance(x, NumpyTensor)
    assert np.issubdtype(x.dtype, basestring)
    assert all_equal(x, ['1', '2', 'inf'])

    # Scalar or empty input
    x = vector(5.0)  # becomes 1d, size 1
    assert x.shape == (1,)

    x = vector([])  # becomes 1d, size 0
    assert x.shape == (0,)


def test_auto_weighting(call_variant, weighting, auto_weighting_optimize):
    """Test the auto_weighting decorator for different adjoint variants."""
    rn = odl.rn(2)
    rn_w = odl.rn(2, weighting=weighting)

    class ScalingOpBase(odl.Operator):

        def __init__(self, dom, ran, c):
            super(ScalingOpBase, self).__init__(dom, ran, linear=True)
            self.c = c

    if call_variant == 'oop':

        class ScalingOp(ScalingOpBase):

            def _call(self, x):
                return self.c * x

            @property
            @auto_weighting(optimize=auto_weighting_optimize)
            def adjoint(self):
                return ScalingOp(self.range, self.domain, self.c)

    elif call_variant == 'ip':

        class ScalingOp(ScalingOpBase):

            def _call(self, x, out):
                out[:] = self.c * x
                return out

            @property
            @auto_weighting(optimize=auto_weighting_optimize)
            def adjoint(self):
                return ScalingOp(self.range, self.domain, self.c)

    elif call_variant == 'dual':

        class ScalingOp(ScalingOpBase):

            def _call(self, x, out=None):
                if out is None:
                    out = self.c * x
                else:
                    out[:] = self.c * x
                return out

            @property
            @auto_weighting(optimize=auto_weighting_optimize)
            def adjoint(self):
                return ScalingOp(self.range, self.domain, self.c)

    else:
        assert False

    op1 = ScalingOp(rn, rn_w, 1.5)
    op2 = ScalingOp(rn_w, rn, 1.5)

    for op in [op1, op2]:
        dom_el = noise_element(op.domain)
        ran_el = noise_element(op.range)
        assert pytest.approx(op(dom_el).inner(ran_el),
                             dom_el.inner(op.adjoint(ran_el)))


def test_auto_weighting_noarg():
    """Test the auto_weighting decorator without the optimize argument."""
    rn = odl.rn(2)
    rn_w = odl.rn(2, weighting=2)

    class ScalingOp(odl.Operator):

        def __init__(self, dom, ran, c):
            super(ScalingOp, self).__init__(dom, ran, linear=True)
            self.c = c

        def _call(self, x):
            return self.c * x

        @property
        @auto_weighting
        def adjoint(self):
            return ScalingOp(self.range, self.domain, self.c)

    op1 = ScalingOp(rn, rn, 1.5)
    op2 = ScalingOp(rn_w, rn_w, 1.5)
    op3 = ScalingOp(rn, rn_w, 1.5)
    op4 = ScalingOp(rn_w, rn, 1.5)

    for op in [op1, op2, op3, op4]:
        dom_el = noise_element(op.domain)
        ran_el = noise_element(op.range)
        assert pytest.approx(op(dom_el).inner(ran_el),
                             dom_el.inner(op.adjoint(ran_el)))


def test_auto_weighting_cached_adjoint():
    """Check if auto_weighting plays well with adjoint caching."""
    rn = odl.rn(2)
    rn_w = odl.rn(2, weighting=2)

    class ScalingOp(odl.Operator):

        def __init__(self, dom, ran, c):
            super(ScalingOp, self).__init__(dom, ran, linear=True)
            self.c = c
            self._adjoint = None

        def _call(self, x):
            return self.c * x

        @property
        @auto_weighting
        def adjoint(self):
            if self._adjoint is None:
                self._adjoint = ScalingOp(self.range, self.domain, self.c)
            return self._adjoint

    op = ScalingOp(rn, rn_w, 1.5)
    dom_el = noise_element(op.domain)
    op_eval_before = op(dom_el)

    adj = op.adjoint
    adj_again = op.adjoint
    assert adj_again is adj

    # Check that original op is intact
    assert not hasattr(op, '_call_unweighted')  # op shouldn't be mutated
    op_eval_after = op(dom_el)
    assert all_equal(op_eval_before, op_eval_after)

    dom_el = noise_element(op.domain)
    ran_el = noise_element(op.range)
    op(dom_el)
    op.adjoint(ran_el)
    assert pytest.approx(op(dom_el).inner(ran_el),
                         dom_el.inner(op.adjoint(ran_el)))


def test_auto_weighting_raise_on_return_self():
    """Check that auto_weighting raises when adjoint returns self."""
    rn = odl.rn(2)

    class InvalidScalingOp(odl.Operator):

        def __init__(self, dom, ran, c):
            super(InvalidScalingOp, self).__init__(dom, ran, linear=True)
            self.c = c
            self._adjoint = None

        def _call(self, x):
            return self.c * x

        @property
        @auto_weighting
        def adjoint(self):
            return self

    # This would be a vaild situation for adjont just returning self
    op = InvalidScalingOp(rn, rn, 1.5)
    with pytest.raises(TypeError):
        op.adjoint


if __name__ == '__main__':
    odl.util.test_file(__file__)
