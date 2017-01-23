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

from odl import vector
from odl.space.npy_tensors import NumpyTensor, NumpyGeneralizedTensor
from odl.util.testutils import all_equal


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
    assert isinstance(x, NumpyGeneralizedTensor)
    assert x.dtype == np.dtype('int')
    assert all_equal(x, inp)

    # Tensors
    inp = ['a', 'b', 'c']
    x = vector(inp)
    assert isinstance(x, NumpyGeneralizedTensor)
    assert np.issubdtype(x.dtype, basestring)
    assert all_equal(x, inp)

    x = vector([1, 2, 'inf'])  # Becomes string type
    assert isinstance(x, NumpyGeneralizedTensor)
    assert np.issubdtype(x.dtype, basestring)
    assert all_equal(x, ['1', '2', 'inf'])

    # Scalar or empty input
    x = vector(5.0)  # becomes 1d, size 1
    assert x.shape == (1,)

    x = vector([])  # becomes 1d, size 0
    assert x.shape == (0,)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
