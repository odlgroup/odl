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
from odl.util.testutils import all_equal


def test_vector_numpy():

    # Rn
    inp = [1.0, 2.0, 3.0]

    x = vector(inp)
    assert isinstance(x, odl.NumpyTensor)
    assert x.dtype == np.dtype('float64')
    assert all_equal(x, inp)

    x = vector([1.0, 2.0, float('inf')])
    assert x.dtype == np.dtype('float64')
    assert isinstance(x, odl.NumpyTensor)

    x = vector([1.0, 2.0, float('nan')])
    assert x.dtype == np.dtype('float64')
    assert isinstance(x, odl.NumpyTensor)

    x = vector([1, 2, 3], dtype='float32')
    assert x.dtype == np.dtype('float32')
    assert isinstance(x, odl.NumpyTensor)

    # Cn
    inp = [1 + 1j, 2, 3 - 2j]

    x = vector(inp)
    assert isinstance(x, odl.NumpyTensor)
    assert x.dtype == np.dtype('complex128')
    assert all_equal(x, inp)

    x = vector([1, 2, 3], dtype='complex64')
    assert isinstance(x, odl.NumpyTensor)

    # Fn
    inp = [1, 2, 3]

    x = vector(inp)
    assert isinstance(x, odl.NumpyTensorSetVector)
    assert x.dtype == np.dtype('int')
    assert all_equal(x, inp)

    # Tensors
    inp = ['a', 'b', 'c']
    x = vector(inp)
    assert isinstance(x, odl.NumpyTensorSetVector)
    assert np.issubdtype(x.dtype, basestring)
    assert all_equal(x, inp)

    x = vector([1, 2, 'inf'])  # Becomes string type
    assert isinstance(x, odl.NumpyTensorSetVector)
    assert np.issubdtype(x.dtype, basestring)
    assert all_equal(x, ['1', '2', 'inf'])

    # Input not one-dimensional
    x = vector(5.0)  # OK
    assert x.shape == (1,)

    with pytest.raises(ValueError):
        vector([[1, 0], [0, 1]])


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
