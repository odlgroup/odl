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
from past.builtins import basestring

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
from odl import vector
from odl.util.testutils import all_equal


def test_vector_numpy():

    # Rn
    inp = [1.0, 2.0, 3.0]

    x = vector(inp)
    assert isinstance(x, odl.NumpyFnVector)
    assert x.dtype == np.dtype('float64')
    assert all_equal(x, inp)

    x = vector([1.0, 2.0, float('inf')])
    assert x.dtype == np.dtype('float64')
    assert isinstance(x, odl.NumpyFnVector)

    x = vector([1.0, 2.0, float('nan')])
    assert x.dtype == np.dtype('float64')
    assert isinstance(x, odl.NumpyFnVector)

    x = vector([1, 2, 3], dtype='float32')
    assert x.dtype == np.dtype('float32')
    assert isinstance(x, odl.NumpyFnVector)

    # Cn
    inp = [1 + 1j, 2, 3 - 2j]

    x = vector(inp)
    assert isinstance(x, odl.NumpyFnVector)
    assert x.dtype == np.dtype('complex128')
    assert all_equal(x, inp)

    x = vector([1, 2, 3], dtype='complex64')
    assert isinstance(x, odl.NumpyFnVector)

    # Fn
    inp = [1, 2, 3]

    x = vector(inp)
    assert isinstance(x, odl.NumpyNtuplesVector)
    assert x.dtype == np.dtype('int')
    assert all_equal(x, inp)

    # Ntuples
    inp = ['a', 'b', 'c']
    x = vector(inp)
    assert isinstance(x, odl.NumpyNtuplesVector)
    assert np.issubdtype(x.dtype, basestring)
    assert all_equal(x, inp)

    x = vector([1, 2, 'inf'])  # Becomes string type
    assert isinstance(x, odl.NumpyNtuplesVector)
    assert np.issubdtype(x.dtype, basestring)
    assert all_equal(x, ['1', '2', 'inf'])

    # Input not one-dimensional
    x = vector(5.0)  # OK
    assert x.shape == (1,)

    with pytest.raises(ValueError):
        vector([[1, 0], [0, 1]])


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
