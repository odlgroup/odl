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
import numpy as np
import pytest

# ODL imports
import odl
from odl.util.testutils import (all_equal, all_almost_equal, almost_equal,
                                skip_if_no_cuda)


@skip_if_no_cuda
def test_CudaRnToRn():
    r3 = odl.Rn(3)
    cur3 = odl.CudaRn(3)

    cast = odl.EmbeddingFnInFn(r3, cur3)

    x = cur3.element([1, 2, 3])
    y = cast(x)
    assert y in r3
    assert all_almost_equal(x, y)

    # adjoint
    assert isinstance(cast.adjoint, odl.EmbeddingFnInFn)

    z = cast.adjoint(y)
    assert z in cur3
    assert all_almost_equal(z, x)


def test_LpToRn():
    L2 = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(L2, [3, 3])

    r9 = odl.Rn(9)

    cast = odl.EmbeddingFnInFn(r9, discr)

    x = discr.element([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    y = cast(x)
    assert y in r9
    assert all_almost_equal(x, y)

    # adjoint
    assert isinstance(cast.adjoint, odl.EmbeddingFnInFn)

    z = cast.adjoint(y)
    assert z in discr
    assert all_almost_equal(z, x)


@skip_if_no_cuda
def test_LpToRn_cuda():
    L2 = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(L2, [3, 3], impl='cuda')

    r9 = odl.Rn(9)

    cast = odl.EmbeddingFnInFn(r9, discr)

    x = discr.element([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    y = cast(x)
    assert y in r9
    assert all_almost_equal(y, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    # adjoint
    assert isinstance(cast.adjoint, odl.EmbeddingFnInFn)

    z = cast.adjoint(y)
    assert z in discr
    assert all_almost_equal(z, x)


def test_LpToRn_forder():
    L2 = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(L2, [3, 3], order='F')

    r9 = odl.Rn(9)

    cast = odl.EmbeddingFnInFn(r9, discr)

    x = discr.element([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

    # Call
    y = cast(x)
    assert y in r9
    assert all_almost_equal(y, [1, 4, 7, 2, 5, 8, 3, 6, 9])

    # Zero overhead
    assert x.ntuple.data.ctypes.data == y.data.ctypes.data

    # adjoint
    assert isinstance(cast.adjoint, odl.EmbeddingFnInFn)

    z = cast.adjoint(y)
    assert z in discr
    assert all_almost_equal(z, x)

    # Zero overhead
    assert z.ntuple.data.ctypes.data == y.data.ctypes.data


def test_PSpaceInFn():
    r3x2 = odl.ProductSpace(odl.Rn(3), 2)
    r6 = odl.Rn(6)

    cast = odl.EmbeddingPowerSpaceInFn(r6, r3x2)

    x = r3x2.element([[1, 2, 3],
                      [4, 5, 6]])

    # Call
    y = cast(x)
    assert y in r6
    assert all_almost_equal(y, [1, 2, 3, 4, 5, 6])

    # adjoint
    assert isinstance(cast.adjoint, odl.EmbeddingFnInPowerSpace)

    z = cast.adjoint(y)
    assert z in r3x2
    assert all_almost_equal(z, x)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
