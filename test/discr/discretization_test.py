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

# External
import pytest
import numpy as np

# Internal
import odl
from odl.discr.discretization import dspace_type
from odl.util.testutils import skip_if_no_cuda


def test_dspace_type_numpy():
    # Plain function set -> Ntuples-like
    fset = odl.FunctionSet(odl.Interval(0, 1), odl.Strings(2))
    assert dspace_type(fset, 'numpy') == odl.Ntuples
    assert dspace_type(fset, 'numpy', np.int) == odl.Ntuples

    # Real space
    rspc = odl.FunctionSpace(odl.Interval(0, 1), field=odl.RealNumbers())
    assert dspace_type(rspc, 'numpy') == odl.Rn
    assert dspace_type(rspc, 'numpy', np.float32) == odl.Rn
    assert dspace_type(rspc, 'numpy', np.int) == odl.Fn
    with pytest.raises(TypeError):
        dspace_type(rspc, 'numpy', np.complex)
    with pytest.raises(TypeError):
        dspace_type(rspc, 'numpy', np.dtype('<U2'))

    # Complex space
    cspc = odl.FunctionSpace(odl.Interval(0, 1), field=odl.ComplexNumbers())
    assert dspace_type(cspc, 'numpy') == odl.Cn
    assert dspace_type(cspc, 'numpy', np.complex64) == odl.Cn
    with pytest.raises(TypeError):
        dspace_type(cspc, 'numpy', np.float)
    with pytest.raises(TypeError):
        assert dspace_type(cspc, 'numpy', np.int)
    with pytest.raises(TypeError):
        dspace_type(cspc, 'numpy', np.dtype('<U2'))


@skip_if_no_cuda
def test_dspace_type_cuda():
    # Plain function set -> Ntuples-like
    fset = odl.FunctionSet(odl.Interval(0, 1), odl.Strings(2))
    assert dspace_type(fset, 'cuda') == odl.CudaNtuples
    assert dspace_type(fset, 'cuda', np.int) == odl.CudaNtuples

    # Real space
    rspc = odl.FunctionSpace(odl.Interval(0, 1), field=odl.RealNumbers())
    assert dspace_type(rspc, 'cuda') == odl.CudaRn
    assert dspace_type(rspc, 'cuda', np.float32) == odl.CudaRn
    assert dspace_type(rspc, 'cuda', np.int) == odl.CudaFn
    with pytest.raises(TypeError):
        dspace_type(rspc, 'cuda', np.complex)
    with pytest.raises(TypeError):
        dspace_type(rspc, 'cuda', np.dtype('<U2'))

    # Complex space (not implemented)
    cspc = odl.FunctionSpace(odl.Interval(0, 1), field=odl.ComplexNumbers())
    with pytest.raises(NotImplementedError):
        dspace_type(cspc, 'cuda')
    with pytest.raises(NotImplementedError):
        dspace_type(cspc, 'cuda', np.complex64)
    with pytest.raises(TypeError):
        dspace_type(cspc, 'cuda', np.float)
    with pytest.raises(TypeError):
        assert dspace_type(cspc, 'cuda', np.int)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
