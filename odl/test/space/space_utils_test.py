# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

import odl
from odl import vector
from odl.space.entry_points import TENSOR_SPACE_IMPLS
from odl.core.util.testutils import all_equal, default_precision_dict
import pytest 

error_dict = {
    'pytorch' : TypeError,
    'numpy'   : ValueError
}

def test_vector_numpy(odl_impl_device_pairs):

    impl, device = odl_impl_device_pairs
    tspace = TENSOR_SPACE_IMPLS[impl]((0))
    tspace_element_type = tspace.element_type

    inp = [[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]]

    x = vector(inp, impl=impl, device=device)

    assert isinstance(x, tspace_element_type)
    assert x.dtype_identifier == default_precision_dict[impl]['float']
    assert all_equal(x, inp)

    x = vector([1.0, 2.0, float('inf')], impl=impl, device=device)
    assert x.dtype_identifier == default_precision_dict[impl]['float']
    assert isinstance(x, tspace_element_type)

    x = vector([1.0, 2.0, float('nan')], impl=impl, device=device)
    assert x.dtype_identifier == default_precision_dict[impl]['float']
    assert isinstance(x, tspace_element_type)

    x = vector([1, 2, 3], dtype='float32', impl=impl, device=device)
    assert x.dtype_identifier == 'float32'
    assert isinstance(x, tspace_element_type)

    # Cn
    inp = [[1 + 1j, 2, 3 - 2j],
            [4 + 1j, 5, 6 - 1j]]

    x = vector(inp, impl=impl, device=device)
    assert isinstance(x, tspace_element_type)
    assert x.dtype_identifier == default_precision_dict[impl]['complex']
    assert all_equal(x, inp)

    x = vector([1, 2, 3], dtype='complex64', impl=impl, device=device)
    assert isinstance(x, tspace_element_type)

    # Generic TensorSpace
    inp = [1, 2, 3]
    x = vector(inp,impl=impl, device=device)
    assert isinstance(x, tspace_element_type)
    assert x.dtype_identifier == 'int64'
    assert all_equal(x, inp)

    inp = ['a', 'b', 'c']
    with pytest.raises(ValueError):
        x = vector(inp ,impl=impl, device=device)

    inp = [1, 2, 'inf']
    with pytest.raises(error_dict[impl]):
        x = vector(inp,impl=impl, device=device)

    # Scalar or empty input
    x = vector(5.0 ,impl=impl, device=device) # becomes 1d, size 1
    assert x.shape == ()

    x = vector([])  # becomes 1d, size 0
    assert x.shape == (0,)


if __name__ == '__main__':
    odl.core.util.test_file(__file__)
