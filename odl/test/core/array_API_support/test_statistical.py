import pytest

import odl

from odl.core.array_API_support.comparisons  import odl_all_equal

from odl.core.util.pytest_config import IMPL_DEVICE_PAIRS
from odl.core.util.testutils import (
    noise_elements, simple_fixture)

DEFAULT_SHAPE = (4,4)

keepdims_function = simple_fixture(
    'keepdims', 
    ['max',
    'mean',
    'min',
    'prod',
    'std',
    'sum',
    'var' ]
    )

cumulative_function = simple_fixture(
    'cumulative', 
    ['cumulative_prod',
    'cumulative_sum']
    )

keepdims = simple_fixture(
    'keepdims', 
    [True, False]
    )

axis = simple_fixture(
    'axis', 
    [0, 1]
    )


@pytest.fixture(scope='module', params=IMPL_DEVICE_PAIRS)
def float_tspace(request, odl_real_floating_dtype):
    impl, device = request.param
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_real_floating_dtype, 
        impl=impl, 
        device=device
    )

def test_keepdims_function(float_tspace, keepdims_function, keepdims):
    ns = float_tspace.array_namespace
    arr_fn = getattr(ns, keepdims_function)
    odl_fn = getattr(odl, keepdims_function)

    x_arr, x = noise_elements(float_tspace, 1)
    y = odl_fn(x, keepdims=keepdims)
    y_arr = arr_fn(x_arr, keepdims=keepdims)
    assert odl_all_equal(y, y_arr)

def test_cumulative_function(float_tspace, cumulative_function, axis):
    ns = float_tspace.array_namespace
    arr_fn = getattr(ns, cumulative_function)
    odl_fn = getattr(odl, cumulative_function)

    x_arr, x = noise_elements(float_tspace, 1)
    y = odl_fn(x, axis=axis)
    y_arr = arr_fn(x_arr, axis=axis)
    assert odl_all_equal(y, y_arr)
