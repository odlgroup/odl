import pytest

import odl

from odl.core.array_API_support import odl_all_equal

from odl.core.util.pytest_config import IMPL_DEVICE_PAIRS
from odl.core.util.testutils import (
    noise_elements, simple_fixture)

DEFAULT_SHAPE = (4,4)

DEFAULT_FILL = 5

from_array = simple_fixture(
    'from_array', ["asarray", "empty_like", "full_like", 'ones_like', 'tril', 'triu', 'zeros_like']
    )

from_impl = simple_fixture(
    'from_impl', ['arange', 'empty', 'eye', "full", 'linspace', 'meshgrid', 'ones', 'zeros']
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

def test_from_array(float_tspace, from_array):
    ns = float_tspace.array_namespace
    arr_fn = getattr(ns, from_array)
    odl_fn = getattr(odl, from_array)

    x_arr, x = noise_elements(float_tspace, 1)

    if from_array == 'full_like':
        y_arr = arr_fn(x_arr, fill_value=DEFAULT_FILL)
        y     = odl_fn(x, fill_value=DEFAULT_FILL)
    else:
        y_arr = arr_fn(x_arr)
        y     = odl_fn(x)
    if from_array == 'empty_like':
        pytest.skip("Skipping equality check for empty_like")
    
    else:
        assert odl_all_equal(y_arr, y)

# Pytorch and Numpy API still vary, making the systematic testing of these functions premature
# def test_from_impl(float_tspace, from_impl):
#     ns = float_tspace.array_namespace
#     arr_fn = getattr(ns, from_impl)
#     odl_fn = getattr(odl, from_impl)

#     # x_arr, x = noise_elements(float_tspace, 1)
#     args = ()
#     kwargs = {
#             'shape'  : (4,4),
#             'dtype'  : float_tspace.dtype_identifier,
#             'device' : float_tspace.device
#         }
#     if from_impl == 'arange':
#         args = [1]
#         kwargs['start'] = 1
#         kwargs['stop'] = 10
#         kwargs['step'] = 1
            
#     elif from_impl == 'eye':
#         kwargs['n_rows'] = 4
#         kwargs['n_cols'] = 4
#         kwargs['k'] = 0

#     elif from_impl == 'meshgrid':
#         args = [
#             float_tspace.array_backend.array_constructor([0,1,2,3], 
#             device = float_tspace.device,
#             dtype = float_tspace.dtype),
#             float_tspace.array_backend.array_constructor([0,1,2,3], 
#             device = float_tspace.device,
#             dtype = float_tspace.dtype)
#             ]

#     elif from_impl == 'tril' or from_impl == 'triu':
#         kwargs['k'] = 2

#     print(args, kwargs)
#     assert odl_all_equal(
#         arr_fn(*args, **kwargs), odl_fn(*args, **kwargs)
    # )
