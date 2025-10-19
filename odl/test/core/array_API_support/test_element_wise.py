import pytest

import odl
from odl.core.util.pytest_config import IMPL_DEVICE_PAIRS
from odl.core.util.testutils import (
    all_almost_equal, all_equal, noise_array, noise_element, noise_elements,
    isclose, simple_fixture)



DEFAULT_SHAPE = (4,4)

one_operand_op = simple_fixture(
    'one_operand_op', 
    ['abs', 'asinh', 'atan', 'conj', 'cos', 'cosh', 'exp', 'expm1', 'floor', 'imag', 'isfinite', 'isinf', 'isnan', 'log', 'log1p', 'log2', 'log10', 'logical_not', 'positive', 'real', 'reciprocal', 'round', 'sign', 'signbit', 'sin', 'sinh', 'sqrt', 'square', 'tan', 'tanh', 'trunc']
    )

domain_restricted_op = simple_fixture(
    'domain_restricted_op', 
    ['acos', 'acosh', 'asin', 'atanh']
    )

integer_op = simple_fixture(
    'integer_op', 
    ['bitwise_invert',]
    )

two_operands_op = simple_fixture(
    'two_operands_op', 
    ['add', 'atan2', 'copysign', 'divide', 'equal', 'floor_divide', 'greater', 'greater_equal', 'hypot', 'less', 'less_equal', 'logaddexp', 'logical_and', 'logical_or', 'logical_xor', 'maximum', 'minimum', 'multiply', 'nextafter', 'not_equal', 'pow', 'remainder', 'subtract']
    )

two_operands_op_integer = simple_fixture(
    'two_operands_op_integer', 
    ['bitwise_and', 'bitwise_left_shift', 'bitwise_or', 'bitwise_right_shift', 'bitwise_xor']
    )

kwargs_op = simple_fixture(
    'kwargs_op', 
    ['clip']
    )

inplace = simple_fixture(
    'inplace', 
    [True, False]
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

@pytest.fixture(scope='module', params=IMPL_DEVICE_PAIRS)
def integer_tspace(request):
    impl, device = request.param
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype='int64',
        impl=impl, 
        device=device
    )

def test_one_operand_op_real(float_tspace, one_operand_op, inplace):
    if one_operand_op == 'imag' and float_tspace.impl == 'pytorch':
        pytest.skip(f'imag is not implemented for tensors with non-complex dtypes in Pytorch.')
    ns = float_tspace.array_namespace
    arr_fn = getattr(ns, one_operand_op)
    odl_fn = getattr(odl, one_operand_op)

    x_arr, x = noise_elements(float_tspace, 1)
    x_arr = ns.abs(x_arr) + 0.1
    x = odl.abs(x) + 0.1
    
    if inplace:
        if one_operand_op in ['imag', 'sign', 'real', 'positive', 'isnan', 'isinf', 'isfinite']:
            pytest.skip(f'{one_operand_op} is not supported for inplace updates')
        if one_operand_op == 'signbit':
            out = odl.tensor_space(
                shape=DEFAULT_SHAPE, 
                dtype=bool,
                impl=float_tspace.impl, 
                device=float_tspace.device
            ).element()
        else:
            out = float_tspace.element()
        out_arr = out.data
        y = odl_fn(x, out=out)
        y_arr = arr_fn(x_arr, out=out_arr)
        assert all_equal(y, y_arr)
        assert all_equal(y, out)
    
    else:        
        y = odl_fn(x)
        y_arr = arr_fn(x_arr)
        assert all_equal(y, y_arr)

def test_one_operand_op_real_kwargs(float_tspace, kwargs_op, inplace):
    ns = float_tspace.array_namespace
    arr_fn = getattr(ns, kwargs_op)
    odl_fn = getattr(odl, kwargs_op)

    x_arr, x = noise_elements(float_tspace, 1)
    if inplace:
        out = float_tspace.element()
        out_arr = out.data
        y = odl_fn(x, out=out)
        y_arr = arr_fn(x_arr, out=out_arr)
        assert all_equal(y, y_arr)
        assert all_equal(y, out)
    else:
        y = odl_fn(x, min=0, max=1)
        y_arr = arr_fn(x_arr, min=0, max=1)
        assert all_equal(y, y_arr)

def test_one_operand_op_integer(integer_tspace, integer_op, inplace):
    ns = integer_tspace.array_namespace
    arr_fn = getattr(ns, integer_op)
    odl_fn = getattr(odl, integer_op)

    x_arr, x = noise_elements(integer_tspace, 1)
    ### ODL operation
    if inplace:
        out = integer_tspace.element()
        out_arr = out.data
        y = odl_fn(x, out=out)
        y_arr = arr_fn(x_arr, out=out_arr)
        assert all_equal(y, y_arr)
        assert all_equal(y, out)

    else:
        y = odl_fn(x)
        y_arr = arr_fn(x_arr)

        assert all_equal(y, y_arr)

def test_domain_restricted_op(float_tspace, domain_restricted_op):
    ns = float_tspace.array_namespace
    arr_fn = getattr(ns, domain_restricted_op)
    odl_fn = getattr(odl, domain_restricted_op)

    x = 0.5 * float_tspace.one()
    x_arr = x.data
    if inplace:
        out = float_tspace.element()
        out_arr = out.data
        y = odl_fn(x, out=out)
        y_arr = arr_fn(x_arr, out=out_arr)
        assert all_almost_equal(y, y_arr)
        assert all_almost_equal(y, out)
        assert all_almost_equal(y_arr, out_arr)
    else:
        y = odl_fn(x)
        y_arr = arr_fn(x_arr)
        assert all_almost_equal(y, y_arr)
    
def test_two_operands_op_real(float_tspace, two_operands_op):
    ns = float_tspace.array_namespace

    arr_fn = getattr(ns, two_operands_op)
    odl_fn = getattr(odl, two_operands_op)

    [x_arr, y_arr], [x, y] = noise_elements(float_tspace, 2)
    if inplace:
        out = float_tspace.element()
        out_arr = out.data
        z = odl_fn(x, y, out=out)
        z_arr = arr_fn(x_arr, y_arr, out=out_arr)
        assert all_almost_equal(z, z_arr)
        assert all_almost_equal(z, out)
        assert all_almost_equal(z_arr, out_arr)
    else:
        z = odl_fn(x, y)
        z_arr = arr_fn(x_arr, y_arr)
        assert all_almost_equal(z, z_arr)

def test_two_operands_op_integer(integer_tspace, two_operands_op_integer):
    ns = integer_tspace.array_namespace
    arr_fn = getattr(ns, two_operands_op_integer)
    odl_fn = getattr(odl, two_operands_op_integer)

    [x_arr, y_arr], [x, y] = noise_elements(integer_tspace, 2)
    if inplace:
        out = integer_tspace.element()
        out_arr = out.data
        z = odl_fn(x, y, out=out)
        z_arr = arr_fn(x_arr, y_arr, out=out_arr)
        assert all_equal(z, z_arr)
        assert all_equal(z, out)
    else:
        z = odl_fn(x, y)
        z_arr = arr_fn(x_arr, y_arr)
        assert all_almost_equal(z, z_arr)
