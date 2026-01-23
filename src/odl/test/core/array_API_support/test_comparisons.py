import pytest

import odl

from odl.core.util.testutils import (
    noise_elements, simple_fixture)

DEFAULT_SHAPE = (4,4)

elementwise_comparison = simple_fixture(
    'elementwise', ["isclose" ]
    )

reduction_comparison = simple_fixture(
    'reduction', ["allclose", "odl_all_equal"]
    )

truth_value_comparison = simple_fixture(
    'truth_value', ["all", "any",]
    )


@pytest.fixture(scope='module')
def float_tspace(odl_real_floating_dtype, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_real_floating_dtype, 
        impl=impl, 
        device=device
    )

def test_elementwise(float_tspace, elementwise_comparison):
    ns = float_tspace.array_namespace
    arr_fn = getattr(ns, elementwise_comparison)
    odl_fn = getattr(odl, elementwise_comparison)

    xarr0, x0 = noise_elements(float_tspace, 1)
    xarr1, x1 = noise_elements(float_tspace, 1)

    assert (arr_fn(xarr0, xarr0) == odl_fn(x0, x0)).all()
    assert (arr_fn(xarr0, xarr1) == odl_fn(x0, x1)).all()
    assert (arr_fn(xarr1, xarr0) == odl_fn(x1, x0)).all()

def test_reduction(float_tspace, reduction_comparison):
    ns = float_tspace.array_namespace
    xarr0, x0 = noise_elements(float_tspace, 1)
    xarr1, x1 = noise_elements(float_tspace, 1)
    odl_fn = getattr(odl, reduction_comparison)

    if reduction_comparison == 'allclose':
        arr_fn = getattr(ns, reduction_comparison)
        
    elif reduction_comparison == 'odl_all_equal':
        all_fn   = getattr(ns, 'all')
        equal_fn = getattr(ns, 'equal')
        def arr_fn(x, y):
            return all_fn(equal_fn(x, y)) 
        
    else:
        raise ValueError
        
    assert arr_fn(xarr0, xarr0) == odl_fn(x0, x0)
    assert arr_fn(xarr0, xarr1) == odl_fn(x0, x1)
    assert arr_fn(xarr1, xarr0) == odl_fn(x1, x0)

def test_array_truth_value(float_tspace, truth_value_comparison):
    ns = float_tspace.array_namespace
    arr_fn = getattr(ns, truth_value_comparison)
    odl_fn = getattr(odl, truth_value_comparison)

    xarr0, x0 = noise_elements(float_tspace, 1)
    xarr1, x1 = noise_elements(float_tspace, 1)

    arr_isclose = getattr(ns, 'isclose')
    odl_isclose = getattr(odl, 'isclose')

    expr_0 = arr_isclose(xarr0, xarr0) == odl_isclose(x0, x0)
    expr_1 = arr_isclose(xarr0, xarr1) == odl_isclose(x0, x1)
    expr_2 = arr_isclose(xarr1, xarr0) == odl_isclose(x1, x0)
    assert arr_fn(expr_0) == odl_fn(expr_0)
    assert arr_fn(expr_1) == odl_fn(expr_1)
    assert arr_fn(expr_2) == odl_fn(expr_2)
    
