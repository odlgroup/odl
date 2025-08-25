import pytest

import odl
from odl.util.pytest_config import IMPL_DEVICE_PAIRS
from odl.util.testutils import all_almost_equal

try:
    import torch
except ImportError:
    pass

skip_if_no_pytorch = pytest.mark.skipif(
        "'pytorch' not in odl.space.entry_points.TENSOR_SPACE_IMPLS",
        reason='PYTORCH not available',
    )

IMPLS = [pytest.param(value, marks=skip_if_no_pytorch) for value in IMPL_DEVICE_PAIRS]

DEFAULT_SHAPE = (4,4)

@pytest.fixture(scope='module', params=IMPLS)
def tspace(request, odl_floating_dtype):
    impl, device = request.param
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_floating_dtype, 
        impl=impl, 
        device=device
    )

@pytest.fixture(scope='module')
def numpy_tspace(odl_floating_dtype):
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_floating_dtype, 
        impl='numpy', 
        device='cpu'
    )

@pytest.fixture(scope='module')
def pytorch_tspace_cpu(odl_floating_dtype):
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_floating_dtype, 
        impl='pytorch', 
        device='cpu'
    )

@pytest.fixture(scope='module')
def pytorch_tspace_gpu(odl_floating_dtype):
    if torch.cuda.device_count() == 0:
        pytest.skip(reason="No Cuda-capable GPU available")

    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_floating_dtype, 
        impl='pytorch', 
        device='cuda:0'
    )

def test_same_backend_same_device(tspace, odl_arithmetic_op):
    """Test that operations between two elements on separate spaces with the same backend are possible"""
    x = next(tspace.examples)[1]
    y = next(tspace.examples)[1]
    op = odl_arithmetic_op
    z_arr = op(x.data, y.data)
    z = op(x, y)
    assert all_almost_equal([x, y, z], [x.data, y.data, z_arr])

@skip_if_no_pytorch
def test_different_backends(
        numpy_tspace, pytorch_tspace_cpu, pytorch_tspace_gpu,
        odl_arithmetic_op
        ):
    """Test that operations between two elements on separate spaces with different device or impl are not possible"""
    x_np = next(numpy_tspace.examples)[1]
    x_pt_cpu = next(pytorch_tspace_cpu.examples)[1]
    x_pt_gpu = next(pytorch_tspace_gpu.examples)[1]
    op = odl_arithmetic_op

    # Same device, different backend
    with pytest.raises(AssertionError):
        res = op(x_np, x_pt_cpu) 
    
    with pytest.raises(TypeError):
        res = op(x_np, x_pt_cpu.data) 

    with pytest.raises(TypeError):
        res = op(x_np.data, x_pt_cpu) 

    # Same backend, different device
    with pytest.raises(AssertionError):
        res = op(x_pt_gpu, x_pt_cpu) 
    
    with pytest.raises(TypeError):
        res = op(x_pt_gpu.data, x_pt_cpu) 

    with pytest.raises(TypeError):
        res = op(x_pt_gpu, x_pt_cpu.data) 

    # Different device, different backend
    with pytest.raises(AssertionError):
        res = op(x_np, x_pt_gpu) 

    with pytest.raises(TypeError):
        res = op(x_np, x_pt_gpu.data) 

    with pytest.raises(TypeError):
        res = op(x_np.data, x_pt_gpu) 


    