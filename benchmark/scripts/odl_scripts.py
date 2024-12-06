from typing import Dict
import time 

import numpy as np
import odl

def mri_mlem_adam(
    parameters     : Dict
    ):
    # Unpack dictionnary 
    max_iterations = parameters['max_iterations']
    dimension = parameters['dimension']
    n_points  = parameters['n_points']
    subsampling  : float = parameters['subsampling']
    learning_rate: float = parameters['learning_rate']
    beta1: float = parameters['beta1']
    beta2: float = parameters['beta2']
    eps:   float = parameters['eps']
    # Create a space
    space = odl.uniform_discr(
            [0 for _ in range(dimension)], 
            [n_points for _ in range(dimension)], 
            [n_points for _ in range(dimension)]
        )
    # Create MRI operator. First fourier transform, then subsample
    ft = odl.trafos.FourierTransform(space)
    sampling_points = np.random.rand(*ft.range.shape) < subsampling #type:ignore
    sampling_mask = ft.range.element(sampling_points)
    mri_op = sampling_mask * ft

    # Create noisy MRI data
    phantom = odl.phantom.shepp_logan(space, modified=True)
    noisy_data = mri_op(phantom) + odl.phantom.white_noise(mri_op.range) * 0.1  #type:ignore

    g = odl.solvers.L2Norm(mri_op.range).translated(noisy_data) * mri_op

    # Solve
    x = mri_op.domain.zero()
    odl.solvers.adam(
        g, x, 
        maxiter=max_iterations, 
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps)

    ### Return the data; compare it against target (l2 norm)
    return np.linalg.norm(phantom - x.data)


def forward(
    parameters     : Dict,
    ):
    # Unpack dictionnary 
    dimension = parameters['dimension']
    n_points  = parameters['n_points']
    reco_space_impl   = parameters['reco_space_impl']
    device_name = parameters['device_name']
    ray_trafo_impl    = parameters['ray_trafo_impl']
    n_angles = parameters['n_angles']
    # Create a space
    if reco_space_impl == 'numpy':
        reco_space = odl.uniform_discr(
                min_pt = [-20 for _ in range(dimension)], 
                max_pt = [ 20 for _ in range(dimension)], 
                shape  = [n_points for _ in range(dimension)],
                dtype  = 'float32',
                impl   = reco_space_impl
            )
    else:
        reco_space = odl.uniform_discr(
                min_pt = [-20 for _ in range(dimension)], 
                max_pt = [ 20 for _ in range(dimension)], 
                shape  = [n_points for _ in range(dimension)],
                dtype  = 'float32',
                impl   = reco_space_impl, 
                torch_device = device_name
            )
    
    angle_partition = odl.uniform_partition(0, 2*np.pi, n_angles)
    detector_partition = odl.uniform_partition(
        [-30, -30], 
        [30, 30], 
        [512, 512]
        )
    geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

    ray_trafo = odl.tomo.RayTransform(
        reco_space, 
        geometry, 
        impl=ray_trafo_impl
        )
    
    phantom = odl.phantom.shepp_logan(reco_space, modified=True)
    
    start = time.time()
    proj_data = ray_trafo(phantom)    
    end = time.time()
    return {'time':end-start}

def backward(
        parameters : Dict
        ):
    # Unpack dictionnary 
    dimension = parameters['dimension']
    n_points  = parameters['n_points']
    reco_space_impl   = parameters['reco_space_impl']
    device_name = parameters['device_name']
    ray_trafo_impl    = parameters['ray_trafo_impl']
    n_angles = parameters['n_angles']
    # Create a space
    if reco_space_impl == 'numpy':
        reco_space = odl.uniform_discr(
                min_pt = [-20 for _ in range(dimension)], 
                max_pt = [ 20 for _ in range(dimension)], 
                shape  = [n_points for _ in range(dimension)],
                dtype  = 'float32',
                impl   = reco_space_impl
            )
    else:
        reco_space = odl.uniform_discr(
                min_pt = [-20 for _ in range(dimension)], 
                max_pt = [ 20 for _ in range(dimension)], 
                shape  = [n_points for _ in range(dimension)],
                dtype  = 'float32',
                impl   = reco_space_impl, 
                torch_device = device_name
            )
    
    angle_partition = odl.uniform_partition(0, 2*np.pi, n_angles)
    detector_partition = odl.uniform_partition(
        [-30, -30], 
        [30, 30], 
        [512, 512]
        )
    geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

    ray_trafo = odl.tomo.RayTransform(
        reco_space, 
        geometry, 
        impl=ray_trafo_impl
        )
    
    phantom = odl.phantom.shepp_logan(reco_space, modified=True)
    
    proj_data = ray_trafo(phantom)

    start = time.time()
    rec_data  = ray_trafo.adjoint(proj_data)
    end = time.time()
    return {'time':end-start}

def pytorch_wrapper_forward(
        parameters : Dict
    ):
    if parameters['operator'] == 'new_operator':
        from odl.contrib.torch.new_operator import OperatorModule
    elif parameters['operator'] == 'old_operator':
        from odl.contrib.torch.operator import OperatorModule
    else:
        raise NotImplementedError
    import torch
    dimension = parameters['dimension']
    n_points  = parameters['n_points']
    reco_space_impl   = parameters['reco_space_impl']
    device_name = parameters['device_name']
    ray_trafo_impl    = parameters['ray_trafo_impl']
    n_angles = parameters['n_angles']
    if reco_space_impl == 'numpy':
        reco_space = odl.uniform_discr(
                min_pt = [-20 for _ in range(dimension)], 
                max_pt = [ 20 for _ in range(dimension)], 
                shape  = [n_points for _ in range(dimension)],
                dtype  = 'float32',
                impl   = reco_space_impl
            )
    else:
        reco_space = odl.uniform_discr(
                    min_pt = [-20 for _ in range(dimension)], 
                    max_pt = [ 20 for _ in range(dimension)], 
                    shape  = [n_points for _ in range(dimension)],
                    dtype  = 'float32',
                    impl   = reco_space_impl, 
                    torch_device = device_name
                )
    odl_phantom = odl.phantom.shepp_logan(reco_space, modified=True)
    if reco_space_impl == 'numpy':
        phantom : torch.Tensor = torch.from_numpy(odl_phantom.asarray()).unsqueeze(0).unsqueeze(0).to(device_name)
    else:
        phantom : torch.Tensor = odl_phantom.asarray().unsqueeze(0).unsqueeze(0).to(device_name)
    # <!> enforce float32 conversion, rather than float64
    phantom = phantom.to(dtype=torch.float32)
    # <!> make tensor contiguous from creation
    phantom = phantom.contiguous()
    # <!> for the example, input_tensor.requires_grad == True
    phantom.requires_grad_()
    # Make a 3d single-axis parallel beam geometry with flat detector
    # Angles: uniformly spaced, n = 180, min = 0, max = pi
    angle_partition = odl.uniform_partition(0, 2*np.pi, n_angles)
    detector_partition = odl.uniform_partition(
        [-30, -30], 
        [30, 30], 
        [512, 512]
        )
    geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

    # Ray transform (= forward projection).
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=ray_trafo_impl)

    forward_module  = OperatorModule(ray_trafo)
    # backward_module = OperatorModule(ray_trafo.adjoint)
    start = time.time()
    sinogram :torch.Tensor = forward_module(phantom) #type:ignore
    end = time.time()
    return {'time':end-start}

def pytorch_wrapper_backward(
        parameters : Dict
    ):
    if parameters['operator'] == 'new_operator':
        from odl.contrib.torch.new_operator import OperatorModule
    elif parameters['operator'] == 'old_operator':
        from odl.contrib.torch.operator import OperatorModule
    else:
        raise NotImplementedError
    import torch
    dimension = parameters['dimension']
    n_points  = parameters['n_points']
    reco_space_impl   = parameters['reco_space_impl']
    device_name = parameters['device_name']
    ray_trafo_impl    = parameters['ray_trafo_impl']
    n_angles = parameters['n_angles']
    if reco_space_impl == 'numpy':
        reco_space = odl.uniform_discr(
                min_pt = [-20 for _ in range(dimension)], 
                max_pt = [ 20 for _ in range(dimension)], 
                shape  = [n_points for _ in range(dimension)],
                dtype  = 'float32',
                impl   = reco_space_impl
            )
    else:
        reco_space = odl.uniform_discr(
                    min_pt = [-20 for _ in range(dimension)], 
                    max_pt = [ 20 for _ in range(dimension)], 
                    shape  = [n_points for _ in range(dimension)],
                    dtype  = 'float32',
                    impl   = reco_space_impl, 
                    torch_device = device_name
                )
    odl_phantom = odl.phantom.shepp_logan(reco_space, modified=True)
    if reco_space_impl == 'numpy':
        phantom : torch.Tensor = torch.from_numpy(odl_phantom.asarray()).unsqueeze(0).unsqueeze(0).to(device_name)
    else:
        phantom : torch.Tensor = odl_phantom.asarray().unsqueeze(0).unsqueeze(0).to(device_name)
    # <!> enforce float32 conversion, rather than float64
    phantom = phantom.to(dtype=torch.float32)
    # <!> make tensor contiguous from creation
    phantom = phantom.contiguous()
    # <!> for the example, input_tensor.requires_grad == True
    phantom.requires_grad_()
    # Make a 3d single-axis parallel beam geometry with flat detector
    # Angles: uniformly spaced, n = 180, min = 0, max = pi
    angle_partition = odl.uniform_partition(0, 2*np.pi, n_angles)
    detector_partition = odl.uniform_partition(
        [-30, -30], 
        [30, 30], 
        [512, 512]
        )
    geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

    # Ray transform (= forward projection).
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=ray_trafo_impl)

    forward_module  = OperatorModule(ray_trafo)
    backward_module = OperatorModule(ray_trafo.adjoint)
    sinogram :torch.Tensor = forward_module(phantom)
    start = time.time()
    reconstruction = backward_module(sinogram) #type:ignore
    end = time.time()
    return {'time':end-start}