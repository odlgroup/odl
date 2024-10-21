import torch
import numpy as np

import odl

from odl.contrib.torch.new_operator import OperatorModule

import matplotlib.pyplot as plt

if __name__ == '__main__':
    device_name = 'cuda:0'
    ### Define input tensor
    dimension = 3 
    n_points  = 64
    space = odl.uniform_discr(
            [-20 for _ in range(dimension)], 
            [ 20 for _ in range(dimension)], 
            [n_points for _ in range(dimension)],
            impl='pytorch', torch_device=device_name
            )

    odl_phantom = odl.phantom.shepp_logan(space, modified=True)
    phantom : torch.Tensor = odl_phantom.asarray().unsqueeze(0).unsqueeze(0).to(device_name)
    plt.matshow(phantom[0,0,32].detach().cpu())
    plt.savefig('phantom')
    plt.close()

    # <!> enforce float32 conversion, rather than float64
    phantom = phantom.to(dtype=torch.float32)
    # <!> make tensor contiguous from creation
    phantom = phantom.contiguous()
    # <!> for the example, input_tensor.requires_grad == True
    phantom.requires_grad_()
    # Make a 3d single-axis parallel beam geometry with flat detector
    # Angles: uniformly spaced, n = 180, min = 0, max = pi
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 32)
    detector_partition = odl.uniform_partition([-30] * 2, [30] * 2, [100] * 2)
    geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

    # Ray transform (= forward projection).
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda_pytorch')

    forward_module  = OperatorModule(ray_trafo)
    backward_module = OperatorModule(ray_trafo.adjoint)
    sinogram :torch.Tensor = forward_module(phantom) #type:ignore
    
    x = torch.zeros(
        size = phantom.size(), 
        device = device_name, 
        requires_grad=True
        )

    optimiser = torch.optim.Adam( #type:ignore
        [x], 
        lr = 1e-3
        ) 
    
    noisy_data = forward_module(phantom)
    mse_loss =torch.nn.MSELoss()

    for _ in range(100):
        optimiser.zero_grad()
        current_data = forward_module(x)
        loss = mse_loss(current_data, noisy_data)
        loss.mean().backward()
        optimiser.step()
    
    plt.matshow(x[0,0,32].detach().cpu())
    plt.savefig('optimised')
    plt.close()

