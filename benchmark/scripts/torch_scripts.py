from typing import Dict

import numpy as np
import torch
import torch.random

import odl

    
def complex_mse_loss(output:torch.Tensor, target:torch.Tensor):
    return (0.5*(output - target)**2).mean(dtype=torch.complex64)

def mri_mlem_adam(
    parameters:Dict,
    dimension   :    int,
    n_points  :      int,
    max_iterations : int
    ):
    subsampling : float = parameters['subsampling']
    device_name : str   = parameters['device_name']
    learning_rate: float = parameters['learning_rate']
    beta1:float = parameters['beta1']
    beta2:float = parameters['beta2']
    eps   = parameters['eps']

    space = odl.uniform_discr(
            [0 for _ in range(dimension)], 
            [n_points for _ in range(dimension)], 
            [n_points for _ in range(dimension)]
        )

    phantom = odl.phantom.shepp_logan(space, modified=True)
    phantom = torch.from_numpy(phantom.asarray()).unsqueeze(0).unsqueeze(0).to(device_name)

    x = torch.zeros(
        size = phantom.size(), 
        device = device_name, 
        requires_grad=True
        )

    optimiser = torch.optim.Adam( 
        [x], 
        lr = learning_rate, 
        betas= (beta1, beta2),
        eps = eps
        ) 
    
    class FwdOp(torch.nn.Module):
        def __init__(
                self, 
                phantom:torch.Tensor, 
                subsampling:float,
                device
                ):
            super(FwdOp, self).__init__()
            self.sampling_mask = torch.rand(phantom.size(), device=device) < subsampling

        def forward(self, input_tensor:torch.Tensor):
            return self.sampling_mask * torch.fft.fftn(input_tensor)
    
    mri_op = FwdOp(phantom, subsampling, device_name)
    
    noisy_data  = mri_op(phantom) + torch.normal(
        mean=torch.zeros(phantom.size()), 
        std=torch.ones(phantom.size())).to(device_name) * 0.1

    for _ in range(max_iterations):
        optimiser.zero_grad()
        current_data = mri_op(x)
        loss = complex_mse_loss(current_data, noisy_data)
        loss.mean().backward()
        optimiser.step()

    return np.linalg.norm(
        phantom.detach().cpu().numpy() - x.data.detach().cpu().numpy()
        )