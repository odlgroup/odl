"""
Example using a filtered back-projection (FBP) in fan beam using `fbp_op`.

Note that the FBP is only approximate in this geometry, but still gives a
decent reconstruction that can be used as an initial guess in more complicated
methods.
"""

import numpy as np
import odl
import time

# --- Set up geometry of the problem --- #

device = 'cuda:0'

reco_space = odl.uniform_discr(
min_pt=[-20, -20], max_pt=[20, 20], shape=[300,300],
dtype='float32', impl='pytorch', device=device)

# Make a circular cone beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = 512, min = -60, max = 60
detector_partition = odl.uniform_partition(-60, 60, 512)
# Geometry with large fan angle
geometry = odl.applications.tomo.FanBeamGeometry(
    angle_partition, detector_partition, src_radius=40, det_radius=40)


# --- Create Filtered Back-projection (FBP) operator --- #


# Ray transform (= forward projection).
ray_trafo = odl.applications.tomo.RayTransform(reco_space, geometry)

# Create FBP operator using utility function
# We select a Hann filter, and only use the lowest 80% of frequencies to avoid
# high frequency noise.
fbp = odl.applications.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

# Wrapping the FBP inside a torch.nn.Module
from odl.contrib.torch import OperatorModule
fbp = OperatorModule(fbp)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.core.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Transform the projection_data into a pytorch_tensor and one extra channel dimension:
proj_data = proj_data.asarray().unsqueeze(0)

# We can now define a dummy neural network that will be trained to denoise a single sample (sic)
import torch
network = torch.nn.Sequential(
    torch.nn.Conv2d(1,4,5,padding=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4,1,5,padding=2),
    torch.nn.ReLU()
    ).to(device=device)

optimiser = torch.optim.Adam(network.parameters())

loss_function = torch.nn.MSELoss()

target_reconstruction = fbp(proj_data)
# we corruput the input data with some random noise
noisy_data  = proj_data + (0.1**0.5)*torch.randn(1,360,512, device=device)

for i in range(100):
    optimiser.zero_grad()
    input_reconstruction = network(fbp(noisy_data))
    loss_value = loss_function(input_reconstruction, target_reconstruction)
    loss_value.backward()
    optimiser.step()
    print(f'MSE loss value: {loss_value.item():.5f}')

import matplotlib.pyplot as plt
plt.matshow(fbp(noisy_data)[0].detach().cpu())
plt.savefig('input_reconstruction')
plt.clf()

plt.matshow(input_reconstruction[0].detach().cpu())
plt.savefig('improved_reconstruction')
plt.clf()

plt.matshow(target_reconstruction[0].detach().cpu())
plt.savefig('target_reconstruction')
plt.clf()



