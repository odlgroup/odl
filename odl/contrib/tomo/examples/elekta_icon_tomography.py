"""Example of using the Elekta Icon geometry.

In this example we create an Elekta Icon geometry and use it to create some
artificial data and reconstruct it using backprojection.
"""

import odl
from odl.contrib import tomo

# Get default geometry and space
geometry = tomo.elekta_icon_geometry()
space = tomo.elekta_icon_space()

# Create ray transform
ray_transform = odl.tomo.RayTransform(space, geometry,
                                      use_cache=False)

# Get default FDK reconstruction
recon_op = tomo.elekta_icon_reconstruction(ray_transform,
                                           parker_weighting=False)

# Create simplified-phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create artificial data
projection = ray_transform(phantom)

# Reconstruct the artificial data
reconstruction = recon_op(projection)
