"""Example of using the Elekta XVI geometry.

In this example we create an Elekta XVI geometry and use it to create some
artificial data and reconstruct it using filtered backprojection.

Note that this is a 3d dataset and requires some memory to run.
"""

import odl
from odl.contrib import tomo

# Get default geometry and space
geometry = tomo.elekta_xvi_geometry()
space = tomo.elekta_xvi_space(shape=(112, 112, 112))

# Create ray transform
ray_transform = odl.tomo.RayTransform(space, geometry, use_cache=False)

# Get default FDK reconstruction operator
recon_op = tomo.elekta_xvi_fbp(ray_transform)

# Create simplified phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create artificial data
projections = ray_transform(phantom)

# Reconstruct the artificial data
reconstruction = recon_op(projections)

# Display the results
phantom.show('phantom xz', coords=[None, 0, None])
reconstruction.show('reconstruction xz', coords=[None, 0, None])

phantom.show('phantom xy', coords=[None, None, 0])
reconstruction.show('reconstruction xy', coords=[None, None, 0])

phantom.show('phantom yz', coords=[0, None, None])
reconstruction.show('reconstruction yz', coords=[0, None, None])

projections.show('projections')
