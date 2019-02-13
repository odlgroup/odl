"""Example of using the Elekta XVI geometry.

In this example we create an Elekta XVI geometry and use it to create some
artificial data and reconstruct it using ART, specifically kaczmarz' method.
"""

import numpy as np
import odl
from odl.contrib import tomo

# The number of subsets to use in kaczmarz' method
subsets = 20

# Get default geometry and space
geometry = tomo.elekta_xvi_geometry()
space = tomo.elekta_xvi_space(shape=(112, 112, 112))

# Create sub-geometries using geometry indexing
step = int(np.ceil(geometry.angles.size / subsets))
geometries = [geometry[i * step:(i + 1) * step] for i in range(subsets)]

# Create ray transform
ray_transforms = [odl.tomo.RayTransform(space, geom, use_cache=False)
                  for geom in geometries]

# Create simple phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create artificial data
projections = [rt(phantom) for rt in ray_transforms]

# Reconstruct using kaczmarz
callback = odl.solvers.CallbackShow()
x = space.zero()
reconstruction = odl.solvers.kaczmarz(ray_transforms, x, projections, 20,
                                      omega=0.005,
                                      callback=callback, callback_loop='inner')
