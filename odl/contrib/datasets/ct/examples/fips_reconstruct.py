"""Example of using the FIPS datasets."""

import odl
import odl.contrib.datasets.ct.fips as fips

# Walnut example
space = odl.uniform_discr([-20, -20], [20, 20], [2296, 2296])
geometry = odl.contrib.datasets.ct.fips.walnut_geometry()

ray_transform = odl.tomo.RayTransform(space, geometry)
fbp_op = odl.tomo.fbp_op(ray_transform, filter_type='Hann')

data = fips.walnut_data()
fbp_op(data).show('Walnut FBP reconstruction', clim=[0, 0.05])

# Lotus root example
space = odl.uniform_discr([-50, -50], [50, 50], [2240, 2240])
geometry = fips.lotus_root_geometry()

ray_transform = odl.tomo.RayTransform(space, geometry)
fbp_op = odl.tomo.fbp_op(ray_transform, filter_type='Hann')

data = fips.lotus_root_data()
fbp_op(data).show('Lotus root FBP reconstruction', clim=[0, 0.1])
