"""Example using FBP in helical 3D geometry using `fbp_op`.

Note that the FBP is only approximate in this geometry, but still gives a
decent reconstruction that can be used as an initial guess in more complex
methods.

In helical geometries, the data are in general over-sampled which causes
streak artefacts and a wrong scaling. This can be reduced using a
Tam-Danielson window.
"""

import numpy as np
import odl


# --- Set up geometry of the problem --- #


# Reconstruction space: discretized functions on the cube
# [-20, 20]^3  with 200 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[200, 200, 200],
    dtype='float32')

# Create helical geometry
geometry = odl.tomo.helical_geometry(space,
                                     src_radius=100, det_radius=100,
                                     num_turns=7.5, num_angles=1000)

# --- Create Filtered Back-projection (FBP) operator --- #


# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(space, geometry)

# Unwindowed fbp
# We select a Hamming filter, and only use the lowest 80% of frequencies to
# avoid high frequency noise.
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hamming', frequency_scaling=0.8)

# Create Tam-Danielson window to improve result
windowed_fbp = fbp * odl.tomo.tam_danielson_window(ray_trafo)


# --- Show some examples --- #


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Calculate FBP reconstructions, once without window, once with window
fbp_reconstruction = fbp(proj_data)
w_fbp_reconstruction = windowed_fbp(proj_data)

# Show a slice of phantom, projections, and reconstruction
phantom.show(title='Phantom',
             coords=[0, None, None], clim=[-0.1, 1.1])
proj_data.show(title='Simulated Data (Sinogram)')
fbp_reconstruction.show(title='Filtered Back-projection',
                        coords=[0, None, None], clim=[-0.1, 1.1])
w_fbp_reconstruction.show(title='Windowed Filtered back-projection',
                          coords=[0, None, None], clim=[-0.1, 1.1],
                          force_show=True)
