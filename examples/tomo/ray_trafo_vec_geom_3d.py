"""Example using the ray transform a custom vector geometry.

We manually build a "circle plus line trajectory" (CLT) geometry by
extracting the vectors from a circular geometry and extending it by
vertical shifts, starting at the initial position.
"""

import numpy as np
import odl

# Reconstruction space: discretized functions on the cube [-20, 20]^3
# with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[300, 300, 300],
    dtype='float32')

# First part of the geometry: a 3D single-axis parallel beam geometry with
# flat detector
# Angles: uniformly spaced, n = 180, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 180)
# Detector: uniformly sampled, n = (512, 512), min = (-30, -30), max = (30, 30)
detector_partition = odl.uniform_partition([-30, -30], [30, 30], [512, 512])
circle_geometry = odl.tomo.CircularConeFlatGeometry(
    angle_partition, detector_partition, src_radius=1000, det_radius=100,
    axis=[1, 0, 0])

circle_vecs = odl.tomo.astra_conebeam_3d_geom_to_vec(circle_geometry)

# Cover the whole volume vertically, somewhat undersampled though
vert_shift_min = -22
vert_shift_max = 22
num_shifts = 180
vert_shifts = np.linspace(vert_shift_min, vert_shift_max, num=num_shifts)
inital_vecs = circle_vecs[0]

# Start from the initial position of the circle vectors and add the vertical
# shifts to the columns 2 and 5 (source and detector z positions)
line_vecs = np.repeat(circle_vecs[0][None, :], num_shifts, axis=0)
line_vecs[:, 2] += vert_shifts
line_vecs[:, 5] += vert_shifts

# Build the composed geometry and the corresponding ray transform
# (= forward projection)
composed_vecs = np.vstack([circle_vecs, line_vecs])
composed_geom = odl.tomo.ConeVecGeometry(detector_partition.shape,
                                         composed_vecs)

ray_trafo = odl.tomo.RayTransform(reco_space, composed_geom)

# Create a Shepp-Logan phantom (modified version) and projection data
phantom = odl.phantom.shepp_logan(reco_space, True)
proj_data = ray_trafo(phantom)

# Back-projection can be done by simply calling the adjoint operator on the
# projection data (or any element in the projection space).
backproj = ray_trafo.adjoint(proj_data)

# Show the slice z=0 of phantom and backprojection, as well as a projection
# image at theta=0 and a sinogram at v=0 (middle detector row)
phantom.show(coords=[None, None, 0], title='Phantom, middle z slice')
backproj.show(coords=[None, None, 0], title='Back-projection, middle z slice')
proj_data.show(indices=[0, slice(None), slice(None)],
               title='Projection 0 (circle start)')
proj_data.show(indices=[45, slice(None), slice(None)],
               title='Projection 45 (circle 1/4)')
proj_data.show(indices=[90, slice(None), slice(None)],
               title='Projection 90 (circle 1/2)')
proj_data.show(indices=[135, slice(None), slice(None)],
               title='Projection 135 (circle 3/4)')
proj_data.show(indices=[179, slice(None), slice(None)],
               title='Projection 179 (circle end)')
proj_data.show(indices=[180, slice(None), slice(None)],
               title='Projection 180 (line start)')
proj_data.show(indices=[270, slice(None), slice(None)],
               title='Projection 270 (line middle)')
proj_data.show(indices=[359, slice(None), slice(None)],
               title='Projection 359 (line end)')
proj_data.show(coords=[None, None, 0], title='Sinogram, middle slice')
