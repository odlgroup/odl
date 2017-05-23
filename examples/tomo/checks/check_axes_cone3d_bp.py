"""Cone beam 3D example for checking that orientations are handled correctly.

Due to differing axis conventions between ODL and the ray transform
back-ends, a check is needed to confirm that the translation steps are
done correctly.

The back-projected data should be a blurry version of the phantom, with
all features in the correct positions, not flipped or rotated.
"""

import numpy as np
import odl


# --- Set up the things that never change --- #


vol_shape = (100, 150, 200)
vol_max_pt = np.array(vol_shape, dtype=float) / 2
vol_min_pt = -vol_max_pt
reco_space = odl.uniform_discr(vol_min_pt, vol_max_pt, vol_shape,
                               dtype='float32')
phantom = odl.phantom.indicate_proj_axis(reco_space)

assert np.allclose(reco_space.cell_sides, 1)

grid = odl.RectGrid(np.linspace(0, 2 * np.pi, 360, endpoint=False))
angle_partition = odl.uniform_partition_fromgrid(grid)

# Make detector large enough to cover the object
src_radius = 500
det_radius = 1000
opening_angle = np.arctan(vol_max_pt[2] / src_radius)
det_size = np.floor(1.1 * (src_radius + det_radius) * np.sin(opening_angle))
det_shape = (int(det_size),) * 2
det_max_pt = np.array([det_size / 2, det_size / 2])
det_min_pt = -det_max_pt
detector_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

assert np.allclose(detector_partition.cell_sides, 1)


# --- Test case 1: axis = [0, 0, 1] --- #


geometry = odl.tomo.CircularConeFlatGeometry(
    angle_partition, detector_partition, src_radius, det_radius,
    axis=[0, 0, 1])

# Create projections and back-projection
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
proj_data = ray_trafo(phantom)
backproj = ray_trafo.adjoint(proj_data)
backproj.show('Backprojection, axis = [0, 0, 1], middle z slice',
              indices=[slice(None), slice(None), 100])
phantom.show('Phantom, middle z slice',
             indices=[slice(None), slice(None), 100])


# --- Test case 2: axis = [0, 1, 0] --- #


geometry = odl.tomo.CircularConeFlatGeometry(
    angle_partition, detector_partition, src_radius, det_radius,
    axis=[0, 1, 0])

# Create projections and back-projection
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
proj_data = ray_trafo(phantom)
backproj = ray_trafo.adjoint(proj_data)
backproj.show('Backprojection, axis = [0, 1, 0], middle y slice',
              indices=[slice(None), 75, slice(None)])
phantom.show('Phantom, middle y slice',
             indices=[slice(None), 75, slice(None)])


# --- Test case 3: axis = [1, 0, 0] --- #


geometry = odl.tomo.CircularConeFlatGeometry(
    angle_partition, detector_partition, src_radius, det_radius,
    axis=[1, 0, 0])

# Create projections and back-projection
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
proj_data = ray_trafo(phantom)
backproj = ray_trafo.adjoint(proj_data)
backproj.show('Backprojection, axis = [1, 0, 0], almost max x slice',
              indices=[95, slice(None), slice(None)])
phantom.show('Phantom, almost max x slice',
             indices=[95, slice(None), slice(None)])
