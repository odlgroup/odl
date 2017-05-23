"""Parallel 2D example for checking that orientations are handled correctly.

Due to differing axis conventions between ODL and the ray transform
back-ends, a check is needed to confirm that the translation steps are
done correctly.

Both pairs of plots of ODL projections and NumPy axis sums should look
the same in the sense that they should show the same features in the
right arrangement (not flipped, rotated, etc.).
"""

import matplotlib.pyplot as plt
import numpy as np
import odl


# --- Set up the things that never change --- #


# Get a slice through the indiceat_proj_axis phantom
phantom_3d_shape = (100, 150, 200)
phantom_3d_max_pt = np.array(phantom_3d_shape, dtype=float) / 2
phantom_3d_min_pt = -phantom_3d_max_pt
phantom_3d_space = odl.uniform_discr(
    phantom_3d_min_pt, phantom_3d_max_pt, phantom_3d_shape, dtype='float32')
phantom_3d = odl.phantom.indicate_proj_axis(phantom_3d_space)

phantom_arr = phantom_3d.asarray()[:, :, 100]

img_shape = (100, 150)
img_max_pt = np.array(img_shape, dtype=float) / 2
img_min_pt = -img_max_pt
shift = (-25, 0)  # this should shift the projections in the same direction
reco_space = odl.uniform_discr(img_min_pt + shift, img_max_pt + shift,
                               img_shape, dtype='float32')
phantom = reco_space.element(phantom_arr)

assert np.allclose(reco_space.cell_sides, 1)

# Check projections at 0, 90, 180 and 270 degrees
grid = odl.RectGrid([0, np.pi / 2, np.pi, 3 * np.pi / 2])
angle_partition = odl.uniform_partition_fromgrid(grid)

# Make detector large enough to cover the object
det_size = np.floor(1.1 * np.sqrt(np.sum(np.square(img_shape))))
det_shape = int(det_size)
det_max_pt = det_size / 2
det_min_pt = -det_max_pt
detector_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

assert np.allclose(detector_partition.cell_sides, 1)

# Sum manually using Numpy
sum_along_x = np.sum(phantom.asarray(), axis=0)
sum_along_y = np.sum(phantom.asarray(), axis=1)


# --- Test --- #


geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
# Check initial configuration
assert np.allclose(geometry.det_axis_init, [1, 0])
assert np.allclose(geometry.det_pos_init, [0, 1])

# Create projections
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')
proj_data = ray_trafo(phantom)

# Axis in this image is x. This corresponds to 0 degrees.
proj_data.show(indices=[0, slice(None)],
               title='Projection at 0 degrees ~ Sum along y axis')
fig, ax = plt.subplots()
ax.plot(sum_along_y)
ax.set_xlabel('x')
plt.title('Sum along y axis')
plt.show()
# Check axes in geometry
axis_sum_y = geometry.det_axis(np.deg2rad(0))
assert np.allclose(axis_sum_y, [1, 0])

# Axis in this image is y. This corresponds to 90 degrees.
proj_data.show(indices=[1, slice(None)],
               title='Projection at 90 degrees ~ Sum along x axis')
fig, ax = plt.subplots()
ax.plot(sum_along_x)
ax.set_xlabel('y')
plt.title('Sum along x axis')
plt.show()
# Check axes in geometry
axis_sum_x = geometry.det_axis(np.deg2rad(90))
assert np.allclose(axis_sum_x, [0, 1])
