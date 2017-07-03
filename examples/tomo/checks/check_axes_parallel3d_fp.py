"""Parallel 3D example for checking that orientations are handled correctly.

Due to differing axis conventions between ODL and the ray transform
back-ends, a check is needed to confirm that the translation steps are
done correctly.

All pairs of plots of ODL projections and NumPy axis sums should look
the same in the sense that they should show the same features in the
right arrangement (not flipped, rotated, etc.).
"""

# %% Set up the things that never change

import matplotlib.pyplot as plt
import numpy as np
import odl

# Set back-end here (for `None` the fastest available is chosen)
impl = None
# Set a volume shift. This should move the projections in the same direction.
shift = np.array([0.0, 25.0, 0.0])

vol_shape = (100, 150, 200)
vol_max_pt = np.array(vol_shape, dtype=float) / 2
vol_min_pt = -vol_max_pt
reco_space = odl.uniform_discr(vol_min_pt + shift, vol_max_pt + shift,
                               vol_shape, dtype='float32')
phantom = odl.phantom.indicate_proj_axis(reco_space)

assert np.allclose(reco_space.cell_sides, 1)

# Check projections at 0, 90, 180 and 270 degrees
grid = odl.RectGrid([0, np.pi / 2, np.pi, 3 * np.pi / 2])
angle_partition = odl.uniform_partition_fromgrid(grid)

# Make detector large enough to cover the object
det_size = np.floor(1.1 * np.sqrt(np.sum(np.square(vol_shape))))
det_shape = (int(det_size), int(det_size))
det_max_pt = np.array([det_size / 2, det_size / 2])
det_min_pt = -det_max_pt
detector_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

assert np.allclose(detector_partition.cell_sides, 1)

# Sum manually using Numpy
sum_along_x = np.sum(phantom, axis=0)
sum_along_y = np.sum(phantom, axis=1)
sum_along_z = np.sum(phantom, axis=2)


# %% Test case 1: axis = [0, 0, 1] -- setup


geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=[0, 0, 1])
# Check initial configuration
assert np.allclose(geometry.det_axes_init[0], [1, 0, 0])
assert np.allclose(geometry.det_axes_init[1], [0, 0, 1])
assert np.allclose(geometry.det_pos_init, [0, 1, 0])

# Create projections
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)
proj_data = ray_trafo(phantom)


# %% axis = [0, 0, 1], projection along y axis


# Axes in this image are (x, z). This corresponds to
# axis = [0, 0, 1], 0 degrees
proj_data.show(indices=[0, None, None],
               title='Projection at 0 degrees, axis [0, 0, 1], u = x, v = z')
fig, ax = plt.subplots()
ax.imshow(sum_along_y.T, cmap='bone', origin='lower')
ax.set_xlabel('x')
ax.set_ylabel('z')
plt.title('Sum along y axis')
plt.show()
# Check axes in geometry
axes_sum_y = geometry.det_axes(np.deg2rad(0))
assert np.allclose(axes_sum_y[0], [1, 0, 0])
assert np.allclose(axes_sum_y[1], [0, 0, 1])


# %% axis = [0, 0, 1], projection along x axis


# Axes in this image are (y, z). This corresponds to
# axis = [0, 0, 1], 90 degrees
proj_data.show(indices=[1, None, None],
               title='Projection at 90 degrees, axis [0, 0, 1], u = y, v = z')
fig, ax = plt.subplots()
ax.imshow(sum_along_x.T, cmap='bone', origin='lower')
ax.set_xlabel('y')
ax.set_ylabel('z')
plt.title('Sum along x axis')
plt.show()
# Check axes in geometry
axes_sum_x = geometry.det_axes(np.deg2rad(90))
assert np.allclose(axes_sum_x[0], [0, 1, 0])
assert np.allclose(axes_sum_x[1], [0, 0, 1])


# %% Test case 2: axis = [0, 1, 0] -- setup


geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=[0, 1, 0])
# Check initial configuration
assert np.allclose(geometry.det_axes_init[0], [1, 0, 0])
assert np.allclose(geometry.det_axes_init[1], [0, 1, 0])
assert np.allclose(geometry.det_pos_init, [0, 0, -1])

# Create projections
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)
proj_data = ray_trafo(phantom)


# %% axis = [0, 1, 0], projection along z axis


# Axes in this image are (x, y). This corresponds to:
# axis = [0, 1, 0], 0 degrees
proj_data.show(indices=[0, None, None],
               title='Projection at 0 degrees, axis [0, 1, 0], u = x, v = y')
fig, ax = plt.subplots()
ax.imshow(sum_along_z.T, cmap='bone', origin='lower')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('Sum along z axis')
plt.show()
# Check geometry axes
axes_sum_z = geometry.det_axes(np.deg2rad(0))
assert np.allclose(axes_sum_z[0], [1, 0, 0])
assert np.allclose(axes_sum_z[1], [0, 1, 0])


# %% axis = [0, 1, 0], projection along x axis


# Axes in this image are (z, y). This corresponds to
# axis = [0, 1, 0], 270 degrees
proj_data.show(indices=[3, None, None],
               title='Projection at 270 degrees, axis [0, 1, 0], u = z, v = y')
fig, ax = plt.subplots()
ax.imshow(sum_along_x, cmap='bone', origin='lower')
ax.set_xlabel('z')
ax.set_ylabel('y')
plt.title('Sum along x axis, transposed')
plt.show()
# Check geometry axes
axes_sum_x_T = geometry.det_axes(np.deg2rad(270))
assert np.allclose(axes_sum_x_T[0], [0, 0, 1])
assert np.allclose(axes_sum_x_T[1], [0, 1, 0])


# %% Test case 3: axis = [1, 0, 0] -- setup


geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=[1, 0, 0])
# Check initial configuration
assert np.allclose(geometry.det_axes_init[0], [0, 0, -1])
assert np.allclose(geometry.det_axes_init[1], [1, 0, 0])
assert np.allclose(geometry.det_pos_init, [0, 1, 0])

# Create projections
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)
proj_data = ray_trafo(phantom)


# %% axis = [1, 0, 0], projection along z axis


# Axes in this image are (y, x). This corresponds to
# axis = [1, 0, 0], 90 degrees
proj_data.show(indices=[1, None, None],
               title='Projection at 90 degrees, axis [1, 0, 0], u = y, v = x')
fig, ax = plt.subplots()
ax.imshow(sum_along_z, cmap='bone', origin='lower')
ax.set_xlabel('y')
ax.set_ylabel('x')
plt.title('Sum along z axis, transposed')
plt.show()
# Check geometry axes
axes_sum_z_T = geometry.det_axes(np.deg2rad(90))
assert np.allclose(axes_sum_z_T[0], [0, 1, 0])
assert np.allclose(axes_sum_z_T[1], [1, 0, 0])


# %% axis = [1, 0, 0], projection along y axis


# Axes in this image are (z, x). This corresponds to
# axis = [1, 0, 0], 180 degrees
proj_data.show(indices=[2, None, None],
               title='Projection at 180 degrees, axis [1, 0, 0], u = z, v = x')
fig, ax = plt.subplots()
ax.imshow(sum_along_y, cmap='bone', origin='lower')
ax.set_xlabel('z')
ax.set_ylabel('x')
plt.title('Sum along y axis, transposed')
plt.show()
# Check geometry axes
axes_sum_y = geometry.det_axes(np.deg2rad(180))
assert np.allclose(axes_sum_y[0], [0, 0, 1])
assert np.allclose(axes_sum_y[1], [1, 0, 0])
