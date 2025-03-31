"""Cone2D_vec example for checking that orientations are handled correctly.

Due to differing axis conventions between ODL and the ray transform
back-ends, a check is needed to confirm that the translation steps are
done correctly.

Both pairs of plots of ODL projections and NumPy axis sums should look
the same in the sense that they should show the same features in the
right arrangement (not flipped, rotated, etc.).

This example is best run in Spyder section-by-section (CTRL-Enter).
"""

# %% Set up the things that never change

import matplotlib.pyplot as plt
import numpy as np
import odl

# Set back-end here (for `None` the fastest available is chosen)
impl = None
# Set a volume shift. This should move the projections in the same direction.
shift = np.array([0.0, 25.0])

img_shape = (100, 150)
img_max_pt = np.array(img_shape, dtype=float) / 2
img_min_pt = -img_max_pt
reco_space = odl.uniform_discr(img_min_pt + shift, img_max_pt + shift,
                               img_shape, dtype='float32')
phantom = odl.phantom.indicate_proj_axis(reco_space)

assert np.allclose(reco_space.cell_sides, 1)

# Generate ASTRA vectors, but in the ODL geometry convention.
# We use 0, 90, 180 and 270 degrees as angles, with the detector starting
# with reference point (0, 1000) and axis (1, 0), rotating counter-clockwise.
# The source points are chosen opposite to the detector at radius 500.
# The detector `u` vector from pixel 0 to pixel 1 is equal to the detector
# axis, since we choose the pixel size to be equal to 1.
src_radius = 500
det_radius = 1000
det_refpoints = np.array([(0, 1), (-1, 0), (0, -1), (1, 0)]) * det_radius
src_points = -det_refpoints / det_radius * src_radius
det_axes = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])
vectors = np.empty((4, 6))
vectors[:, 0:2] = src_points
vectors[:, 2:4] = det_refpoints
vectors[:, 4:6] = det_axes

# Choose enough pixels to cover the object projections
fan_angle = np.arctan(img_max_pt[1] / src_radius)
det_size = np.floor(2 * (src_radius + det_radius) * np.sin(fan_angle))
det_shape = int(det_size)

# Sum manually using Numpy
sum_along_x = np.sum(phantom, axis=0)
sum_along_y = np.sum(phantom, axis=1)


# %% Test forward projection along y axis


geometry = odl.tomo.ConeVecGeometry(det_shape, vectors)
# Check initial configuration
assert np.allclose(geometry.det_axes(0), [[1, 0]])
assert np.allclose(geometry.det_refpoint(0), [[0, det_radius]])

# Create projections
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)
proj_data = ray_trafo(phantom)

# Axis in this image is x. This corresponds to 0 degrees or index 0.
proj_data.show(
    indices=[0, None],
    title='Projection at 0 degrees ~ Sum along y axis'
)
fig, ax = plt.subplots()
ax.plot(sum_along_y)
ax.set(xlabel="x", title='Sum along y axis')
plt.show()
# Check axes in geometry
axis_sum_y = geometry.det_axis(0)
assert np.allclose(axis_sum_y, [1, 0])


# %% Test forward projection along x axis


# Axis in this image is y. This corresponds to 90 degrees or index 1.
proj_data.show(indices=[1, None],
               title='Projection at 90 degrees ~ Sum along x axis')
fig, ax = plt.subplots()
ax.plot(sum_along_x)
ax.set_xlabel('y')
plt.title('Sum along x axis')
plt.show()
# Check axes in geometry
axis_sum_x = geometry.det_axis(1)
assert np.allclose(axis_sum_x, [0, 1])
