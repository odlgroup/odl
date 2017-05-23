"""Parallel 2D example for checking that orientations are handled correctly.

Due to differing axis conventions between ODL and the ray transform
back-ends, a check is needed to confirm that the translation steps are
done correctly.

The back-projected data should be a blurry version of the phantom, with
all features in the correct positions, not flipped or rotated.
"""

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
shift = (0, 25)  # this should shift the projections in the same direction
reco_space = odl.uniform_discr(img_min_pt + shift, img_max_pt + shift,
                               img_shape, dtype='float32')
phantom = reco_space.element(phantom_arr)

assert np.allclose(reco_space.cell_sides, 1)

# Take 1 degree increments, full angular range
grid = odl.RectGrid(np.linspace(0, 2 * np.pi, 360, endpoint=False))
angle_partition = odl.uniform_partition_fromgrid(grid)

# Make detector large enough to cover the object
det_size = np.floor(1.1 * np.sqrt(np.sum(np.square(img_shape))))
det_shape = int(det_size)
det_max_pt = det_size / 2
det_min_pt = -det_max_pt
detector_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

assert np.allclose(detector_partition.cell_sides, 1)


# --- Test back-projection --- #


geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')
proj_data = ray_trafo(phantom)
back_proj = ray_trafo.adjoint(proj_data)
back_proj.show('Back-projection')
phantom.show('Phantom')
