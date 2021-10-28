"""Parallel2D_vec example for checking that orientations are handled correctly.

Due to differing axis conventions between ODL and the ray transform
back-ends, a check is needed to confirm that the translation steps are
done correctly.

The back-projected data should be a blurry version of the phantom, with
all features in the correct positions, not flipped or rotated.
"""

import numpy as np
import odl

# Set back-end here (for `None` the fastest available is chosen)
impl = None
# Set a volume shift. This should not have any influence on the back-projected
# data.
shift = np.array([0.0, 25.0])

img_shape = (100, 150)
img_max_pt = np.array(img_shape, dtype=float) / 2
img_min_pt = -img_max_pt
reco_space = odl.uniform_discr(img_min_pt + shift, img_max_pt + shift,
                               img_shape, dtype='float32')
phantom = odl.phantom.indicate_proj_axis(reco_space)

assert np.allclose(reco_space.cell_sides, 1)

# Make standard parallel beam geometry with 360 angles and cast it to a
# vec geometry
geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=360)
geometry = odl.tomo.ParallelVecGeometry(
    det_shape=geometry.det_partition.shape,
    vectors=odl.tomo.parallel_2d_geom_to_astra_vecs(geometry)
)

# Test back-projection
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)
proj_data = ray_trafo(phantom)
backproj = ray_trafo.adjoint(proj_data)
backproj.show('Back-projection')
phantom.show('Phantom')
