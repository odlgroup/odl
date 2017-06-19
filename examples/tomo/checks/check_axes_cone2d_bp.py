"""Cone beam 2D example for checking that orientations are handled correctly.

Due to differing axis conventions between ODL and the ray transform
back-ends, a check is needed to confirm that the translation steps are
done correctly.

The back-projected data should be a blurry version of the phantom, with
all features in the correct positions, not flipped or rotated.

This example is best run in Spyder section-by-section (CTRL-Enter).
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

# Make fan beam geometry with 360 angles
src_radius = 500
det_radius = 1000
geometry = odl.tomo.cone_beam_geometry(reco_space, src_radius, det_radius,
                                       num_angles=360)

# Test back-projection
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)
proj_data = ray_trafo(phantom)
back_proj = ray_trafo.adjoint(proj_data)
back_proj.show('Back-projection')
phantom.show('Phantom')
