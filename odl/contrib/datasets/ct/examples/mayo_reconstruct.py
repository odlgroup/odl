"""Reconstruct Mayo dataset using FBP and compare to reference recon.

Note that this example requires that projection and reconstruction data
of a CT scan from the Mayo dataset have been previously downloaded (see
[the webpage](https://ctcicblog.mayo.edu/hubcap/patient-ct-projection-data-library/))
and are stored in the locations indicated by "proj_dir" and "rec_dir".


In this example we only use a subset of the data for performance reasons.
The number of projections per patient varies in the full dataset. To get
a reconstruction of the central part of the volume, modify the indices in
the argument of the `mayo.load_projections` function accordingly.
"""
import numpy as np
import os
import odl
from odl.contrib.datasets.ct import mayo
from time import perf_counter

# replace with your local directory
mayo_dir = ''
# define projection and reconstruction data directories
# e.g. for patient L004 full dose CT scan:
proj_dir = os.path.join(
    mayo_dir, 'L004/08-21-2018-10971/1.000000-Full dose projections-24362/')
rec_dir = os.path.join(
    mayo_dir, 'L004/08-21-2018-84608/1.000000-Full dose images-59704/')

# Load projection data restricting to a central slice
print("Loading projection data from {:s}".format(proj_dir))
geometry, proj_data = mayo.load_projections(proj_dir,
                                            indices=slice(16000, 19000))
# Load reconstruction data
print("Loading reference data from {:s}".format(rec_dir))
recon_space, volume = mayo.load_reconstruction(rec_dir)

# ray transform
ray_trafo = odl.tomo.RayTransform(recon_space, geometry)

# Interpolate projection data for a flat grid
radial_dist = geometry.src_radius + geometry.det_radius
flat_proj_data = mayo.interpolate_flat_grid(proj_data,
                                            ray_trafo.range.grid,
                                            radial_dist)

# Define FBP operator
fbp = odl.tomo.fbp_op(ray_trafo, padding=True)

# Tam-Danielsson window to handle redundant data
td_window = odl.tomo.tam_danielson_window(ray_trafo, n_pi=3)

# Calculate FBP reconstruction
start = perf_counter()
fbp_result = fbp(td_window * flat_proj_data)
stop = perf_counter()
print('FBP done after {:.3f} seconds'.format(stop-start))

fbp_result_HU = (fbp_result-0.0192)/0.0192*1000

# Compare the computed recon to reference reconstruction (coronal slice)
ref = recon_space.element(volume)
diff = recon_space.element(volume - fbp_result_HU.asarray())

# Compare the computed recon to reference reconstruction (coronal slice)
fbp_result_HU.show('Recon (axial)')
ref.show('Reference (axial)')
diff.show('Diff (axial)')

coords = [0, None, None]
fbp_result_HU.show('Recon (sagittal)', coords=coords)
ref.show('Reference (sagittal)', coords=coords)
