"""Reconstruct Mayo dataset using FBP and compare to reference recon.

Note that this example requires that Mayo has been previously downloaded and is
stored in the location indicated by "mayo_dir".

In this example we only use a subset of the data for performance reasons,
there are ~32 000 projections in the full dataset.
"""
import numpy as np
import odl
from odl.contrib.datasets.ct import mayo
from time import perf_counter

# define data folders
proj_folder = odl.__path__[0] + '/../../data/LDCT-and-Projection-data/' \
             'L004/08-21-2018-10971/1.000000-Full dose projections-24362/'
img_folder = odl.__path__[0] + '/../../data/LDCT-and-Projection-data/' \
             'L004/08-21-2018-84608/1.000000-Full dose images-59704/'

# Load projection data
print("Loading projection data from {:s}".format(proj_folder))
geometry, proj_data = mayo.load_projections(proj_folder,
                                            indices=slice(16000, 19000))
# Load reconstruction data
print("Loading reference data from {:s}".format(img_folder))
recon_space, volume = mayo.load_reconstruction(img_folder)

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

# Save reconstruction in Numpy format
fbp_filename = proj_folder+'fbp_result.npy'
print("Saving reconstruction data in {:s}".format(fbp_filename))
np.save(fbp_filename, fbp_result_HU)

# Compare the computed recon to reference reconstruction (coronal slice)
ref = recon_space.element(volume)
diff = recon_space.element(volume - fbp_result_HU.asarray())

# Compare the computed recon to reference reconstruction (coronal slice)
fbp_result_HU.show('Recon (coronal)')
ref.show('Reference (coronal)')
diff.show('Diff (coronal)')

coords = [0, None, None]
fbp_result_HU.show('Recon (sagittal)', coords=coords)
ref.show('Reference (sagittal)', coords=coords)