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

# Load a subset of the projection data
data_folder = mayo_dir + '/Training Cases/L067/full_DICOM-CT-PD'
geometry, proj_data = mayo.load_projections(data_folder,
                                            indices=slice(20000, 28000))

# Reconstruction space and ray transform
space = odl.uniform_discr_frompartition(partition, dtype='float32')
ray_trafo = odl.applications.tomo.RayTransform(space, geometry)

# Define FBP operator
fbp = odl.applications.tomo.fbp_op(ray_trafo, padding=True)

# Tam-Danielsson window to handle redundant data
td_window = odl.applications.tomo.tam_danielson_window(ray_trafo, n_pi=3)

# Calculate FBP reconstruction
start = perf_counter()
fbp_result = fbp(td_window * flat_proj_data)
stop = perf_counter()
print('FBP done after {:.3f} seconds'.format(stop-start))

fbp_result_HU = (fbp_result-0.0192)/0.0192*1000

# Compare the computed recon to reference reconstruction (coronal slice)
ref = space.element(volume)
fbp_result.show('Recon (coronal)', clim=[0.7, 1.3])
ref.show('Reference (coronal)', clim=[0.7, 1.3])
(ref - fbp_result).show('Diff (coronal)', clim=[-0.1, 0.1])

# Also visualize sagittal slice (note that we only used a subset)
coords = [0, None, None]
fbp_result_HU.show('Recon (sagittal)', coords=coords)
ref.show('Reference (sagittal)', coords=coords)
