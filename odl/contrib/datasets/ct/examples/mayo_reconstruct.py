"""Reconstruct Mayo dataset using FBP and compare to reference recon.

Note that this example requires that Mayo has been previously downloaded and is
stored in the location indicated by "mayo_dir".

In this example we only use a subset of the data for performance reasons,
there are ~48 000 projections in the full dataset.
"""

import odl
from odl.contrib.datasets.ct import mayo

mayo_dir = ''  # replace with your local folder

# Load reference reconstruction
volume_folder = mayo_dir + '/Training Cases/L067/full_1mm_sharp'
partition, volume = mayo.load_reconstruction(volume_folder)

# Load a subset of the projection data
data_folder = mayo_dir + '/Training Cases/L067/full_DICOM-CT-PD'
geometry, proj_data = mayo.load_projections(data_folder,
                                            indices=slice(20000, 28000))

# Reconstruction space and ray transform
space = odl.uniform_discr_frompartition(partition, dtype='float32')
ray_trafo = odl.tomo.RayTransform(space, geometry)

# Define FBP operator
fbp = odl.tomo.fbp_op(ray_trafo, padding=True)

# Tam-Danielsson window to handle redundant data
td_window = odl.tomo.tam_danielson_window(ray_trafo, n_pi=3)

# Calculate FBP reconstruction
fbp_result = fbp(td_window * proj_data)

# Compare the computed recon to reference reconstruction (coronal slice)
ref = space.element(volume)
fbp_result.show('Recon (coronal)', clim=[0.7, 1.3])
ref.show('Reference (coronal)', clim=[0.7, 1.3])
(ref - fbp_result).show('Diff (coronal)', clim=[-0.1, 0.1])

# Also visualize sagittal slice (note that we only used a subset)
coords = [0, None, None]
fbp_result.show('Recon (sagittal)', clim=[0.7, 1.3], coords=coords)
ref.show('Reference (sagittal)', clim=[0.7, 1.3], coords=coords)
(ref - fbp_result).show('Diff (sagittal)', clim=[-0.1, 0.1], coords=coords)
