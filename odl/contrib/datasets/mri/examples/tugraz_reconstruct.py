"""Example of using the TU Graz datasets."""

import odl.contrib.datasets.mri.tugraz as tugraz
from odl.util.testutils import run_doctests

# 4-channel head example
data = tugraz.mri_head_data_4_channel()
pseudo_inv = tugraz.mri_head_reco_op_4_channel()

reconstruction = pseudo_inv(data)
reconstruction.show(clim=[0, 1])

# 32-channel head example
data = tugraz.mri_head_data_32_channel()
pseudo_inv = tugraz.mri_head_reco_op_32_channel()

reconstruction = pseudo_inv(data)
reconstruction.show(clim=[0, 1])

# 8-channel knee example
data = tugraz.mri_knee_data_8_channel()
pseudo_inv = tugraz.mri_knee_reco_op_8_channel()

reconstruction = pseudo_inv(data)
reconstruction.show(clim=[0, 1])

# Run doctests
run_doctests()
