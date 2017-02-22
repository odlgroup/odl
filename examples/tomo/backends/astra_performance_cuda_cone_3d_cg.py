"""Performance example of running native ASTRA vs using ODL for reconstruction.

In this example, a 256x256x256 image is reconstructed using the Conjugate
Gradient Least Squares method on the GPU.

In general, pure ASTRA is faster than ODL since it does not need to perform any
copies and all arithmetic is performed on the GPU. Despite this, ODL is not
much slower. In this example, the overhead is about x3, depending on the
hardware used.
"""

import astra
import numpy as np
import matplotlib.pyplot as plt
import odl


# Common geometry parameters

domain_size = np.array([256, 256, 256])
n_angles = 360
det_size = 512
niter = 10

# Create reconstruction space
reco_space = odl.uniform_discr(-domain_size / 2, domain_size / 2, domain_size)

# Create geometry
apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
dpart = odl.uniform_partition([-500, -500], [500, 500],
                              [det_size, det_size])
geometry = odl.tomo.CircularConeFlatGeometry(apart, dpart,
                                             src_radius=500, det_radius=500)


data = odl.phantom.shepp_logan(reco_space, modified=True).asarray()

# --- ASTRA ---

# Define ASTRA geometry
astra_vol_geom = astra.create_vol_geom(*domain_size)
det_row_count = geometry.det_partition.shape[1]
det_col_count = geometry.det_partition.shape[0]
vec = odl.tomo.backends.astra_setup.astra_conebeam_3d_geom_to_vec(geometry)
astra_proj_geom = astra.create_proj_geom('cone_vec', det_row_count,
                                         det_col_count, vec)

# Create ASTRA projector
proj_cfg = {}
proj_cfg['type'] = 'cuda3d'
proj_cfg['VolumeGeometry'] = astra_vol_geom
proj_cfg['ProjectionGeometry'] = astra_proj_geom
proj_cfg['options'] = {}
proj_id = astra.projector3d.create(proj_cfg)

# Create sinogram
sinogram_id, sinogram = astra.create_sino3d_gpu(data,
                                                astra_proj_geom,
                                                astra_vol_geom)

# Create a data object for the reconstruction
rec_id = astra.data3d.create('-vol', astra_vol_geom)

# Set up the parameters for a reconstruction algorithm using the CUDA backend
cfg = astra.astra_dict('CGLS3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = proj_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

with odl.util.Timer('ASTRA run'):
    # Run the algorithm
    astra.algorithm.run(alg_id, niter)

# Get the result
rec = astra.data3d.get(rec_id)

# Clean up.
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(sinogram_id)
astra.projector3d.delete(proj_id)

# --- ODL ---

# Create ray transform
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Create sinogram
rhs = ray_trafo(data)

# Solve with CGLS (aka CGN)
x = reco_space.zero()
with odl.util.Timer('ODL run'):
    odl.solvers.conjugate_gradient_normal(ray_trafo, x, rhs, niter=niter)

coords = (slice(None), slice(None), 128)

# Display results for comparison
plt.figure('data')
plt.imshow(data.T[coords], origin='lower', cmap='bone')
plt.figure('ASTRA reconstruction')
plt.imshow(rec.T[coords], origin='lower', cmap='bone')
plt.figure('ODL reconstruction')
plt.imshow(x.asarray().T[coords], origin='lower', cmap='bone')
plt.show()
