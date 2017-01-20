"""Performance example of running native ASTRA vs using ODL for reconstruction.

In this example, a 512x512 image is reconstructed using the Conjugate Gradient
Least Squares method on the CPU.

In general, ASTRA is faster than ODL since it does not need to perform any
copies and all arithmetic is performed in place. Despite this, ODL is not much
slower. In this example, the overhead is about x1.2, depending on the
hardware used.
"""

import astra
import numpy as np
import matplotlib.pyplot as plt
import scipy
import odl


# Common geometry parameters

domain_size = np.array([512, 512])
n_angles = 180
det_size = 362
niter = 20
data = np.rot90(scipy.misc.ascent().astype('float'), -1)


# --- ASTRA ---


# Define ASTRA geometry
vol_geom = astra.create_vol_geom(domain_size[0], domain_size[1])
proj_geom = astra.create_proj_geom('parallel',
                                   np.linalg.norm(domain_size) / det_size,
                                   det_size,
                                   np.linspace(0, np.pi, n_angles))

# Create ASTRA projector
proj_id = astra.create_projector('line', proj_geom, vol_geom)

# Create sinogram
sinogram_id, sinogram = astra.create_sino(data, proj_id)

# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the CPU backend
cfg = astra.astra_dict('CGLS')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = proj_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

with odl.util.Timer('ASTRA run'):
    # Run the algorithm
    astra.algorithm.run(alg_id, niter)

# Get the result
rec = astra.data2d.get(rec_id)

# Clean up.
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)

# --- ODL ---

# Create reconstruction space
reco_space = odl.uniform_discr(-domain_size / 2, domain_size / 2, domain_size)

# Create geometry
geometry = odl.tomo.parallel_beam_geometry(reco_space, n_angles, det_size)

# Create ray transform
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')

# Create sinogram
rhs = ray_trafo(data)

# Solve with CGLS (aka CGN)
x = reco_space.zero()
with odl.util.Timer('ODL run'):
    odl.solvers.conjugate_gradient_normal(ray_trafo, x, rhs, niter=niter)

# Display results for comparison
plt.figure('data')
plt.imshow(data.T, origin='lower', cmap='bone')
plt.figure('ASTRA sinogram')
plt.imshow(sinogram.T, origin='lower', cmap='bone')
plt.figure('ASTRA reconstruction')
plt.imshow(rec.T, origin='lower', cmap='bone')
plt.figure('ODL sinogram')
plt.imshow(rhs.asarray().T, origin='lower', cmap='bone')
plt.figure('ODL reconstruction')
plt.imshow(x.asarray().T, origin='lower', cmap='bone')
plt.show()
