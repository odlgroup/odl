"""Tomography using the `kaczmarz` solver.

Solves the inverse problem

    A(x) = g

Where ``A`` is a fan (cone) beam forward projector, ``x`` the result and
 ``g`` is given data.

In order to solve this using `kaczmarz`'s method, the operator is split into
several sub-operators (each representing a subset of the angles and detector
points). This allows a faster solution.
"""

import odl


# --- Set up the forward operator (ray transform) --- #


# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a fan (cone) beam geometry with flat detector
geometry = odl.tomo.cone_beam_geometry(space,
                                       src_radius=40, det_radius=40,
                                       num_angles=360, det_shape=360)

# Here we split the geometry according to both angular subsets and
# detector subsets.
# For practical applications these choices should be fine tuned,
# these values are selected to give an illustrative visualization.

split = 'block'

if split == 'block':
    # Split the data into blocks:
    # 111 222 333
    n = 20
    ns = geometry.angles.size // n

    ray_trafos = [odl.tomo.RayTransform(space, geometry[i * ns:(i + 1) * ns])
                  for i in range(n)]
elif split == 'interlaced':
    # Split the data into slices:
    # 123 123 123
    n = 20

    ray_trafos = [odl.tomo.RayTransform(space, geometry[i::n])
                  for i in range(n)]

# --- Generate artificial data --- #


# Create phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create sinogram of forward projected phantom with noise
data = [ray_trafo(phantom) for ray_trafo in ray_trafos]

# Compute steplengths
omega = [odl.power_method_opnorm(ray_trafo) ** (-2)
         for ray_trafo in ray_trafos]

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Choose a starting point
x = space.zero()

# Run the algorithm, call the callback in each iteration for visualization.
# Note that using only 2 iterations still gives a decent reconstruction.
odl.solvers.kaczmarz(
    ray_trafos, x, data, niter=2, omega=omega,
    callback=callback, callback_loop='inner')

# Display images
phantom.show(title='original image')
x.show(title='reconstructed image', force_show=True)
