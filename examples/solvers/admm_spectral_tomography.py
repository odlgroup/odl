"""Total variation spectral tomography using preconditioned nonlinear ADMM.

In this example we solve the optimization problem

    min_x  ||A(x) - y||_2^2 + lam * ||grad(x)||_{nuc, 1, 2}

where ``||.||_{nuc, 1, 2}`` is the nuclear L1-L2 norm and ``A`` a 2 x 4
operator matrix

        (c_1 * exp(-R)  c_2 * exp(-R)                              )
    A = (                              c_3 * exp(-R)  c_4 * exp(-R))

with constants ``c_i`` and the parallel beam ray transform ``R``. In other
words, ``A`` acts on an input ``x`` with 4 components as follows:

           (c_1 * exp(-R(x_1)) + c_2 * exp(-R(x_2)))
    A(x) = (c_3 * exp(-R(x_3)) + c_4 * exp(-R(x_4)))

This reflects a simplified model for 2D spectral CT with 4 (discrete) beam
energies and 2 detector energy bins.

The problem is rewritten in decoupled form as

    min_x g(L(x))

with a separable sum ``g`` of functionals and the stacked operator ``L``:

    g(z) = ||z_1 - g||_2^2 + lam * ||z_2||_1,

               ( A(x)    )
    z = L(x) = ( grad(x) ).

See the documentation of the `admm_precon_nonlinear` solver for further
details.
Note that this problem is underdetermined as a simple count of input
channels vs. data channels shows. This can be changed in the code below
(along with ``souce_spectrum`` and ``energy_factors``).
"""

import numpy as np
import odl

# --- Set up the forward operator --- #

# Space of a monochromatic phantom (volume)
single_energy_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100], dtype='float32')
reco_space = single_energy_space ** 2

# Create a parallel beam geometry with flat detector, using 180 angles
geometry = odl.tomo.parallel_beam_geometry(single_energy_space, num_angles=180)

# Define number of energy bins on both sides as well as the source spectrum
num_bins_reco = 4
num_bins_data = 2
assert num_bins_reco >= num_bins_data
assert num_bins_reco % num_bins_data == 0
group_size = num_bins_reco // num_bins_data
# Note: the spectrum (i.e. intensity per channel) is additive on the data
# side, i.e., adding more entries adds to the total signal in the data bins
source_spectrum = [8e4, 1e5, 8e4, 6e4]
assert len(source_spectrum) == num_bins_reco

# Create the forward operator
ray_trafo = odl.tomo.RayTransform(single_energy_space, geometry,
                                  impl='astra_cpu')
exp_op = odl.ufunc_ops.exp(ray_trafo.range)
op_mat = []
for data_bin in range(num_bins_data):
    row = [0] * num_bins_reco
    for j in range(data_bin * group_size, (data_bin + 1) * group_size):
        row[j] = source_spectrum[j] * exp_op * (-ray_trafo)
    op_mat.append(row)

fwd_op = odl.ProductSpaceOperator(op_mat)


# Small helper to visualize the forward operator
def show_fwd_op(num_bins_reco, num_bins_data):
    entry_len = 13
    dtype = 'U' + str(entry_len)
    mat = np.empty((num_bins_data, num_bins_reco), dtype=dtype)
    mat[:] = ' ' * entry_len
    for i in range(mat.shape[0]):
        for j in range(i * group_size, (i + 1) * group_size):
            mat[i, j] = 'c_{} * exp(-R)'.format(j)
    print(mat)


print('Forward operator (R = ray transform, c_i = constants):')
show_fwd_op(num_bins_reco, num_bins_data)

# --- Generate artificial data --- #

# Create multispectral phantom, very simple: C_energy * single_energy_phantom
energy_factors = [1e-2, 8e-3, 6e-3, 1e-3]
assert len(energy_factors) == num_bins_reco
single_energy_phantom = odl.phantom.shepp_logan(single_energy_space,
                                                modified=True)
phantom = [fac * single_energy_phantom for fac in energy_factors]

# Simulate noiseless (expected) projections and photon counts
phantom = fwd_op.domain.element(phantom)
meas_intens = fwd_op(phantom)
counts = odl.phantom.poisson_noise(meas_intens, seed=123)

# --- Formulate problem in log space (more stable) --- #

# The new opertor is F_i = -log(A_i / expected_counts[i]), where
# the expected counts are the sums of counts over the groups
expected_bin_counts = []
for i in range(num_bins_data):
    expected_bin_counts.append(
        sum(source_spectrum[i * group_size: (i + 1) * group_size]))

scaling_ops = [odl.ScalingOperator(fwd_op.range[i], 1 / expected_bin_counts[i])
               for i in range(num_bins_data)]
scaling = odl.DiagonalOperator(*scaling_ops)
log_fwd_op = -odl.ufunc_ops.log(fwd_op.range) * scaling * fwd_op

# Transform the simulated counts accordingly
log_counts = fwd_op.range.element()
for i in range(num_bins_data):
    log_counts[i] = -np.log(counts[i] / expected_bin_counts[i])

# --- Set up the inverse problem --- #

# Gradient operator for the TV part (one per reco_space component)
grad = odl.DiagonalOperator(odl.Gradient(single_energy_space), num_bins_reco)

# Stacking of the two operators
L = odl.BroadcastOperator(log_fwd_op, grad)

# Data matching functional
data_fit = odl.solvers.L2NormSquared(log_fwd_op.range).translated(log_counts)
# Regularization functional, we use the Nuclear L1-L2 norm to couple the
# energy channels
reg_func = 0.08 * odl.solvers.NuclearNorm(grad.range)
g = odl.solvers.SeparableSum(data_fit, reg_func)

# Enforce a value range for the reco to avoid underflow of exp(-x)
f = odl.solvers.IndicatorBox(L.domain, 0, 1e-2)

# --- Select parameters and solve using ADMM --- #

niter = 200  # Number of iterations
delta = 1.0  # Step size for the constraint update

# Optionally pass a callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration(step=5) &
            odl.solvers.CallbackShow(step=5))

# Choose a starting point, the FBP reconstruction from the first bin
fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.9)
fbp = fbp_op(log_counts[0])
x = []
for _ in range(num_bins_reco):
    x.append(fbp.copy())
x = L.domain.element(x)

# Run the algorithm
odl.solvers.nonsmooth.admm.admm_precon_nonlinear(
    x, f, g, L, delta, niter, opnorm_factor=0.5, callback=callback)

# Display images
phantom.show(title='Phantom')
counts.show(title='Simulated photon counts (Sinogram)')
fbp.show('Monochromatic FBP reconstruction (bin 0)')
x.show(title='TV reconstruction', force_show=True)
