r"""Total-variation regularized tomography example using the adupdates solver.

This example solves a linear inverse problem of the form :math:`Ax = y`, where
:math:`x \in \mathbb{R}^n` is the (unknown) original image to be reconstructed,
:math:`y \in \mathbb{R}^m` is the noisy data and :math:`A \in \mathbb{R}^{m
\times n}` is the measurement matrix describing the discretized physical model.

To solve this problem, we first split the measurement matrix and the data into
:math:`m` parts and solve the least squares problem

..math::
    \min_{x\in\mathbb{R}^n} D(x) := \sum_{j = 1}^m \| A_j x - y_j \|^2.

To regularize, we add terms for the total variation and for a functional which
guarantees that the solution will be pointwise nonnegative. The variational
regularization problem we are solving is therefore

..math::
    \min_{x\in \mathbb{R}^n_+} D(x) + \sum_{i = 1}^d (\| \partial_{i, 1} x \|_1
    + \| \partial_{i, 2} x \|_1.

Here, :math:`\partial_{i, 1}` and :math:`\partial_{i, 2}` contain the even and
odd components, respectively, of the discretized :math:`i`-th partial
derivative, and :math:`d` is the dimension of the tomography problem. In this
example, :math:`d = 2`. We solve the problem with the alternating dual updates
method. For further details, see
`[MF2015] <http://ieeexplore.ieee.org/document/7271047/>`_.

References
----------
[MF2015] McGaffin, M G, and Fessler, J A. *Alternating dual updates
algorithm for X-ray CT reconstruction on the GPU*. IEEE Transactions
on Computational Imaging, 1.3 (2015), pp 186--199.
"""

import numpy as np
import odl

# The following parameters determine how to split the data (sinograms) and if
# the solver should do the inner iterations in a fixed order or at random.
SPLIT_METHOD = 'interlaced'  # How to split the data ('block' or 'interlaced')?
SPLIT_NUMBER = 20     # How many pieces of data?
RANDOM = True         # Choose the oder of the inner iterations at random?

# --- Create simulated data (phantom) ---

# Reconstruction space: Set of two-dimensional quadratic images.
reco_space = odl.uniform_discr(min_pt=[-40.0, -40.0],
                               max_pt=[40.0, 40.0],
                               shape=[1024, 1024])
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create the forward operators. They correspond to a fully sampled parallel
# beam geometry.
geometry = odl.tomo.parallel_beam_geometry(reco_space)

if SPLIT_METHOD == 'block':
    # Split the data into blocks:
    # 111 222 333
    ns = geometry.angles.size // SPLIT_NUMBER
    ray_trafos = [odl.tomo.RayTransform(reco_space,
                                        geometry[i * ns:(i + 1) * ns])
                  for i in range(SPLIT_NUMBER)]

elif SPLIT_METHOD == 'interlaced':
    # Split the data into slices:
    # 123 123 123
    ray_trafos = [odl.tomo.RayTransform(reco_space,
                                        geometry[i::SPLIT_NUMBER])
                  for i in range(SPLIT_NUMBER)]
else:
    raise ValueError('unknown data split`{}`: typo?'.format(SPLIT_METHOD))

# Create the artificial data.
data_spaces = [op.range for op in ray_trafos]
noisefree_data = [op(phantom) for op in ray_trafos]
data = [proj + 0.10 * np.ptp(proj) * odl.phantom.white_noise(proj.space)
        for proj in noisefree_data]

# Functionals and operators for the total variation. This is the l1 norm of the
# (discretized) gradient of the reconstruction. For each of the dimensions
# we create two functionals and two operators.

# Start with empty lists ...
tv_functionals = []
tv_operators = []
tv_stepsizes = []

# ... and for each dimension of the reconstruction space ...
reco_shape = reco_space.shape
reco_dim = len(reco_shape)
for dim in range(reco_dim):
    # ... add two operators taking only the even and odd elements,
    # respectively, in that dimension.
    partial_der = odl.PartialDerivative(
        reco_space, dim, pad_mode='order0')
    all_points = list(np.ndindex(reco_shape))
    even_pts = [list(p) for p in all_points if p[dim] % 2 == 0]
    even_pts = np.array(even_pts).T.tolist()
    odd_pts = [list(p) for p in all_points if p[dim] % 2 == 1]
    odd_pts = np.array(odd_pts).T.tolist()
    op1 = reco_space.cell_sides[dim] * odl.SamplingOperator(
        reco_space, even_pts) * partial_der
    op2 = reco_space.cell_sides[dim] * odl.SamplingOperator(
        reco_space, odd_pts) * partial_der
    tv_functionals += [odl.solvers.L1Norm(op1.range),
                       odl.solvers.L1Norm(op2.range)]
    tv_operators += [op1, op2]
    tv_stepsizes += [0.5 / reco_shape[dim], 0.5 / reco_shape[dim]]

# Functional and operator enforcing the nonnegativity of the image.
nonneg_functional = odl.solvers.IndicatorNonnegativity(reco_space)
nonneg_operator = odl.IdentityOperator(reco_space)
nonneg_stepsize = 1.0

# ... and the data fit functionals. The coefficient is a regularization
# paratemeter, which determines the tradeoff between data fit and regularity.
data_fit_functionals = [1.0 *
                        odl.solvers.L2NormSquared(ds).translated(rhs)
                        for (ds, rhs) in zip(data_spaces, data)]
# In the stepsizes, we avoid the possible division by zero by adding a small
# positive value. The matrix corresponding to the operator `op` has only
# nonnegative entries, which ensures that the final results are positve.
data_fit_stepsizes = [1.0 / (1e-6 + op(op.adjoint(ds.one())))
                      for (ds, op) in zip(data_spaces, ray_trafos)]
# Alternative choice without vector-valued stepsizes could be
# data_fit_stepsizes = [1.0 / op.norm(estimate=True) ** 2 for op in ray_trafos]

# Now we build up the ingredients of our algorithm:
# Start at a zero image, ...
x = reco_space.zero()

# ... collect all the functionals, ...
g = [nonneg_functional] + data_fit_functionals + tv_functionals

# ... collect all the operators, ...
L = [nonneg_operator] + ray_trafos + tv_operators

# ... and collect all the inner stepsizes, which were chosen according to the
# properties of the operators in `L`.
inner_stepsizes = [nonneg_stepsize] + data_fit_stepsizes + tv_stepsizes

odl.solvers.adupdates(x, g, L, stepsize=1.0, inner_stepsizes=inner_stepsizes,
                      niter=5, random=RANDOM, callback=None,
                      callback_loop=None)

# Show the result within a window between zero and one.
x.show(vmin=0.0, vmax=1.0)
