from typing import Dict

import numpy as np
import odl

def mri_mlem_adam(
    parameters:Dict,
    dimension   :    int,
    n_points  :      int,
    max_iterations : int
    ):
    subsampling  : float = parameters['subsampling']
    learning_rate: float = parameters['learning_rate']
    beta1: float = parameters['beta1']
    beta2: float = parameters['beta2']
    eps:   float = parameters['eps']
    # Create a space
    space = odl.uniform_discr(
            [0 for _ in range(dimension)], 
            [n_points for _ in range(dimension)], 
            [n_points for _ in range(dimension)]
        )
    # Create MRI operator. First fourier transform, then subsample
    ft = odl.trafos.FourierTransform(space)
    sampling_points = np.random.rand(*ft.range.shape) < subsampling #type:ignore
    sampling_mask = ft.range.element(sampling_points)
    mri_op = sampling_mask * ft

    # Create noisy MRI data
    phantom = odl.phantom.shepp_logan(space, modified=True)
    noisy_data = mri_op(phantom) + odl.phantom.white_noise(mri_op.range) * 0.1  #type:ignore

    g = odl.solvers.L2Norm(mri_op.range).translated(noisy_data) * mri_op

    # Solve
    x = mri_op.domain.zero()
    odl.solvers.adam(
        g, x, 
        maxiter=max_iterations, 
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps)

    ### Return the data; compare it against target (l2 norm)
    return np.linalg.norm(phantom - x.data)
