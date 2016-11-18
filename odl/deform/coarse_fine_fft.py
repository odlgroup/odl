# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:45:10 2016

@author: chong
"""
"""
Example for part of memory-saving LDDMM solver.
"""

import odl
import numpy as np


def gaussian_kernel_matrix_adaptive_grid(grid1, grid2, sigma):
    """Return the kernel matrix for Gaussian kernel.

    The Gaussian kernel matrix ``K`` in ``n`` dimensions is defined as::
    
        k_ij = exp(- |x_i - y_j|^2 / (2 * sigma^2))
    
    where ``x_i`` and ``y_j`` runs through grid1 points and grid2 points.
    The final matrix has size ``N x M``, where ``N`` is the total
    number of grid1 points, ``M`` is the total number of grid2 points.
    
    Parameters
    ----------
    grid1 : `RegularGrid`
        Grid where the image points are defined
    grid2 : `RegularGrid`
        Grid where the control points are defined
    sigma : `float`
        Width of the Gaussian kernel
    """
    point_arrs1 = grid1.points()
    point_arrs2 = grid2.points()
    matrices = np.zeros((point_arrs1.shape[0], point_arrs2.shape[0]))
    for i, parr1 in enumerate(point_arrs1):
        for j, parr2 in enumerate(point_arrs2):
            matrices[i,j] = sum((parr1 - parr2) * (parr1 - parr2))

    return np.exp(-matrices / (2 * sigma**2))


d = 2

sparse_domain = odl.uniform_discr([-1]*d, [1]*d, [51]*d)
dense_domain = odl.uniform_discr([-1]*d, [1]*d, [101]*d)

sparse_ft_op = odl.trafos.FourierTransform(sparse_domain)
dense_ft_op = odl.trafos.FourierTransform(dense_domain)

sigma = 0.1

# From coarse to fine 
coarse_fine_resizing = odl.ResizingOperator(sparse_ft_op.range,
                                            dense_ft_op.range)                                    

kernel = sparse_ft_op.domain.element(
    lambda x: np.exp(-sum(xi**2 for xi in x) / (2 * sigma**2)))
#k /= k.inner(k.space.one())

# Compute by FFT-based Convolution
kernel_ft = sparse_ft_op(kernel)
conv_op = (2 * np.pi)**(d/2.0) * dense_ft_op.inverse * coarse_fine_resizing *\
    kernel_ft * sparse_ft_op

# Creat function to be acted by kernel
shepp_logan = odl.phantom.shepp_logan(sparse_domain, modified=True)

shepp_logan.show('shepp_logan')
kernel.show('k')
result = conv_op(shepp_logan)
result.show('operator')

# Compute by matrix-vector multiplying
kernel_matrix = gaussian_kernel_matrix_adaptive_grid(dense_domain,
                                                     sparse_domain, sigma)
shepp_logan_convert = np.array(shepp_logan).reshape((shepp_logan.size, 1))

result_comp_temp = np.dot(kernel_matrix, shepp_logan_convert) * \
    sparse_domain.cell_volume
result_comp = result_comp_temp.reshape(dense_domain.shape)
result_comp = conv_op.range.element(result_comp)
result_comp.show('result_comp')

# Compare results
(result_comp - result).show('difference')


## From fine to coarse
#fine_coarse_resizing = odl.ResizingOperator(dense_ft_op.range,
#                                            sparse_ft_op.range)
#kernel_dense = dense_domain.element(
#    lambda x: np.exp(-sum(xi**2 for xi in x) / (2 * sigma**2)))
#kernel_dense_ft = dense_ft_op(kernel_dense)
#
#conv_op2 = (2 * np.pi)**(d/2.0) * sparse_ft_op.inverse * \
#    fine_coarse_resizing * kernel_dense_ft * dense_ft_op
#
#shepp_logan_dense = odl.phantom.shepp_logan(dense_domain, modified=True)
#shepp_logan_dense.show('shepp_logan')
#kernel_dense.show('k')
#result = conv_op2(shepp_logan_dense)
#result.show('operator')
#
#kernel_matrix = gaussian_kernel_matrix_adaptive_grid(sparse_domain,
#                                                     dense_domain, sigma)
#shepp_logan_convert = np.array(shepp_logan_dense).reshape(
#    (shepp_logan_dense.size, 1))
#
#result_comp_temp = np.dot(kernel_matrix, shepp_logan_convert) * \
#    dense_domain.cell_volume
#result_comp = result_comp_temp.reshape(sparse_domain.shape)
#
#result_comp = conv_op2.range.element(result_comp)
#result_comp.show('result_comp')
## Compare results
#(result_comp - result).show('difference')

#conv_op.inverse(result).show()
#
#space = odl.uniform_discr(
#    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
#    dtype='float32', interp='linear')
#ft_op = odl.trafos.FourierTransform(space)
