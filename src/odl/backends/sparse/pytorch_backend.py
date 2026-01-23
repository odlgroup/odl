# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# pylint: disable=line-too-long
# pylint: disable=invalid-name

"""PyTorch implementation of Sparse Matrices."""

from torch import sparse_coo_tensor, Tensor, sparse_coo, matmul

from odl.core.sparse.sparse_template import SparseMatrixFormat, _registered_sparse_formats

def is_sparse_COO(matrix):
    """Convenience function to check whether a matrix is a sparse PyTorch Tensor with the COO format"""
    return isinstance(matrix, Tensor) and matrix.is_sparse and matrix.layout == sparse_coo

if ('pytorch' not in _registered_sparse_formats
        or 'COO' not in _registered_sparse_formats['pytorch']):
    pytorch_coo_tensor = SparseMatrixFormat(
        sparse_format='COO',
        impl = 'pytorch',
        constructor = sparse_coo_tensor,
        is_of_this_sparse_format = is_sparse_COO,
        to_dense = lambda matrix: matrix.to_dense(),
        matmul_spmatrix_with_vector = matmul
    )
