# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# pylint: disable=line-too-long

"""Scipy implementation of Sparse Matrices."""

from scipy.sparse import coo_matrix

from odl.core.sparse.sparse_template import SparseMatrixFormat, _registered_sparse_formats

if ('scipy' not in _registered_sparse_formats
        or 'COO' not in _registered_sparse_formats['scipy']):
    scipy_coo_tensor = SparseMatrixFormat(
        sparse_format='COO',
        impl = 'scipy',
        constructor = coo_matrix,
        is_of_this_sparse_format = lambda x : isinstance(x, coo_matrix),
        to_dense = lambda matrix: matrix.toarray(),
        matmul_spmatrix_with_vector = lambda matrix, x: matrix.dot(x)
    )

