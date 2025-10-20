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

