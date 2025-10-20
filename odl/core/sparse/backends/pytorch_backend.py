from torch import sparse_coo_tensor, Tensor, sparse_coo, matmul

from odl.core.sparse.backends.sparse_template import SparseMatrixFormat, _registered_sparse_formats

def is_sparse_COO(matrix):
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
