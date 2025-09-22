from torch import sparse_coo_tensor, Tensor, sparse_coo

from .sparse_template import SparseMatrixFormat

def is_sparse_COO(matrix):
    return isinstance(matrix, Tensor) and matrix.is_sparse and matrix.layout == sparse_coo
    
pytorch_coo_tensor = SparseMatrixFormat(
    sparse_format='COO',
    impl = 'pytorch',
    constructor = sparse_coo_tensor,
    is_of_this_sparse_format = is_sparse_COO,
    to_dense = lambda matrix: matrix.to_dense()
)
