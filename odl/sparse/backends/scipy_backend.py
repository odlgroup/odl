from scipy.sparse import coo_matrix

from .sparse_template import SparseMatrixFormat
    
scipy_coo_tensor = SparseMatrixFormat(
    sparse_format='COO',
    impl = 'scipy',
    constructor = coo_matrix,
    is_sparse = lambda x : isinstance(x, coo_matrix)
)

SUPPORTED_IMPLS = {
    'COO':scipy_coo_tensor
    }