from scipy.sparse import coo_matrix

from .sparse_template import SparseMatrixFormat
    
scipy_coo_tensor = SparseMatrixFormat(
    sparse_format='COO',
    impl = 'scipy',
    constructor = coo_matrix,
    is_of_this_sparse_format = lambda x : isinstance(x, coo_matrix)
)
