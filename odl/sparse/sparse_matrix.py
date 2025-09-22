
from odl.sparse.backends.scipy_backend import SUPPORTED_IMPLS as scp_impls

SUPPORTED_IMPLS     = {'scipy':scp_impls}
SUPPORTED_INSTANCES = list(scp_impls.values())

IS_INITIALIZED = False

def _initialize_if_needed():
    """Initialize ``SUPPORTED_IMPLS`` if not already done."""
    global IS_INITIALIZED
    global SUPPORTED_INSTANCES
    if not IS_INITIALIZED:
        import importlib.util       
        torch_module = importlib.util.find_spec("torch")
        if torch_module is not None:
            try:
                from odl.sparse.backends.pytorch_backend import SUPPORTED_IMPLS as pt_impls
                SUPPORTED_IMPLS['pytorch'] = pt_impls
                SUPPORTED_INSTANCES += list(pt_impls.values())
            except ModuleNotFoundError:
                pass
        IS_INITIALIZED = True

class SparseMatrix():    
    """
    SparseMatrix is the ODL interface to the sparse Matrix supports in different backends.

    Note:
    The user is responsible for using the *args and **kwargs expected by the respective backends:
    Pytorch: 
        -> COO: https://docs.pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html
    Scipy: 
        -> COO: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.
        
    Examples:
    SparseMatrix('COO', 'pytorch', 
                [[0, 1, 1],[2, 0, 2]], [3, 4, 5], 
                device='cuda:0')
    SparseMatrix('COO', 'scipy', 
                (3, 4))
    """
    def __new__(cls,  format:str, impl:str, *args, **kwargs):

        _initialize_if_needed()

        sparse_impl = SUPPORTED_IMPLS[impl][format]

        return sparse_impl.constructor(*args, **kwargs)
    
def is_sparse(matrix):
    _initialize_if_needed()
    for instance in SUPPORTED_INSTANCES:
        if instance.is_of_this_sparse_format(matrix):
            return True
    return False

def get_sparse_matrix_impl(matrix):
    _initialize_if_needed()
    assert is_sparse(matrix), 'The matrix is not a supported sparse matrix'
    for instance in SUPPORTED_INSTANCES:
        if instance.is_of_this_sparse_format(matrix):
            return instance.impl

def get_sparse_matrix_format(matrix):
    _initialize_if_needed()
    assert is_sparse(matrix), 'The matrix is not a supported sparse matrix'
    for instance in SUPPORTED_INSTANCES:
        if instance.is_of_this_sparse_format(matrix):
            return instance.sparse_format

if __name__ == '__main__':
    print(SparseMatrix('COO', 'pytorch', 
                       [[0, 1, 1],[2, 0, 2]], [3, 4, 5], 
                       device='cuda:0'))
    print(SparseMatrix('COO', 'scipy', (3, 4)))
