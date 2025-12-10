
from odl.core.sparse.sparse_template import SparseMatrixFormat, _registered_sparse_formats


from typing import Optional


IS_INITIALIZED = False

def _initialize_if_needed():
    """Initialize ``_registered_sparse_formats`` if not already done."""
    global IS_INITIALIZED
    if not IS_INITIALIZED:
        import odl.backends.sparse.scipy_backend
        import importlib.util       
        torch_module = importlib.util.find_spec("torch")
        if torch_module is not None:
            try:
                import odl.backends.sparse.pytorch_backend
            except ModuleNotFoundError:
                pass
        IS_INITIALIZED = True


class SparseMatrix:
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

        # sanity checks
        assert isinstance(
            format, str
        ), f"The sparse data format can only be a string, got {type(format)}"
        assert isinstance(
            impl, str
        ), f"The impl argument can only be a str, got {type(impl)}"

        # Getting the backend (scipy, Pytorch...)
        backend_formats = _registered_sparse_formats.get(impl)
        if backend_formats is None:
            raise ValueError(
                f"The backend {impl} is not supported. Only {list(_registered_sparse_formats.keys())} are registered backends."
            )
        # Getting the format (COO, CSR...)
        sparse_impl = backend_formats.get(format)
        if sparse_impl is None:
            raise ValueError(
                f"No format {impl}. Only {list(backend_formats.keys())} are registered backends."
            )

        return sparse_impl.constructor(*args, **kwargs)


def lookup_sparse_format(matrix: object) -> Optional[SparseMatrixFormat]:
    """Looks up the sparse format of a matrix.
    Goes through the registered backends (scipy, pytorch...) and formats (COO, CSR...)

    Args:
        matrix (object): The matrix we want to get the sparse format of

    Returns:
        Optional[SparseMatrixFormat]: returns the sparse-format identifier if
        the matrix has one of the registered formats. Otherwise `None`.

    Notes:
        "sp_bkend" = sparse backend
        "sp_fmt"   = sparse format
    """
    _initialize_if_needed()
    for sp_bkend in _registered_sparse_formats.values():
        for sp_fmt in sp_bkend.values():
            if sp_fmt.is_of_this_sparse_format(matrix):
                return sp_fmt
    return None


def is_sparse(matrix:object) -> bool:
    """Checks whether the object is a sparse matrix in one
    of the format known to ODL.

    Args:
        matrix (object): input matrix

    Returns:
        bool: True if matrix is sparse else False
    """
    return lookup_sparse_format(matrix) is not None


def get_sparse_matrix_impl(matrix:object) -> str:
    """Gets the implementation string name of a matrix (which
    must be in one of the sparse formats known to ODL).

    Args:
        matrix (object): matrix

    Returns:
        str: The implementation string identifier ('pytorch', 'scipy', ...)
    """
    instance = lookup_sparse_format(matrix)
    assert instance is not None, "The matrix is not a supported sparse matrix"
    return instance.impl


def get_sparse_matrix_format(matrix:object) -> str:
    """Gets the format string name of a matrix

    Args:
        matrix (object): matrix

    Returns:
        str: The format string identifier ('COO', 'CSR', ...)
    """
    instance = lookup_sparse_format(matrix)
    assert instance is not None, "The matrix is not a supported sparse matrix"
    return instance.sparse_format

if __name__ == '__main__':
    print(SparseMatrix('COO', 'pytorch', 
                       [[0, 1, 1],[2, 0, 2]], [3, 4, 5], 
                       device='cuda:0'))
    print(SparseMatrix('COO', 'scipy', (3, 4)))
