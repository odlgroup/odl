# pylint: disable=line-too-long

from odl.core.sparse.sparse_template import (
    SparseMatrixFormat,
    _registered_sparse_formats,
)


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

    def __new__(cls, format: str, impl: str, *args, **kwargs):

        _initialize_if_needed()

        sparse_impl = _registered_sparse_formats[impl][format]

        return sparse_impl.constructor(*args, **kwargs)


def lookup_sparse_format(matrix: object) -> Optional[SparseMatrixFormat]:
    _initialize_if_needed()
    for sp_bkend in _registered_sparse_formats.values():
        for sp_fmt in sp_bkend.values():
            if sp_fmt.is_of_this_sparse_format(matrix):
                return sp_fmt
    return None


def is_sparse(matrix):
    return lookup_sparse_format(matrix) is not None


def get_sparse_matrix_impl(matrix):
    instance = lookup_sparse_format(matrix)
    assert instance is not None, "The matrix is not a supported sparse matrix"
    return instance.impl


def get_sparse_matrix_format(matrix):
    instance = lookup_sparse_format(matrix)
    assert instance is not None, "The matrix is not a supported sparse matrix"
    return instance.sparse_format


if __name__ == "__main__":
    print(
        SparseMatrix(
            "COO", "pytorch", [[0, 1, 1], [2, 0, 2]], [3, 4, 5], device="cuda:0"
        )
    )
    print(SparseMatrix("COO", "scipy", (3, 4)))
