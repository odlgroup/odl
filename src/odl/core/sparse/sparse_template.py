"""ODL wrapper for multi-backend Sparse Matrices support."""

from dataclasses import dataclass
from typing import Callable

_registered_sparse_formats = {}


@dataclass
class SparseMatrixFormat:
    """Metainformation about Sparse Matrices in some storage format.
    This is a singleton class, it has a post-init method that registers the Sparse Matrix
    format in a global, internal dict
    (with [impl][format] key structure, for instance ['scipy']['COO'])

    Attributes
    ----------
    sparse_format : str
        The shorthand descriptor of the layout of the sparse matrix, e.g 'COO'
    impl : str
        The identifier of the backend, e.g 'scipy'
    constructor : Callable
        Constructor of the Sparse backend
    is_of_this_sparse_format : Callable[[object], bool]
        Checks whether an array is of the same backend and layout of the given SparseMatrixFormat 
    to_dense : Callable
        Makes a sparse matrix dense
    matmul_spmatrix_with_vector : Callable
        Sparse matrix multiplication function (the signature differs accross backends)
    """
    sparse_format: str
    impl: str
    constructor: Callable
    is_of_this_sparse_format: Callable[[object], bool]
    to_dense: Callable
    matmul_spmatrix_with_vector: Callable

    def __post_init__(self):
        if self.impl not in _registered_sparse_formats:
            _registered_sparse_formats[self.impl] = {}
        if self.sparse_format in _registered_sparse_formats[self.impl]:
            raise KeyError(
                f"A {self.sparse_format} sparse format for backend {self.impl}"
               + " is already registered."
               + " Every sparse format needs to have a unique identifier combination."
            )
        _registered_sparse_formats[self.impl][self.sparse_format] = self
