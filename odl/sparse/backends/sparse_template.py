from dataclasses import dataclass
from typing import Callable

_registered_sparse_implementations = {}
_registered_sparse_formats = {}

@dataclass
class SparseMatrixFormat:
    sparse_format : str
    impl : str
    constructor : Callable
    is_sparse : Callable
    def __post_init__(self):
        _registered_sparse_implementations[self.impl]  = self
        _registered_sparse_formats[self.sparse_format] = self
