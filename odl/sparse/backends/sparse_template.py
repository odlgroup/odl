from dataclasses import dataclass
from typing import Callable

_registered_sparse_formats = {}

@dataclass
class SparseMatrixFormat:
    sparse_format : str
    impl : str
    constructor : Callable
    is_sparse : Callable
    def __post_init__(self):
        if self.impl not in _registered_sparse_formats:
            _registered_sparse_formats[self.impl] = {}
        if self.sparse_format in _registered_sparse_formats[self.impl]:
            raise KeyError(f"A {self.sparse_format} sparse format for backend {self.impl} is already registered. Every sparse format needs to have a unique identifier combination.")
        _registered_sparse_formats[self.impl][self.sparse_format] = self
