from types import ModuleType
from dataclasses import dataclass
from typing import Callable


__all__ = (
    'AVAILABLE_DEVICES',
    'IMPL_DEVICE_PAIRS',
    'check_device',
    'ArrayBackend', 'lookup_array_backend',)


_registered_array_backends = {}

@dataclass
class ArrayBackend:
    impl: str
    array_namespace: ModuleType
    available_dtypes: dict[str, object]
    array_type: type
    array_constructor: Callable
    make_contiguous: Callable
    identifier_of_dtype: Callable[object, str]
    def __post_init__(self):
        if self.impl in _registered_array_backends:
            raise KeyError(f"An array-backend with the identifier {self.impl} is already registered."
                          + " Every backend needs to have a unique identifier.")
        _registered_array_backends[self.impl] = self
    def get_dtype_identifier(self, **kwargs):
        if 'array' in kwargs:
            assert 'dtype' not in kwargs, 'array and dtype are multually exclusive parameters'
            return self.identifier_of_dtype(kwargs['array'].dtype)
        if 'dtype' in kwargs:
            assert 'array' not in kwargs, 'array and dtype are multually exclusive parameters'
            return self.identifier_of_dtype(kwargs['dtype'])
        raise ValueError("Either 'array' or 'dtype' argument must be provided.")
    
    def __eq__(self, other):
        return isinstance(other, ArrayBackend) and self.impl == other.impl

def lookup_array_backend(impl: str) -> ArrayBackend:
    return _registered_array_backends[impl]

def get_array_and_backend(x, must_be_contiguous=False):
    from odl.space.base_tensors import Tensor
    if isinstance(x, Tensor):
        return x.asarray(must_be_contiguous=must_be_contiguous), x.space.array_backend

    from odl.space.pspace import ProductSpaceElement
    if isinstance(x, ProductSpaceElement):
        return get_array_and_backend(x.asarray(), must_be_contiguous=must_be_contiguous)

    for backend in _registered_array_backends.values():
        if isinstance(x, backend.array_type):
            if must_be_contiguous:
                return backend.make_contiguous(x), backend
            else:
                return x, backend

    else:
        raise ValueError(f"The registered array backends are {list(_registered_array_backends.keys())}. The argument provided is a {type(x)}, check that the backend you want to use is supported and has been correctly instanciated.")

AVAILABLE_DEVICES = {
    'numpy' : ['cpu'],
    # 'pytorch' : ['cpu'] +  [f'cuda:{i}' for i in range(torch.cuda.device_count())]
}

IMPL_DEVICE_PAIRS = []
for impl in AVAILABLE_DEVICES.keys():
    for device in AVAILABLE_DEVICES[impl]:
        IMPL_DEVICE_PAIRS.append((impl, device))

def check_device(impl:str, device:str):
    """
    Checks the device argument 
    This checks that the device requested is available and that its compatible with the backend requested
    """
    assert device in AVAILABLE_DEVICES[impl], f"For {impl} Backend, devices {AVAILABLE_DEVICES[impl]} but {device} was provided."
    
if __name__ =='__main__':
    check_device('numpy', 'cpu')
