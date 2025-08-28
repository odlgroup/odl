# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for the compatibility of ODL with the python array API"""

from types import ModuleType
from dataclasses import dataclass
from typing import Callable, Union

__all__ = (
    'ArrayBackend', 
    'lookup_array_backend',
    'get_array_and_backend',
    'check_device',
    'can_cast',
    )


_registered_array_backends = {}

@dataclass
class ArrayOperation:
    name: str
    operation_call: Callable
    supports_single_input: bool
    supports_two_inputs: bool
    supports_out_argument: bool

@dataclass
class ArrayBackend:
    """
    Class to implement the array backend associated to each TensorSpace Implementations.
    
    Attributes
    ----------
    impl : str
        The implementation of the backend, e.g 'numpy'
    array_namespace : ModuleType
        The actual namespace of the backend, e.g np
    available_dtypes : dict
        A dictionnary mapping a Number/str datatype to the corresponding backend-specific datatype, e.g {float:np.float64, 'float64', np.float64, ...}
    array_type : type
        The type of the array once implemented by the backend, e.g np.ndarray
    array_constructor : Callable
        The function the backend uses to create an array, e.g np.asarray
    make_contiguous : Callable
        The function the backend uses to make an array contiguous, e.g np.ascontiguousasarray
    identifier_of_dtype : Callable
        The function used to get a string representation of a backend-specific dtype
    available_devices : list[str]
        List of devices accepted by the backend 
    to_cpu : Callable
        Function to copy an array to the CPU
    to_numpy: Callable
        Function to create a Numpy version of an array

    """
    impl: str
    array_namespace: ModuleType
    available_dtypes: dict[str, object]
    array_type: type
    array_constructor: Callable
    make_contiguous: Callable
    identifier_of_dtype: Callable
    available_devices : list[str]
    to_cpu : Callable
    to_numpy: Callable
    _elementwise_operations: dict[str, ArrayOperation]
    def __post_init__(self):
        if self.impl in _registered_array_backends:
            raise KeyError(f"An array-backend with the identifier {self.impl} is already registered. Every backend needs to have a unique identifier.")
        _registered_array_backends[self.impl] = self
    def get_dtype_identifier(self, **kwargs) -> str:
        """
        Method for getting a dtype_identifier (str) from an array or a dtype. 
        This is used to retrieve the dtype of a custom object as a string and pass it to another backend.
        The dtype must actually be a dtype object pertaining to the `self` backend.
        Strings or Python types are not allowed here.
        Use `odl.util.dtype_utils._universal_dtype_identifier` for a general conversion from
        dtype-ish objects to identifiers.

        Parameters
        ----------
        **kwargs : 'array' or 'dtype'
            This function inputs either an array OR a dtype
        
        Returns
        -------
        dtype_identifier (str)

        Examples
        --------
        >>> odl.numpy_array_backend.get_dtype_identifier(array=np.zeros(10))
        'float64'
        >>> odl.numpy_array_backend.get_dtype_identifier(array=np.zeros(10, dtype = 'float32'))
        'float32'
        >>> odl.numpy_array_backend.get_dtype_identifier(array=np.zeros(10, float))
        'float64'
        >>> odl.numpy_array_backend.get_dtype_identifier(dtype=np.dtype('float64'))
        'float64'
        >>> odl.numpy_array_backend.get_dtype_identifier(dtype=np.zeros(10, dtype = 'float32').dtype)
        'float32'
        >>> odl.numpy_array_backend.get_dtype_identifier(dtype=np.dtype(float))
        'float64'
        >>> odl.numpy_array_backend.get_dtype_identifier(dtype=np.dtype(float), array=np.zeros(10, float))
        Traceback (most recent call last):
        AssertionError: "array and dtype are mutually exclusive parameters"
        >>> odl.numpy_array_backend.get_dtype_identifier(np.dtype(float))
        Traceback (most recent call last):
        TypeError: "ArrayBackend.get_dtype_identifier() takes 1 positional argument but 2 were given"
        """
        if 'array' in kwargs:
            assert 'dtype' not in kwargs, "array and dtype are mutually exclusive parameters"
            return self.identifier_of_dtype(kwargs['array'].dtype)
        if 'dtype' in kwargs:
            assert 'array' not in kwargs, "array and dtype are mutually exclusive parameters"
            return self.identifier_of_dtype(kwargs['dtype'])
        raise ValueError("Either 'array' or 'dtype' argument must be provided.")

    def _probe_elementwise_operation(self, operation):
        """
        Attempt to use a low-level operation in this backend. If successful, the operation is
        then registered in the `_elementwise_operations` dict in a suitable manner."""
        fn = getattr(self.array_namespace, operation)
        test_input = self.array_constructor([0,1,2])
        test_output = None
        supports_single_input = supports_two_inputs = supports_out_argument = False
        try:
            test_output = fn(test_input)
            supports_single_input = True
        except TypeError:
            pass
        try:
            test_output = fn(test_input, test_input)
            supports_two_inputs = True
        except TypeError:
            pass
        try:
            if supports_single_input:
                fn(test_input, out=test_output)
                supports_out_argument = True
            elif supports_two_inputs:
                fn(test_input, test_input, out=test_output)
                supports_out_argument = True
        except TypeError:
            pass
        if supports_single_input or supports_two_inputs:
            self._elementwise_operations[operation] = ArrayOperation(
                     name = operation,
                     supports_single_input = supports_single_input,
                     supports_two_inputs = supports_two_inputs,
                     supports_out_argument = supports_out_argument)
    
    def __repr__(self):
        """
        Implements the __repr__ method used in print.
        """
        return f"ArrayBackend(impl={self.impl})"
    
    def __eq__(self, other):
        """
        Implements the `==` operator.
        It compares if `other` is also an `ArrayBackend` and if `self` and `other` have the same implementation `impl`
        """
        return isinstance(other, ArrayBackend) and self.impl == other.impl

def lookup_array_backend(impl: str) -> ArrayBackend:
    """
    Convenience function for getting an `ArrayBackend` from an `impl` argument.
    This is helpful to both ensure that a backend actually exists and to retrieve it.

    Parameters
    ----------
    impl : str
        backend identifier

    Examples
    --------
    >>> lookup_array_backend('numpy')
    ArrayBackend(impl=numpy)
    >>> lookup_array_backend('something_else')
    Traceback (most recent call last):
    KeyError: "The implementation something_else is not supported by ODL. Please select a backend in ['numpy']"
    >>> lookup_array_backend(72)
    Traceback (most recent call last):
    AssertionError: f"The impl parameter must be a string, got int"
    """
    assert isinstance(impl, str), f"The impl parameter must be a string, got {type(impl)}"
    try:
        return _registered_array_backends[impl]
    except KeyError:
        raise KeyError(f"The implementation {impl} is not supported by ODL. Please select a backend in {_registered_array_backends.keys()}")

def get_array_and_backend(x, must_be_contiguous=False):
    """
    Convenience function for getting an `ArrayBackend` from an `array-like` argument. 

    Parameters
    ----------
    x : Array-Like.
        It can be a `np.ndarray`, a `torch.Tensor`, an ODL `Tensor` or a `ProductSpaceElement`. Object to return the `ArrayBackend` and actual underlying array from.
    must_be_contiguous : bool
        Boolean flag to indicate whether or not to make the array contiguous.

    Returns
    -------
    x : actual array 
        -> unwrapped from the LinearSpaceElement
        -> returned as is if it was already an array.
    backend : ODL `ArrayBackend` object

    Examples
    --------
    >>> array, backend = get_array_and_backend(np.zeros(2))
    >>> array
    array([ 0.,  0.])
    >>> backend
    ArrayBackend(impl=numpy)
    >>> array, backend = get_array_and_backend([1,2,3])
    Traceback (most recent call last):
    ValueError: f"The registered array backends are ['numpy']. The argument provided is a list, check that the backend you want to use is supported and has been correctly instanciated."
    """
    from odl.space.base_tensors import Tensor
    if isinstance(x, Tensor):
        return x.asarray(must_be_contiguous=must_be_contiguous), x.space.array_backend

    from odl.space.pspace import ProductSpaceElement
    if isinstance(x, ProductSpaceElement):
        return get_array_and_backend(x.asarray(), must_be_contiguous=must_be_contiguous)

    for backend in _registered_array_backends.values():
        backend : ArrayBackend
        if isinstance(x, backend.array_type) or x in backend.available_dtypes.values():
            if must_be_contiguous:
                return backend.make_contiguous(x), backend
            else:
                return x, backend

    else:
        raise ValueError(f"The registered array backends are {list(_registered_array_backends.keys())}. The argument provided is a {type(x)}, check that the backend you want to use is supported and has been correctly instanciated.")

def check_device(impl:str, device: Union[str, object]) -> str:
    """
    Checks the device argument.
    This checks that the device requested is available and that its compatible with the backend requested.

    If successful, returns the standard string identifier of the device.

    Parameters
    ----------
    impl : str
        backend identifier
    device : str or backend-specific device-object
        Device identifier

    Examples
    --------
    >>> odl.check_device('numpy', 'cpu')
    'cpu'
    >>> odl.check_device('numpy', 'anything_but_cpu')
    Traceback (most recent call last):
    ...
    ValueError: For numpy Backend, only devices ['cpu'] are present, but anything_but_cpu was provided.
    """
    backend = lookup_array_backend(impl)
    for known_device in backend.available_devices:
        if device == known_device:
            return device
        elif str(device) == known_device:
            # This works at least for PyTorch, but it is not clear
            # how general this is.
            return str(device)

    raise ValueError(f"For {impl} Backend, only devices {backend.available_devices} are present, but {device} was provided.")
    
def _dtype_info(array_namespace, dtype):
    """
    Return min, max, and kind ('bool', 'int', 'uint', 'float') for a given dtype.
    Works across Array API backends.
    """
    name = str(dtype)
    if "bool" in name:
        return 0, 1, "bool"
    if "int" in name and not "uint" in name:
        iinfo = array_namespace.iinfo(dtype)
        return iinfo.min, iinfo.max, "int"
    if "uint" in name:
        iinfo = array_namespace.iinfo(dtype)
        return iinfo.min, iinfo.max, "uint"
    if "float" in name:
        finfo = array_namespace.finfo(dtype)
        # floats have no exact min/max, but finfo.min/max are usable for range checks
        return finfo.min, finfo.max, "float"
    raise ValueError(f"Unsupported dtype: {dtype}")

def can_cast(array_namespace, from_dtype, to_dtype, casting="safe"):
    """
    NumPy-like can_cast for Python Array API backends.
    Supports 'safe', 'same_kind', and 'unsafe' casting.
    """
    # Convert arrays to dtypes
    if hasattr(from_dtype, "dtype"):
        from_dtype = from_dtype.dtype
    if hasattr(to_dtype, "dtype"):
        to_dtype = to_dtype.dtype

    # Same type always allowed
    if from_dtype == to_dtype:
        return True

    # Unsafe allows anything
    if casting == "unsafe":
        return True

    # Determine type categories
    f_min, f_max, f_kind = _dtype_info(array_namespace, from_dtype)
    t_min, t_max, t_kind = _dtype_info(array_namespace, to_dtype)

    # Safe casting: all values of from_dtype must fit in to_dtype
    if casting == "safe":
        if f_kind == "bool":
            return True  # bool -> anything is safe
        if t_kind == "bool":
            return False  # non-bool -> bool is unsafe
        if f_kind in ("int", "uint") and t_kind in ("int", "uint", "float"):
            return f_min >= t_min and f_max <= t_max
        if f_kind == "float" and t_kind == "float":
            return array_namespace.finfo(to_dtype).precision >= array_namespace.finfo(from_dtype).precision
        return False

    # Same-kind casting: allow within same category or safe upcast to float
    if casting == "same_kind":
        if f_kind == t_kind:
            return True
        # int/uint to float is allowed if range fits
        if f_kind in ("int", "uint") and t_kind == "float":
            return f_min >= t_min and f_max <= t_max
        return False

    raise ValueError(f"Unsupported casting rule: {casting}")

if __name__ =='__main__':
    check_device('numpy', 'cpu')
