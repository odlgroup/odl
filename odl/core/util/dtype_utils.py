# This is an attempt to progressively tidy the 'utility.py' module, which is little more than a heap of unstable/unsupported code waiting to crumble.

# Python imports
from numbers import Number
from functools import lru_cache
# Third-Party import
import array_api_compat as xp
# ODL imports
from odl.core.array_API_support import ArrayBackend, lookup_array_backend
from odl.core.array_API_support.utils import _registered_array_backends
from typing import Optional

__all__ = (
    'is_available_dtype',
    'is_numeric_dtype',
    'is_boolean_dtype',
    'is_int_dtype',
    'is_signed_int_dtype',
    'is_unsigned_int_dtype',
    'is_floating_dtype',
    'is_real_floating_dtype',
    'is_complex_dtype',
    'is_real_dtype',
    'real_dtype',
    'complex_dtype'
)

############################# DATA TYPES #############################
# We store all the data types expected by the python array API as lists, and  the maps for conversion as dicts
BOOLEAN_DTYPES = ['bool']

SIGNED_INTEGER_DTYPES = [
    'int8',
    'int16',
    'int32',
    'int64',
]
UNSIGNED_INTEGER_DTYPES = [
    'uint8',
    'uint16',
    'uint32',
    'uint64'
]

INTEGER_DTYPES = SIGNED_INTEGER_DTYPES + UNSIGNED_INTEGER_DTYPES

FLOAT_DTYPES = [
    'float32',
    'float64'
]

COMPLEX_DTYPES = [
    'complex64',
    'complex128'
]

REAL_DTYPES = INTEGER_DTYPES + FLOAT_DTYPES
SCALAR_DTYPES = REAL_DTYPES + COMPLEX_DTYPES
AVAILABLE_DTYPES = BOOLEAN_DTYPES + REAL_DTYPES + COMPLEX_DTYPES

"""
See type promotion rules https://data-apis.org/array-api/latest/API_specification/type_promotion.html#type-promotion
"""

TYPE_PROMOTION_REAL_TO_COMPLEX = {
    'int8'  : 'complex64',
    'int16' : 'complex64',
    'int32' : 'complex64',
    'int64' : 'complex64',
    'uint8' : 'complex64',
    'uint16' : 'complex64',
    'uint32'  : 'complex128',
    'uint64'  : 'complex128',
    'float32' : 'complex64',
    'float64' : 'complex128'
}

TYPE_PROMOTION_COMPLEX_TO_REAL = {
    'complex64'  : 'float32',
    'complex128' : 'float64'
}

DTYPE_SHORTHANDS = {
    bool: 'bool',
    int: 'int32',
    float: 'float64',
    complex: 'complex128'
}

# These dicts should not be exposed to the users/developpers outside of the module. We rather provide functions that rely on the available array_backends present
def _universal_dtype_identifier(
    dtype: "str | Number |xp.dtype", array_backend_selection: list[ArrayBackend] = None
) -> str:
    """
    Internal helper function to convert a dtype to a backend-agnostic string identifying it semantically.
    (E.g. `'int32'` and `'int64'` and `'float64'` are all possible distinct results, but `np.float64` and
    `torch.float64` and `float` all map to the unique identifier `'float64'`.)
    ambiguity
    The dtype can be provided as a string, a python Number or as an xp.dtype.
    Returns:
    dtype_as_str (str), dtype identifier
    Note:
    xp is written here for type hinting, it refers to the fact that the dtype can be provided as a np.float32 or as a torchfloat32, for instance.
    What concrete types of dtype are allowed is determined by `array_backend_selection`.
    If that argument is not provided, all registered backends are taken into consideration.
    """
    # Lazy import 
    from odl.core.space.entry_points import TENSOR_SPACE_IMPLS

    original_dtype = dtype
    shorthand_elaboration = ""
    if dtype in DTYPE_SHORTHANDS:
        dtype = DTYPE_SHORTHANDS[dtype]
        shorthand_elaboration = f" (shorthand for {dtype})"

    if isinstance(dtype, (str, Number, type)):
        if dtype in AVAILABLE_DTYPES:
            return dtype

        raise TypeError(
            f"The provided dtype {original_dtype}{shorthand_elaboration} is not available. Please use a dtype in {AVAILABLE_DTYPES}")

    if array_backend_selection is None:
        array_backends = _registered_array_backends.values()
    else:
        array_backends = array_backend_selection
    for array_backend in array_backends:
        if dtype in array_backend.available_dtypes.values():
            return array_backend.identifier_of_dtype(dtype)

    raise ValueError(
        f"The provided dtype {dtype} is not a string, a python Number or a"
        + f" backend-specific dtype of {[be.impl for be in array_backends]}."
        + " Please provide either of these.")

@lru_cache
def is_available_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is available."""
    try:
        _universal_dtype_identifier(dtype)
        return True
    except (ValueError, AssertionError):
        return False


@lru_cache
def is_numeric_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a numeric type."""
    return _universal_dtype_identifier(dtype) in SCALAR_DTYPES


@lru_cache
def is_boolean_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is an boolean type."""
    return _universal_dtype_identifier(dtype) in BOOLEAN_DTYPES


@lru_cache
def is_signed_int_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is an integer type."""
    return _universal_dtype_identifier(dtype) in SIGNED_INTEGER_DTYPES


@lru_cache
def is_unsigned_int_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is an integer type."""
    return _universal_dtype_identifier(dtype) in UNSIGNED_INTEGER_DTYPES


@lru_cache
def is_int_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is an integer type."""
    return _universal_dtype_identifier(dtype) in INTEGER_DTYPES


@lru_cache
def is_floating_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a floating point type."""
    return _universal_dtype_identifier(dtype) in FLOAT_DTYPES + COMPLEX_DTYPES


@lru_cache
def is_real_floating_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a floating point type."""
    return _universal_dtype_identifier(dtype) in FLOAT_DTYPES


@lru_cache
def is_complex_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a complex type."""
    return _universal_dtype_identifier(dtype) in COMPLEX_DTYPES


@lru_cache
def is_real_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a real (including integer) type."""
    return _universal_dtype_identifier(dtype) in REAL_DTYPES


def real_dtype(
    dtype: "str | Number |xp.dtype",
    default=None,
    backend: Optional[ArrayBackend] = None,
) -> str:
    """
    Returns the real counterpart of ``dtype`` if it exists
    Parameters
    ----------
    dtype :
        Input dtype
    default :
        Object to be returned if no real counterpart is found for
        ``dtype``, except for ``None``, in which case an error is raised.
    backend :
        If given, the result dtype will be returned in its version
        specific to that backend (e.g. `torch.float32`), otherwise as a plain string.
    """
    dtype = _universal_dtype_identifier(dtype)

    def for_backend(dt):
        if backend is None:
            return dt
        else:
            try:
                return backend.available_dtypes[dt]
            except KeyError:
                raise ValueError(f"Real version of {dtype} not available on {backend}.")
    if dtype in REAL_DTYPES:
        return for_backend(dtype)
    elif dtype in COMPLEX_DTYPES:
        return for_backend(TYPE_PROMOTION_COMPLEX_TO_REAL[dtype])
    else:
        if default is None:
            raise ValueError(
                f"no real counterpart exists for `dtype` {dtype}")
        else:
            return default
        
def complex_dtype(
    dtype: "str | Number |xp.dtype",
    default=None,
    backend: Optional[ArrayBackend] = None,
) -> str:
    """
    Returns the complex counterpart of ``dtype`` if it exists
    Parameters
    ----------
    dtype :
        Input dtype
    default :
        Object to be returned if no complex counterpart is found for
        ``dtype``. If ``None``, an error is raised in this case.
    backend :
        If given, the result dtype will be returned in its version
        specific to that backend (e.g. `torch.complex64`), otherwise as a plain string.
    """
    dtype = _universal_dtype_identifier(dtype)

    def for_backend(dt):
        if backend is None:
            return dt
        else:
            try:
                return backend.available_dtypes[dt]
            except KeyError:
                raise ValueError(f"Complex version of {dtype} not available on {backend}.")
    if dtype in COMPLEX_DTYPES:
        return for_backend(dtype)
    elif dtype in REAL_DTYPES:
        return for_backend(TYPE_PROMOTION_REAL_TO_COMPLEX[dtype])
    else:
        if default is None:
            raise ValueError(
                f"no complex counterpart exists for `dtype` {dtype}")
        else:
            return default

