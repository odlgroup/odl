# This is an attempt to progressively tidy the 'utility.py' module, which is little more than a heap of unstable/unsupported code waiting to crumble.

# Python imports
from numbers import Number
from functools import lru_cache
# Third-Party import
import array_api_compat as xp
# ODL imports
from odl.array_API_support import lookup_array_backend

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
BOOLEAN_DTYPES = [
    bool,
    "bool"
    ]

SIGNED_INTEGER_DTYPES = [
    int,
    "int8",
    "int16",
    "int32",
    "int64",
]
UNSIGNED_INTEGER_DTYPES = [
    "uint8",
    "uint16",
    "uint32",
    "uint64"
]

INTEGER_DTYPES = SIGNED_INTEGER_DTYPES + UNSIGNED_INTEGER_DTYPES

FLOAT_DTYPES = [
    float,
    "float32",
    "float64"
]

COMPLEX_DTYPES = [
    complex,
    "complex64",
    "complex128"
]

REAL_DTYPES = INTEGER_DTYPES + FLOAT_DTYPES
SCALAR_DTYPES = REAL_DTYPES + COMPLEX_DTYPES
AVAILABLE_DTYPES = BOOLEAN_DTYPES + REAL_DTYPES + COMPLEX_DTYPES 

"""
See type promotion rules https://data-apis.org/array-api/latest/API_specification/type_promotion.html#type-promotion
"""

TYPE_PROMOTION_REAL_TO_COMPLEX = {
    int : "complex64",
    float : "complex128",
    "int8"  : "complex64",
    "int16" : "complex64",
    "int32" : "complex64",
    "int64" : "complex64",
    "uint8" : "complex64",
    "uint16" : "complex64",
    "uint32"  : "complex128",
    "uint64"  : "complex128",
    "float32" : "complex64",
    "float64" : "complex128"
}

TYPE_PROMOTION_COMPLEX_TO_REAL = {
    complex : "float64",
    "complex64"  : "float32",
    "complex128" : "float64"
}

# These dicts should not be exposed to the users/developpers outside of the module. We rather provide functions that rely on the available array_backends present
def _convert_dtype(dtype: "str | Number |xp.dtype") -> str :
    """
    Internal helper function to convert a dtype to a string. The dtype can be provided as a string, a python Number or as a xp.dtype. 
    Returns: 
    dtype_as_str (str), dtype identifier 
    Note:
    xp is written here for type hinting, it refers to the fact that the dtype can be provided as a np.float32 or as a torchfloat32, for instance.
    """
    # Lazy import 
    from odl.space.entry_points import TENSOR_SPACE_IMPLS
    if isinstance(dtype, (str, Number, type)):
        assert dtype in AVAILABLE_DTYPES, f'The provided dtype {dtype} is not available. Please use a dtype in {AVAILABLE_DTYPES}'
        return dtype
    for impl in TENSOR_SPACE_IMPLS:
        array_backend = lookup_array_backend(impl)
        if dtype in array_backend.available_dtypes.values():
            return array_backend.identifier_of_dtype(dtype)
    raise ValueError(f'The provided dtype {dtype} is not a string, a python Number or a backend-specific dtype. Please provide either of these.')

@lru_cache
def is_available_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is available."""
    try: 
        _convert_dtype(dtype)
        return True
    except ValueError or AssertionError:
        return False 
    
@lru_cache
def is_numeric_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a numeric type."""
    return _convert_dtype(dtype) in SCALAR_DTYPES

@lru_cache
def is_boolean_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is an boolean type."""
    return _convert_dtype(dtype) in BOOLEAN_DTYPES

@lru_cache
def is_signed_int_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is an integer type."""
    return _convert_dtype(dtype) in SIGNED_INTEGER_DTYPES

@lru_cache
def is_unsigned_int_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is an integer type."""
    return _convert_dtype(dtype) in UNSIGNED_INTEGER_DTYPES

@lru_cache
def is_int_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is an integer type."""
    return _convert_dtype(dtype) in INTEGER_DTYPES

@lru_cache
def is_floating_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a floating point type."""
    return _convert_dtype(dtype) in FLOAT_DTYPES + COMPLEX_DTYPES

@lru_cache
def is_real_floating_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a floating point type."""
    return _convert_dtype(dtype) in FLOAT_DTYPES

@lru_cache
def is_complex_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a complex type."""
    return _convert_dtype(dtype) in COMPLEX_DTYPES

@lru_cache
def is_real_dtype(dtype: "str | Number |xp.dtype") -> bool:
    """Return ``True`` if ``dtype`` is a real (including integer) type."""
    return _convert_dtype(dtype) in REAL_DTYPES

def real_dtype(dtype: "str | Number |xp.dtype", default=None) -> str:
    """
    Returns the real counterpart of ``dtype`` if it exists
    Parameters
    ----------
    dtype :
        Input dtype
    default :
        Object to be returned if no real counterpart is found for
        ``dtype``, except for ``None``, in which case an error is raised.
    """
    dtype = _convert_dtype(dtype)
    if dtype in REAL_DTYPES:
        return dtype
    elif dtype in COMPLEX_DTYPES:
        return TYPE_PROMOTION_COMPLEX_TO_REAL[dtype]
    else:
        if default is None:
            raise ValueError(
                f"no real counterpart exists for `dtype` {dtype}")
        else:
            return default
        
def complex_dtype(dtype: "str | Number |xp.dtype", default=None) -> str:
    dtype = _convert_dtype(dtype)
    if dtype in COMPLEX_DTYPES:
        return dtype
    elif dtype in REAL_DTYPES:
        return TYPE_PROMOTION_REAL_TO_COMPLEX[dtype]
    else:
        if default is None:
            raise ValueError(
                f"no complex counterpart exists for `dtype` {dtype}")
        else:
            return default

