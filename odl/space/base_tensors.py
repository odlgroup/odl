# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Base classes for implementations of tensor spaces."""

from __future__ import absolute_import, division, print_function

from types import ModuleType
from typing import Dict
from numbers import Integral, Number
import warnings
import numpy as np

import odl
from odl.set.sets import ComplexNumbers, RealNumbers
from odl.set.space import (
    LinearSpace, LinearSpaceElement, LinearSpaceTypeError,
    SupportedNumOperationParadigms, NumOperationParadigmSupport)
from odl.util.vectorization import ArrayBackend, lookup_array_backend
from odl.util import (
    array_str, indent, is_complex_floating_dtype,
    is_numeric_dtype, is_real_floating_dtype, safe_int_conv,
    signature_string)
from odl.util.utility import(
    SCALAR_DTYPES, AVAILABLE_DTYPES,
    FLOAT_DTYPES, COMPLEX_DTYPES,
    TYPE_PROMOTION_COMPLEX_TO_REAL, 
    TYPE_PROMOTION_REAL_TO_COMPLEX)
from .weighting import Weighting

__all__ = ('TensorSpace',)

class TensorSpace(LinearSpace):

    """Base class for sets of tensors of arbitrary data type.

    A tensor is, in the most general sense, a multi-dimensional array
    that allows operations per entry (keep the rank constant),
    reductions / contractions (reduce the rank) and broadcasting
    (raises the rank).
    For non-numeric data type like ``object``, the range of valid
    operations is rather limited since such a set of tensors does not
    define a vector space.
    Any numeric data type, on the other hand, is considered valid for
    a tensor space, although certain operations - like division with
    integer dtype - are not guaranteed to yield reasonable results.

    Under these restrictions, all basic vector space operations are
    supported by this class, along with reductions based on arithmetic
    or comparison, and element-wise mathematical functions ("ufuncs").

    See the `Wikipedia article on tensors`_ for further details.
    See also [Hac2012] "Part I Algebraic Tensors" for a rigorous
    treatment of tensors with a definition close to this one.

    Note also that this notion of tensors is the same as in popular
    Deep Learning frameworks.

    References
    ----------
    [Hac2012] Hackbusch, W. *Tensor Spaces and Numerical Tensor Calculus*.
    Springer, 2012.

    .. _Wikipedia article on tensors: https://en.wikipedia.org/wiki/Tensor
    """

    def __init__(self, shape, dtype, device, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        shape : nonnegative int or sequence of nonnegative ints
            Number of entries of type ``dtype`` per axis in this space. A
            single integer results in a space with rank 1, i.e., 1 axis.
        dtype :
            Data type of elements in this space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string.
            For a data type with a ``dtype.shape``, these extra dimensions
            are added *to the left* of ``shape``.
        """
        # Handle shape and dtype, taking care also of dtypes with shape        
        self._init_dtype(dtype)

        self._init_shape(shape, dtype)

        self._init_device(device)

        self.__use_in_place_ops = kwargs.pop('use_in_place_ops', True)
  
        self._init_weighting(**kwargs)

        field = self._init_field()

        LinearSpace.__init__(self, field)

    ################ Init Methods, Non static ################
    def _init_device(self, device:str):
        odl.check_device(self.impl, device)
        self.__device = device 

    def _init_dtype(self, dtype:str | int | float | complex):
        """
        Process the dtype argument. This parses the (str or Number) dtype input argument to a backend.dtype and sets two attributes

        self.dtype_identifier (str)      -> Used for passing dtype information from one backend to another
        self.__dtype (backend.dtype) -> Actual dtype of the TensorSpace implementation

        Note:
        The check below is here just in case a user initialise a space directly from this class, which is not recommended
        """

        available_dtypes = self.array_backend.available_dtypes

        ### We check if the datatype has been provided in a "sane" way, 
        # 1) a Python scalar type
        if isinstance(dtype, (int, float, complex)):
            self.__dtype_identifier = str(dtype)
            self.__dtype = available_dtypes[dtype] 
        # 2) as a string
        if dtype in available_dtypes.keys():
            self.__dtype_identifier = dtype
            self.__dtype = available_dtypes[dtype]
        ### If the check has failed, i.e the dtype is not a Key of the available_dtypes dict or a python scalar, we try to parse the dtype 
        ### as a string using the self.get_dtype_identifier(dtype=dtype) call: This is for the situation where the dtype passed is
        ### in the .values() of available_dtypes dict (something like 'numpy.float32')
        elif dtype in available_dtypes.values():
            self.__dtype_identifier = self.get_dtype_identifier(dtype=dtype)
            self.__dtype = dtype
            # If that fails, we throw an error: the dtype is not a python scalar dtype, not a string describing the dtype or the 
            # backend call to parse the dtype has failed.
        else:
            raise ValueError(f"The dtype must be in {available_dtypes.keys()} or must be a dtype of the backend, but {dtype} was provided")

    def _init_shape(self, shape, dtype):
        # Handle shape and dtype, taking care also of dtypes with shape
        try:
            shape, shape_in = tuple(safe_int_conv(s) for s in shape), shape
        except TypeError:
            shape, shape_in = (safe_int_conv(shape),), shape
        if any(s < 0 for s in shape):
            raise ValueError(
                "`shape` must have only nonnegative entries, got " "{}".format(shape_in)
            )

        # We choose this order in contrast to Numpy, since we usually want
        # to represent discretizations of vector- or tensor-valued functions,
        # i.e., if dtype.shape == (3,) we expect f[0] to have shape `shape`.
        # <!> this is likely to break in Pytorch
        self.__shape = np.dtype(dtype).shape + shape

    def _init_field(self):
        if self.dtype_identifier in TYPE_PROMOTION_REAL_TO_COMPLEX:
            # real includes non-floating-point like integers
            field = RealNumbers()
            self.__real_dtype = self.dtype
            self.__real_space = self
            self.__complex_dtype = self.array_backend.available_dtypes[
                TYPE_PROMOTION_REAL_TO_COMPLEX[self.dtype_identifier]
            ]
            
            self.__complex_space = None  # Set in first call of astype
        elif self.dtype_identifier in TYPE_PROMOTION_COMPLEX_TO_REAL:
            field = ComplexNumbers()
            self.__real_dtype = self.array_backend.available_dtypes[
                TYPE_PROMOTION_COMPLEX_TO_REAL[self.dtype_identifier]
            ]
            self.__real_space = None  # Set in first call of astype
            self.__complex_dtype = self.dtype
            self.__complex_space = self
        else:
            field = None
        return field
    
    def _init_weighting(self, **kwargs):
        weighting = kwargs.pop("weighting", None)    
        if weighting is None:
            self.__weighting = odl.space_weighting(impl=self.impl, device=self.device, **kwargs)
        else:
            if issubclass(type(weighting), Weighting):
                if weighting.impl != self.impl:
                    raise ValueError(
                        f"`weighting.impl` and space.impl must be consistent, but got \
                        {weighting.impl} and {self.impl}" 
                    )
                if weighting.device != self.device:
                    raise ValueError(
                        f"`weighting.device` and space.device must be consistent, but got \
                        {weighting.device} and {self.device}" 
                    )
                self.__weighting = weighting
                if weighting.shape and weighting.shape != self.shape:
                    raise ValueError(
                        f"`weighting.shape` and space.shape must be consistent, but got \
                        {weighting.shape} and {self.shape}" 
                    )
            elif hasattr(weighting, '__array__') or isinstance(weighting, (int, float)):
                self.__weighting = odl.space_weighting(impl=self.impl, device=self.device, weight=weighting, **kwargs)
            else:
                raise TypeError(
                    f"""Wrong type of 'weighting' argument. Only floats, array-like and odl.Weightings are accepted 
                    """
                    )
    
    ########## Attributes ##########
    @property
    def array_backend(self) -> ArrayBackend:
        return lookup_array_backend(self.impl)

    @property
    def array_namespace(self) -> ModuleType:
        """Name of the array_namespace of this tensor set. This relates to the
        python array api.
        """
        return self.array_backend.array_namespace
    
    @property
    def byaxis(self):
        """Return the subspace defined along one or several dimensions.

        Examples
        --------
        Indexing with integers or slices:

        >>> space = odl.rn((2, 3, 4))
        >>> space.byaxis[0]
        rn(2)
        >>> space.byaxis[1:]
        rn((3, 4))

        Lists can be used to stack spaces arbitrarily:

        >>> space.byaxis[[2, 1, 2]]
        rn((4, 3, 4))
        """
        space = self

        class TensorSpacebyaxis(object):

            """Helper class for indexing by axis."""

            def __getitem__(self, indices):
                """Return ``self[indices]``."""
                try:
                    iter(indices)
                except TypeError:
                    newshape = space.shape[indices]
                else:
                    newshape = tuple(space.shape[i] for i in indices)

                return type(space)(newshape, space.dtype, weighting=space.weighting)

            def __repr__(self):
                """Return ``repr(self)``."""
                return repr(space) + '.byaxis'

        return TensorSpacebyaxis()
    
    @property
    def complex_dtype(self):
        """The complex dtype corresponding to this space's `dtype`.

        Raises
        ------
        NotImplementedError
            If `dtype` is not a numeric data type.
        """
        if not is_numeric_dtype(self.dtype):
            raise NotImplementedError(
                '`complex_dtype` not defined for non-numeric `dtype`')
        return self.__complex_dtype
    
    @property
    def complex_space(self):
        """The space corresponding to this space's `complex_dtype`.

        Raises
        ------
        ValueError
            If `dtype` is not a numeric data type.
        """
        if not is_numeric_dtype(self.dtype):
            raise ValueError(
                '`complex_space` not defined for non-numeric `dtype`')
        return self.astype(self.complex_dtype)
    
    @property
    def device(self):
        """Device on which the tensorSpace is implemented.

        This property should be overridden by subclasses.
        """
        return self.__device
    
    @property
    def dtype(self):
        """Scalar data type of each entry in an element of this space."""
        return self.__dtype
    
    @property
    def dtype_identifier(self):
        """Scalar data type of each entry in an element of this space."""
        return self.__dtype_identifier
    
    @property
    def element_type(self):
        """Type of elements in this space: `Tensor`."""
        return Tensor
    
    @property
    def examples(self):
        """Return example random vectors."""
        # Always return the same numbers
        rand_state = np.random.get_state()
        np.random.seed(1337)

        if is_numeric_dtype(self.dtype):
            yield ('Linearly spaced samples', self.element(
                np.linspace(0, 1, self.size).reshape(self.shape)))
            yield ('Normally distributed noise',
                   self.element(np.random.standard_normal(self.shape)))

        if self.is_real:
            yield ('Uniformly distributed noise',
                   self.element(np.random.uniform(size=self.shape)))
        elif self.is_complex:
            yield ('Uniformly distributed noise',
                   self.element(np.random.uniform(size=self.shape) +
                                np.random.uniform(size=self.shape) * 1j))
        else:
            # TODO: return something that always works, like zeros or ones?
            raise NotImplementedError('no examples available for non-numeric'
                                      'data type')

        np.random.set_state(rand_state)

    @property
    def exponent(self):
        """Exponent of the norm and the distance."""
        return self.weighting.exponent
    
    @property
    def impl(self):
        """Name of the implementation back-end of this tensor set.

        This property should be overridden by subclasses.
        """
        raise NotImplementedError('abstract method')
    
    @property
    def itemsize(self):
        """Size in bytes of one entry in an element of this space."""
        return  int(self.array_backend.array_constructor([], dtype=self.dtype).itemsize)
    
    @property
    def is_complex(self):
        """True if this is a space of complex tensors."""
        return is_complex_floating_dtype(self.dtype)
    
    @property
    def is_real(self):
        """True if this is a space of real tensors."""
        return is_real_floating_dtype(self.dtype)
    
    @property
    def is_weighted(self):
        """Return ``True`` if the space is not weighted by constant 1.0."""
        return self.weighting.__weight != 1.0
        
    @property
    def nbytes(self):
        """Total number of bytes in memory used by an element of this space."""
        return self.size * self.itemsize
    
    @property
    def ndim(self):
        """Number of axes (=dimensions) of this space, also called "rank"."""
        return len(self.shape)

    @property
    def real_dtype(self):
        """The real dtype corresponding to this space's `dtype`.

        Raises
        ------
        NotImplementedError
            If `dtype` is not a numeric data type.
        """
        if not is_numeric_dtype(self.dtype):
            raise NotImplementedError(
                '`real_dtype` not defined for non-numeric `dtype`')
        return self.__real_dtype
    
    @property
    def real_space(self):
        """The space corresponding to this space's `real_dtype`.

        Raises
        ------
        ValueError
            If `dtype` is not a numeric data type.
        """
        if not is_numeric_dtype(self.dtype):
            raise ValueError(
                '`real_space` not defined for non-numeric `dtype`')
        return self.astype(self.real_dtype)
    
    @property
    def supported_num_operation_paradigms(self) -> NumOperationParadigmSupport:
        """NumPy has full support for in-place operation, which is usually
        advantageous to reduce memory allocations.
        This can be deactivated, mostly for testing purposes, by setting
        `use_in_place_ops = False` when constructing the space."""
        if self.__use_in_place_ops:
            return SupportedNumOperationParadigms(
                    in_place = NumOperationParadigmSupport.PREFERRED,
                    out_of_place = NumOperationParadigmSupport.SUPPORTED)
        else:
            return SupportedNumOperationParadigms(
                    in_place = NumOperationParadigmSupport.NOT_SUPPORTED,
                    out_of_place = NumOperationParadigmSupport.PREFERRED)
        
    @property
    def shape(self):
        """Number of scalar elements per axis.

        .. note::
            If `dtype` has a shape, we add it to the **left** of the given
            ``shape`` in the class creation. This is in contrast to NumPy,
            which adds extra axes to the **right**. We do this since we
            usually want to represent discretizations of vector- or
            tensor-valued functions by this, i.e., if
            ``dtype.shape == (3,)`` we expect ``f[0]`` to have shape
            ``shape``.
        """
        return self.__shape
    
    @property
    def size(self):
        """Total number of entries in an element of this space."""
        return (0 if self.shape == () else
                int(np.prod(self.shape, dtype='int64')))
    
    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.__weighting

    ########## public methods ##########
    def astype(self, dtype):
        """Return a copy of this space with new ``dtype``.

        Parameters
        ----------
        dtype :
            Scalar data type of the returned space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string. Data types with non-trivial
            shapes are not allowed.

        Returns
        -------
        newspace : `TensorSpace`
            Version of this space with given data type.
        """
        if dtype is None:
            # Need to filter this out since Numpy iterprets it as 'float'
            raise ValueError('`None` is not a valid data type')

        available_dtypes = self.array_backend.available_dtypes
        
        ### We check if the datatype has been provided in a "sane" way, 
        # 1) a Python scalar type
        if isinstance(dtype, (int, float, complex)):
            dtype_identifier = str(dtype)
            dtype = available_dtypes[dtype]
        # 2) as a string
        elif dtype in available_dtypes.keys():
            dtype_identifier = dtype
            dtype = available_dtypes[dtype]
        ### If the check has failed, i.e the dtype is not a Key of the available_dtypes dict or a python scalar, we try to parse the dtype 
        ### as a string using the self.get_dtype_identifier(dtype=dtype) call: This is for the situation where the dtype passed is
        ### in the .values() of available_dtypes dict (something like 'numpy.float32')
        elif self.get_dtype_identifier(dtype=dtype) in available_dtypes:
            dtype_identifier = self.get_dtype_identifier(dtype=dtype)
            dtype = available_dtypes[dtype_identifier]
            # If that fails, we throw an error: the dtype is not a python scalar dtype, not a string describing the dtype or the 
            # backend call to parse the dtype has failed.
        else:
            raise ValueError(f"The dtype must be in {self.available_dtypes.keys()} or must be a dtype of the backend, but {dtype} was provided")

        # try:
        #     dtype_identifier = dtype
        #     dtype = available_dtypes[dtype]
        # except KeyError:
        #     raise KeyError(f"The dtype must be in {available_dtypes.keys()}, but {dtype} was provided")
        
        if dtype == self.dtype:
            return self

        if dtype_identifier in FLOAT_DTYPES + COMPLEX_DTYPES:
            # Caching for real and complex versions (exact dtype mappings)
            if dtype == self.__real_dtype:
                if self.__real_space is None:
                    self.__real_space = self._astype(dtype_identifier)
                return self.__real_space
            elif dtype == self.__complex_dtype:
                if self.__complex_space is None:
                    self.__complex_space = self._astype(dtype_identifier)
                return self.__complex_space
            else:
                return self._astype(dtype_identifier)
        else:
            return self._astype(dtype_identifier)
        
    def default_dtype(self, field=None):
        """Return the default data type for a given field.

        This method should be overridden by subclasses.

        Parameters
        ----------
        field : `Field`, optional
            Set of numbers to be represented by a data type.
            Currently supported : `RealNumbers`, `ComplexNumbers`
            The default ``None`` means `RealNumbers`

        Returns
        -------
        dtype :
            Backend data type specifier.
        """
        if field is None or field == RealNumbers():
            return self.array_backend.available_dtypes['float32']
        elif field == ComplexNumbers():
           return self.array_backend.available_dtypes['complex64']
        else:
            raise ValueError('no default data type defined for field {}'
                             ''.format(field))
        
    def element(self, inp=None, device=None, copy=True):
        def wrapped_array(arr):
            if arr.shape != self.shape:
                raise ValueError(
                    "shape of `inp` not equal to space shape: "
                    "{} != {}".format(arr.shape, self.shape)
                )
            
            return self.element_type(self, arr)


        def dlpack_transfer(arr, device=None, copy=True):
            # We check that the object implements the dlpack protocol:
            # assert hasattr(inp, "__dlpack_device__") and hasattr(
            #     arr, "__dlpack__"
            # ), """The input does not support the DLpack framework. 
            #     Please convert it to an object that supports it first. 
            # (cf:https://data-apis.org/array-api/latest/purpose_and_scope.html)"""
            try:
                # from_dlpack(inp, device=device, copy=copy)
                # As of Pytorch 2.7, the pytorch API from_dlpack does not implement the
                # keywords that specify the device and copy arguments
                return self.array_namespace.from_dlpack(arr)
            except BufferError:
                raise BufferError(
                    "The data cannot be exported as DLPack (e.g., incompatible dtype, strides, or device). "
                    "It may also be that the export fails for other reasons "
                    "(e.g., not enough memory available to materialize the data)."
                    ""
                )
            except ValueError:
                raise ValueError(
                    "The data exchange is possible via an explicit copy but copy is set to False."
                )
            ### This is a temporary fix, until pytorch provides the right API for dlpack with args!!
            # The RuntimeError should be raised only when using a GPU device 
            except RuntimeError:
                if self.impl == 'numpy':
                    # if isinstance(arr, torch.Tensor):
                    #     arr = arr.detach().cpu()
                    return np.asarray(arr, dtype=self.dtype)
                # elif self.impl == 'pytorch':                    
                #     return torch.asarray(arr, device=self.device, dtype=self.dtype)
                    
                else:
                    raise NotImplementedError

        # Case 1: no input provided
        if inp is None:
            return wrapped_array(
                self.array_namespace.empty(
                    self.shape, dtype=self.dtype, device=self.device
                )
            )
        # Case 2: input is provided
        # Case 2.1: the input is an ODL OBJECT
        # ---> The data of the input is transferred to the space's device and data type AND wrapped into the space.
        if hasattr(inp, "odl_tensor"):
            return wrapped_array(dlpack_transfer(inp.data, device, copy))
        # Case 2.2: the input is an object that implements the python array aPI (np.ndarray, torch.Tensor...)
        # ---> The input is transferred to the space's device and data type AND wrapped into the space.
        elif hasattr(inp, '__array__'):
            return wrapped_array(dlpack_transfer(inp, device, copy))
        # Case 2.3: the input is an array like object [[1,2,3],[4,5,6],...]
        # ---> The input is transferred to the space's device and data type AND wrapped into the space.
        # TODO: Add the iterable type instead of list and tuple and the numerics type instead of int, float, complex
        elif isinstance(inp, (int, float, complex, list, tuple)):
            return wrapped_array(
                self.array_namespace.broadcast_to(
                    self.array_namespace.asarray(inp, device=self.device),
                    self.shape
                    )
                )
        else:
            raise ValueError  
        
    def finfo(self):
        "Machine limits for floating-point data types."
        return self.array_namespace.finfo(self.dtype)
    
    def iinfo(self):
        "Machine limits for integer data types."
        return self.array_namespace.iinfo(self.dtype)
        
    def divide(self, x1, x2, out=None):
        return self._divide(x1, x2, out)
    
    def multiply(self, x1, x2, out=None):
        return self._multiply(x1, x2, out)    
    
    def one(self):
        """Return a tensor of all ones.

        This method should be overridden by subclasses.

        Returns
        -------
        one : `Tensor`
            A tensor of all one.
        """
        return self.element(
            self.array_namespace.ones(self.shape, dtype=self.dtype, device=self.device)
        )
    
    def zero(self):
        """Return a tensor of all zeros.

        This method should be overridden by subclasses.

        Returns
        -------
        zero : `Tensor`
            A tensor of all zeros.
        """
        return self.element(
            self.array_namespace.zeros(self.shape, dtype=self.dtype, device=self.device)
        )

    ######### magic methods #########
    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contains : bool
            ``True`` if ``other`` has a ``space`` attribute that is equal
            to this space, ``False`` otherwise.

        Examples
        --------
        Elements created with the `TensorSpace.element` method are
        guaranteed to be contained in the same space:

        >>> spc = odl.tensor_space((2, 3), dtype='uint64')
        >>> spc.element() in spc
        True
        >>> x = spc.element([[0, 1, 2],
        ...                  [3, 4, 5]])
        >>> x in spc
        True

        Sizes, data types and other essential properties characterize
        spaces and decide about membership:

        >>> smaller_spc = odl.tensor_space((2, 2), dtype='uint64')
        >>> y = smaller_spc.element([[0, 1],
        ...                          [2, 3]])
        >>> y in spc
        False
        >>> x in smaller_spc
        False
        >>> other_dtype_spc = odl.tensor_space((2, 3), dtype='uint32')
        >>> z = other_dtype_spc.element([[0, 1, 2],
        ...                              [3, 4, 5]])
        >>> z in spc
        False
        >>> x in other_dtype_spc
        False

        On the other hand, spaces are not unique:

        >>> spc2 = odl.tensor_space((2, 3), dtype='uint64')
        >>> spc2 == spc
        True
        >>> x2 = spc2.element([[5, 4, 3],
        ...                    [2, 1, 0]])
        >>> x2 in spc
        True
        >>> x in spc2
        True

        Of course, random garbage is not in the space:

        >>> spc = odl.tensor_space((2, 3), dtype='uint64')
        >>> None in spc
        False
        >>> object in spc
        False
        >>> False in spc
        False
        """
        return getattr(other, 'space', None) == self

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            True if ``self`` and ``other`` have the same type, `shape`
            and `dtype`, otherwise ``False``.

        Examples
        --------
        Sizes, data types and other essential properties characterize
        spaces and decide about equality:

        >>> spc = odl.tensor_space(3, dtype='uint64')
        >>> spc == spc
        True
        >>> spc2 = odl.tensor_space(3, dtype='uint64')
        >>> spc2 == spc
        True
        >>> smaller_spc = odl.tensor_space(2, dtype='uint64')
        >>> spc == smaller_spc
        False
        >>> other_dtype_spc = odl.tensor_space(3, dtype='uint32')
        >>> spc == other_dtype_spc
        False
        >>> other_shape_spc = odl.tensor_space((3, 1), dtype='uint64')
        >>> spc == other_shape_spc
        False
        """
        if other is self:
            return True

        return (type(other) is type(self) and
                self.shape == other.shape and
                self.dtype == other.dtype and
                self.impl == other.impl and
                self.weighting == other.weighting and 
                self.device == other.device
                )

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.shape, self.dtype, self.device, self.impl, self.weighting))

    def __len__(self):
        """Number of tensor entries along the first axis."""
        return int(self.shape[0])
    
    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim == 1:
            posargs = [self.size]
        else:
            posargs = [self.shape]
        posargs += [self.device, self.impl, self.dtype_identifier]
        if self.is_real:
            ctor_name = 'rn'
        elif self.is_complex:
            ctor_name = 'cn'
        else:
            ctor_name = 'tensor_space'

        if (ctor_name == 'tensor_space' or
                not self.dtype_identifier in SCALAR_DTYPES or
                self.dtype != self.default_dtype(self.field)):
            optargs = [('dtype', self.dtype_identifier, '')]
            if self.dtype_identifier in (AVAILABLE_DTYPES):
                optmod = '!s'
            else:
                optmod = ''
        else:
            optargs = []
            optmod = ''

        inner_str = signature_string(posargs, optargs, mod=['', optmod])
        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str

        return '{}({})'.format(ctor_name, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)
        
    ########## _underscore methods ##########
    def _astype(self, dtype:str):
        """Internal helper for `astype`.

        Subclasses with differing init parameters should overload this
        method.
        """
        kwargs = {}
        if dtype in FLOAT_DTYPES + COMPLEX_DTYPES:
            # Use weighting only for floating-point types, otherwise, e.g.,
            # `space.astype(bool)` would fail
            weighting = getattr(self, "weighting", None)
            if weighting is not None:
                kwargs["weighting"] = weighting

        return type(self)(self.shape, dtype=dtype, device=self.device, **kwargs)
    
    def _dist(self, x1, x2):
        """Return the distance between ``x1`` and ``x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Elements whose mutual distance is calculated.

        Returns
        -------
        dist : `float`
            Distance between the elements.

        Examples
        --------
        Different exponents result in difference metrics:

        >>> space_2 = odl.rn(3, exponent=2)
        >>> x = space_2.element([-1, -1, 2])
        >>> y = space_2.one()
        >>> space_2.dist(x, y)
        3.0

        >>> space_1 = odl.rn(3, exponent=1)
        >>> x = space_1.element([-1, -1, 2])
        >>> y = space_1.one()
        >>> space_1.dist(x, y)
        5.0

        Weighting is supported, too:

        >>> space_1_w = odl.rn(3, exponent=1, weighting=[2, 1, 1])
        >>> x = space_1_w.element([-1, -1, 2])
        >>> y = space_1_w.one()
        >>> space_1_w.dist(x, y)
        7.0
        """
        return self.weighting.dist(x1.data, x2.data)
    
    def _divide(self, x1, x2, out):
        """Compute the entry-wise quotient ``x1 / x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Dividend and divisor in the quotient.
        out : `NumpyTensor`
            Element to which the result is written.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> x = space.element([2, 0, 4])
        >>> y = space.element([1, 1, 2])
        >>> space.divide(x, y)
        rn(3).element([ 2.,  0.,  2.])
        >>> out = space.element()
        >>> result = space.divide(x, y, out=out)
        >>> result
        rn(3).element([ 2.,  0.,  2.])
        >>> result is out
        True
        """
        return odl.divide(x1, x2, out)
    
    def _inner(self, x1, x2):
        """Return the inner product of ``x1`` and ``x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Elements whose inner product is calculated.

        Returns
        -------
        inner : `field` `element`
            Inner product of the elements.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> x = space.element([1, 0, 3])
        >>> y = space.one()
        >>> space.inner(x, y)
        4.0

        Weighting is supported, too:

        >>> space_w = odl.rn(3, weighting=[2, 1, 1])
        >>> x = space_w.element([1, 0, 3])
        >>> y = space_w.one()
        >>> space_w.inner(x, y)
        5.0
        """
        return self.weighting.inner(x1.data, x2.data)
    
    def _lincomb(self, a, x1, b, x2, out):
        """Implement the linear combination of ``x1`` and ``x2``.

        Compute ``out = a*x1 + b*x2`` using optimized
        BLAS routines if possible.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        a, b : `TensorSpace.field` element
            Scalars to multiply ``x1`` and ``x2`` with.
        x1, x2 : `NumpyTensor`
            Summands in the linear combination.
        out : `NumpyTensor`
            Tensor to which the result is written.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> x = space.element([0, 1, 1])
        >>> y = space.element([0, 0, 1])
        >>> out = space.element()
        >>> result = space.lincomb(1, x, 2, y, out)
        >>> result
        rn(3).element([ 0.,  1.,  3.])
        >>> result is out
        True
        """
        return odl.add(a*x1, b*x2, out)

    def _multiply(self, x1, x2, out):
        """Compute the entry-wise product ``out = x1 * x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Factors in the product.
        out : `NumpyTensor`
            Element to which the result is written.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> x = space.element([1, 0, 3])
        >>> y = space.element([-1, 1, -1])
        >>> space.multiply(x, y)
        rn(3).element([-1.,  0., -3.])
        >>> out = space.element()
        >>> result = space.multiply(x, y, out=out)
        >>> result
        rn(3).element([-1.,  0., -3.])
        >>> result is out
        True
        """
        return odl.multiply(x1, x2, out)

    def _norm(self, x):
        """Return the norm of ``x``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x : `NumpyTensor`
            Element whose norm is calculated.

        Returns
        -------
        norm : `float`
            Norm of the element.

        Examples
        --------
        Different exponents result in difference norms:

        >>> space_2 = odl.rn(3, exponent=2)
        >>> x = space_2.element([3, 0, 4])
        >>> space_2.norm(x)
        5.0
        >>> space_1 = odl.rn(3, exponent=1)
        >>> x = space_1.element([3, 0, 4])
        >>> space_1.norm(x)
        7.0

        Weighting is supported, too:

        >>> space_1_w = odl.rn(3, exponent=1, weighting=[2, 1, 1])
        >>> x = space_1_w.element([3, 0, 4])
        >>> space_1_w.norm(x)
        10.0
        """
        return self.weighting.norm(x.data)
    
    def _binary_num_operation(self, x1, x2, combinator:str, out=None):
        """
        Internal helper function to implement the __magic_functions__ (such as __add__).

        Parameters
        ----------
        x1 : TensorSpaceElement, int, float, complex
            Left operand
        x2 : TensorSpaceElement, int, float, complex
            Right operand
        combinator: str
            Attribute of the array namespace
        out : TensorSpaceElement, Optional
            LinearSpaceElement for out-of-place operations

        Returns
        -------
        TensorSpaceElement
            The result of the operation `combinator` wrapped in a space with the right datatype.

        Notes:
            The dtype of the returned TensorSpaceElement (and the space that wraps it) is infered 
            from the dtype of the array returned by the backend in which the TensorSpaceElement is 
            implemented. \n
            In order to minimise the expensive operations performed under the hood, i.e clearly 
            unspecified by the user, cross-backend AND cross-devices operations are NOT allowed. \n
            -> 1j + TensorSpaceElement(dtype='float32') IS supported \n
            -> TensorSpaceElement(device=device1) + TensorSpaceElement(device=device2) IS NOT supported \n
            -> TensorSpaceElement(impl=impl1) + TensorSpaceElement(impl=imp2) IS NOT supported \n

        The logic is as follows:
        1) if either of the operands are Python numeric types (int, float complex)
            -> the operation is performed on the backend of the TensorSpaceElement and the dtype infered from it.
        2) if the two operands are TensorSpaceElements
            -> the operation is delegated to the general odl.combinator which performs the checks on space shape and 
            device consistency.

        """
        if self.field is None:
            raise NotImplementedError(f"The space has no field.")

        if isinstance(x1, (int, float, complex)) or isinstance(x2, (int, float, complex)):
            fn =  getattr(self.array_namespace, combinator)
            if out is None:
                if isinstance(x1, (int, float, complex)):
                    result_data = fn(x1, x2.data)
                elif isinstance(x2, (int, float, complex)):
                    result_data = fn(x1.data, x2)
                    
            else:
                assert out in self, f"out is not an element of the space."
                if isinstance(x1, (int, float, complex)):
                    result_data = fn(x1, x2.data, out.data)
                elif isinstance(x2, (int, float, complex)):
                    result_data = fn(x1.data, x2, out.data)
                    
            return self.astype(self.get_dtype_identifier(array=result_data)).element(result_data) 

        assert isinstance(x1, Tensor), 'Left operand is not an ODL Tensor'
        assert isinstance(x2, Tensor), 'Right operand is not an ODL Tensor'

        if out is None:     
            return getattr(odl, combinator)(x1, x2)
        else:
            return getattr(odl, combinator)(x1, x2, out)
        
    def get_dtype_identifier(self, **kwargs):
        raise NotImplementedError  

class Tensor(LinearSpaceElement):

    """Abstract class for representation of `TensorSpace` elements."""
    def __init__(self, space, data):
        """Initialize a new instance."""
        # Tensor.__init__(self, space)
        LinearSpaceElement.__init__(self, space)
        self.__data = data

    ######### static methods #########

    ######### Attributes #########
    @property
    def array_backend(self) -> ArrayBackend:
        return self.space.array_backend

    @property
    def array_namespace(self) -> ModuleType:
        """Name of the array_namespace of this tensor.

        This relates to the python array api
        """
        return self.space.array_namespace
    
    @property
    def data(self):
        """The `numpy.ndarray` representing the data of ``self``."""
        return self.__data
    
    @property
    def device(self):
        """Device on which the space lives."""
        return self.space.device
    
    @property
    def dtype(self):
        """Data type of each entry."""
        return self.space.dtype
    
    @property
    def dtype_identifier(self):
        """Data type as a string of each entry."""
        return self.space.dtype_identifier
    
    @property
    def imag(self):
        """Imaginary part of ``self``.

        Returns
        -------
        imag : `NumpyTensor`
            Imaginary part this element as an element of a
            `NumpyTensorSpace` with real data type.

        Examples
        --------
        Get the imaginary part:

        >>> space = odl.cn(3)
        >>> x = space.element([1 + 1j, 2, 3 - 3j])
        >>> x.imag
        rn(3).element([ 1.,  0., -3.])

        Set the imaginary part:

        >>> space = odl.cn(3)
        >>> x = space.element([1 + 1j, 2, 3 - 3j])
        >>> zero = odl.rn(3).zero()
        >>> x.imag = zero
        >>> x
        cn(3).element([ 1.+0.j,  2.+0.j,  3.+0.j])

        Other array-like types and broadcasting:

        >>> x.imag = 1.0
        >>> x
        cn(3).element([ 1.+1.j,  2.+1.j,  3.+1.j])
        >>> x.imag = [2, 3, 4]
        >>> x
        cn(3).element([ 1.+2.j,  2.+3.j,  3.+4.j])
        """
        if self.space.is_real:
            return self.space.zero()
        elif self.space.is_complex:
            real_space = self.space.astype(self.space.real_dtype)
            return real_space.element(self.data.imag)
        else:
            raise NotImplementedError('`imag` not defined for non-numeric '
                                      'dtype {}'.format(self.dtype))
        
    @property
    def impl(self):
        """Name of the implementation back-end of this tensor."""
        return self.space.impl

    @property
    def itemsize(self):
        """Size in bytes of one tensor entry."""
        return self.space.itemsize

    @property
    def nbytes(self):
        """Total number of bytes in memory occupied by this tensor."""
        return self.space.nbytes
    
    @property
    def ndim(self):
        """Number of axes (=dimensions) of this tensor."""
        return self.space.ndim
    
    @property
    def odl_tensor(self):
        """Number of axes (=dimensions) of this tensor."""
        return True
    
    @property
    def real(self):
        """Real part of ``self``.

        Returns
        -------
        real : `NumpyTensor`
            Real part of this element as a member of a
            `NumpyTensorSpace` with corresponding real data type.

        Examples
        --------
        Get the real part:

        >>> space = odl.cn(3)
        >>> x = space.element([1 + 1j, 2, 3 - 3j])
        >>> x.real
        rn(3).element([ 1.,  2.,  3.])

        Set the real part:

        >>> space = odl.cn(3)
        >>> x = space.element([1 + 1j, 2, 3 - 3j])
        >>> zero = odl.rn(3).zero()
        >>> x.real = zero
        >>> x
        cn(3).element([ 0.+1.j,  0.+0.j,  0.-3.j])

        Other array-like types and broadcasting:

        >>> x.real = 1.0
        >>> x
        cn(3).element([ 1.+1.j,  1.+0.j,  1.-3.j])
        >>> x.real = [2, 3, 4]
        >>> x
        cn(3).element([ 2.+1.j,  3.+0.j,  4.-3.j])
        """
        if self.space.is_real:
            return self
        elif self.space.is_complex:
            real_space = self.space.astype(self.space.real_dtype)
            return real_space.element(self.data.real)
        else:
            raise NotImplementedError('`real` not defined for non-numeric '
                                      'dtype {}'.format(self.dtype))
    
    @property
    def shape(self):
        """Number of elements per axis."""
        return self.space.shape

    @property
    def size(self):
        """Total number of entries."""
        return self.space.size

    ######### public methods #########
    def asarray(self, out=None):
        """Extract the data of this array as a ``numpy.ndarray``.

        This method is invoked when calling `numpy.asarray` on this
        tensor.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype.

        Returns
        -------
        asarray : `numpy.ndarray`
            Numpy array with the same data type as ``self``. If
            ``out`` was given, the returned object is a reference
            to it.

        Examples
        --------
        >>> space = odl.rn(3, dtype='float32')
        >>> x = space.element([1, 2, 3])
        >>> x.asarray()
        array([ 1.,  2.,  3.], dtype=float32)
        >>> np.asarray(x) is x.asarray()
        True
        >>> out = np.empty(3, dtype='float32')
        >>> result = x.asarray(out=out)
        >>> out
        array([ 1.,  2.,  3.], dtype=float32)
        >>> result is out
        True
        >>> space = odl.rn((2, 3))
        >>> space.one().asarray()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]])
        """
        if out is None:
            return self.data
        else:
            out[:] = self.data
            return out
    
    def astype(self, dtype):
        """Return a copy of this element with new ``dtype``.

        Parameters
        ----------
        dtype :
            Scalar data type of the returned space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string. Data types with non-trivial
            shapes are not allowed.

        Returns
        -------
        newelem : `NumpyTensor`
            Version of this element with given data type.
        """
        return self.space.astype(dtype).element(self.data.astype(dtype))
    
    def conj(self, out=None):
        """Return the complex conjugate of ``self``.

        Parameters
        ----------
        out : `NumpyTensor`, optional
            Element to which the complex conjugate is written.
            Must be an element of ``self.space``.

        Returns
        -------
        out : `NumpyTensor`
            The complex conjugate element. If ``out`` was provided,
            the returned object is a reference to it.

        Examples
        --------
        >>> space = odl.cn(3)
        >>> x = space.element([1 + 1j, 2, 3 - 3j])
        >>> x.conj()
        cn(3).element([ 1.-1.j,  2.-0.j,  3.+3.j])
        >>> out = space.element()
        >>> result = x.conj(out=out)
        >>> result
        cn(3).element([ 1.-1.j,  2.-0.j,  3.+3.j])
        >>> result is out
        True

        In-place conjugation:

        >>> result = x.conj(out=x)
        >>> x
        cn(3).element([ 1.-1.j,  2.-0.j,  3.+3.j])
        >>> result is x
        True
        """
        if self.space.is_real:
            if out is None:
                return self
            else:
                out[:] = self
                return out

        if not is_numeric_dtype(self.space.dtype):
            raise NotImplementedError('`conj` not defined for non-numeric '
                                      'dtype {}'.format(self.dtype))

        if out is None:
            return self.space.element(self.data.conj())
        else:
            if out not in self.space:
                raise LinearSpaceTypeError('`out` {!r} not in space {!r}'
                                           ''.format(out, self.space))
            self.data.conj(out.data)
            return out
    
    @imag.setter
    def imag(self, newimag):
        """Setter for the imaginary part.

        This method is invoked by ``x.imag = other``.

        Parameters
        ----------
        newimag : array-like or scalar
            Values to be assigned to the imaginary part of this element.

        Raises
        ------
        ValueError
            If the space is real, i.e., no imagninary part can be set.
        """
        if self.space.is_real:
            raise ValueError('cannot set imaginary part in real spaces')
        self.imag.data[:] = newimag

    @real.setter
    def real(self, newreal):
        """Setter for the real part.

        This method is invoked by ``x.real = other``.

        Parameters
        ----------
        newreal : array-like or scalar
            Values to be assigned to the real part of this element.
        """
        self.real.data[:] = newreal
    
    def show(self, title=None, method='', indices=None, force_show=False,
             fig=None, **kwargs):
        """Display the function graphically.

        Parameters
        ----------
        title : string, optional
            Set the title of the figure

        method : string, optional
            1d methods:

                ``'plot'`` : graph plot

                ``'scatter'`` : scattered 2d points (2nd axis <-> value)

            2d methods:

                ``'imshow'`` : image plot with coloring according to
                value, including a colorbar.

                ``'scatter'`` : cloud of scattered 3d points
                (3rd axis <-> value)

        indices : index expression, optional
            Display a slice of the array instead of the full array. The
            index expression is most easily created with the `numpy.s_`
            constructor, i.e. supply ``np.s_[:, 1, :]`` to display the
            first slice along the second axis.
            For data with 3 or more dimensions, the 2d slice in the first
            two axes at the "middle" along the remaining axes is shown
            (semantically ``[:, :, shape[2:] // 2]``).
            This option is mutually exclusive to ``coords``.

        force_show : bool, optional
            Whether the plot should be forced to be shown now or deferred until
            later. Note that some backends always displays the plot, regardless
            of this value.

        fig : `matplotlib.figure.Figure`, optional
            The figure to show in. Expected to be of same "style", as
            the figure given by this function. The most common use case
            is that ``fig`` is the return value of an earlier call to
            this function.

        kwargs : {'figsize', 'saveto', 'clim', ...}, optional
            Extra keyword arguments passed on to the display method.
            See the Matplotlib functions for documentation of extra
            options.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure. It is also shown to the user.

        See Also
        --------
        odl.util.graphics.show_discrete_data : Underlying implementation
        """
        from odl.discr import uniform_grid
        from odl.util.graphics import show_discrete_data

        # Default to showing x-y slice "in the middle"
        if indices is None and self.ndim >= 3:
            indices = tuple(
                [slice(None)] * 2 + [n // 2 for n in self.space.shape[2:]]
            )

        if isinstance(indices, (Integral, slice)):
            indices = (indices,)
        elif indices is None or indices == Ellipsis:
            indices = (slice(None),) * self.ndim
        else:
            indices = tuple(indices)

        # Replace None by slice(None)
        indices = tuple(slice(None) if idx is None else idx for idx in indices)

        if Ellipsis in indices:
            # Replace Ellipsis with the correct number of [:] expressions
            pos = indices.index(Ellipsis)
            indices = (indices[:pos] +
                       (np.s_[:], ) * (self.ndim - len(indices) + 1) +
                       indices[pos + 1:])

        if len(indices) < self.ndim:
            raise ValueError('too few axes ({} < {})'.format(len(indices),
                                                             self.ndim))
        if len(indices) > self.ndim:
            raise ValueError('too many axes ({} > {})'.format(len(indices),
                                                              self.ndim))

        # Squeeze grid and values according to the index expression
        full_grid = uniform_grid([0] * self.ndim, np.array(self.shape) - 1,
                                 self.shape)
        grid = full_grid[indices].squeeze()
        values = self.asarray()[indices].squeeze()

        return show_discrete_data(values, grid, title=title, method=method,
                                  force_show=force_show, fig=fig, **kwargs)

    ######### magic methods #########        
    def __bool__(self):
        """Return ``bool(self)``."""
        if self.size > 1:
            raise ValueError('The truth value of an array with more than one '
                             'element is ambiguous. '
                             'Use np.any(a) or np.all(a)')
        else:
            return bool(self.asarray())
        
    def __complex__(self):
        """Return ``complex(self)``."""
        return self.data.astype(complex).item()
    
    def __float__(self):
        """Return ``float(self)``."""
        return self.data.astype(float).item()
    
    def __int__(self):
        """Return ``int(self)``."""
        return self.data.astype(int).item()
    
    def __copy__(self):
        """Return ``copy(self)``.

        This implements the (shallow) copy interface of the ``copy``
        module of the Python standard library.

        See Also
        --------
        copy

        Examples
        --------
        >>> from copy import copy
        >>> space = odl.rn(3)
        >>> x = space.element([1, 2, 3])
        >>> y = copy(x)
        >>> y == x
        True
        >>> y is x
        False
        """
        return self.copy()
        
    def __getitem__(self, indices):
        """Return ``self[indices]``.

        This method should be overridden by subclasses.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be accessed.

        Returns
        -------
        values : `TensorSpace.dtype` or `Tensor`
            The value(s) at the given indices. Note that depending on
            the implementation, the returned object may be a (writable)
            view into the original array.
        """
        raise NotImplementedError('abstract method')
   
    def __len__(self):
        """Return ``len(self)``.

        The length is equal to the number of entries along axis 0.
        """
        return len(self.space)
    
    def __repr__(self):
        """Return ``repr(self)``."""
        maxsize_full_print = 2 * np.get_printoptions()['edgeitems']
        self_str = array_str(self, nprint=maxsize_full_print)
        if self.ndim == 1 and self.size <= maxsize_full_print:
            return '{!r}.element({})'.format(self.space, self_str)
        else:
            return '{!r}.element(\n{}\n)'.format(self.space, indent(self_str))
        
    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``.

        This method should be overridden by subclasses.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be written to.
        values : scalar, `array-like` or `Tensor`
            The value(s) that are to be assigned.

            If ``index`` is an integer, ``value`` must be a scalar.

            If ``index`` is a slice or a sequence of slices, ``value``
            must be broadcastable to the shape of the slice.
        """
        raise NotImplementedError('abstract method')

    def __str__(self):
        """Return ``str(self)``."""
        return array_str(self)
    
    """
    [+] = implemented
    [-] = not implemented yet
    [X] = Will not be implemented
    The Python array API expects the following operators:
    #####################################################
    ################# Arithmetic Operators #################
    [+] +x: array.__pos__()
    [+] -x: array.__neg__()
    [+] x1 +  x2: array.__add__()
    [+] x1 -  x2: array.__sub__()
    [+] x1 *  x2: array.__mul__()
    [+] x1 /  x2: array.__truediv__()
    [+] x1 // x2: array.__floordiv__()
    [+] x1 %  x2: array.__mod__()
    [+] x1 ** x2: array.__pow__()
    ################# Array Operators #################
    [X] x1 @ x2: array.__matmul__() -> In ODL, a matmul should be implemented as composition of operators
    ################# Bitwise Operators #################
    [X] ~x: array.__invert__()
    [X] x1 &  x2: array.__and__()
    [X] x1 |  x2: array.__or__()
    [X] x1 ^  x2: array.__xor__()
    [X] x1 << x2: array.__lshift__()
    [X] x1 >> x2: array.__rshift__()
    ################# Comparison Operators #################
    [X] x1 <  x2: array.__lt__() ONLY DEFINED FOR REAL-VALUED DATA TYPES
    [X] x1 <= x2: array.__le__() ONLY DEFINED FOR REAL-VALUED DATA TYPES
    [X] x1 >  x2: array.__gt__() ONLY DEFINED FOR REAL-VALUED DATA TYPES
    [X] x1 >= x2: array.__ge__() ONLY DEFINED FOR REAL-VALUED DATA TYPES
    [+] x1 == x2: array.__eq__()
    [+] x1 != x2: array.__ne__()
    #####################################################
    ################# In-place Arithmetic Operators #################
    [+] x1 +=  x2: array.__iadd__()
    [+] x1 -=  x2: array.__isub__()
    [+] x1 *=  x2: array.__imul__()
    [+] x1 /=  x2: array.__itruediv__()
    [+] x1 //= x2: array.__ifloordiv__()
    [+] x1 %=  x2: array.__imod__()
    [+] x1 **= x2: array.__ipow__()
    ################# In-place Array Operators #################
    [X] x1 @= x2: array.__imatmul__() -> In ODL, a matmul should be implemented as composition of operators
    ################# In-place Bitwise Operators #################
    [X] x1 &=  x2: array.__iand__()
    [X] x1 |=  x2: array.__ior__()
    [X] x1 ^=  x2: array.__ixor__()
    [X] x1 <<= x2: array.__ilshift__()
    [X] x1 >>= x2: array.__irshift__()
    ################# Reflected Arithmetic Operators #################
    [+] x2 +  x1: array.__radd__()
    [+] x2 -  x1: array.__rsub__()
    [+] x2 *  x1: array.__rmul__()
    [+] x2 /  x1: array.__rtruediv__()
    [+] x2 // x1: array.__rfloordiv__()
    [+] x2 %  x1: array.__rmod__()
    [+] x2 ** x1: array.__rpow__()
    ################# Reflected Array Operators #################
    [X] x2 @ x1: array.__rmatmul__() -> In ODL, a matmul should be implemented as composition of operators
    ################# Reflected Bitwise Operators #################
    [X] x2 &  x1: array.__rand__()
    [X] x2 |  x1: array.__ror__()
    [X] x2 ^  x1: array.__rxor__()
    [X] x2 << x1: array.__rlshift__()
    [X] x2 >> x1: array.__rrshift__()
    """
    ####### Arithmetic Operators #######
    def __pos__(self):
        """Return obj positive (+obj)."""
        return odl.positive(self)
    
    def __neg__(self):
        """Return obj positive (+obj)."""
        return odl.negative(self)
    
    def __add__(self, other):
        """Return ``self + other``."""
        return self.space._binary_num_operation(
            self, other, 'add'
        )
    
    def __sub__(self, other):
        """Return ``self - other``."""
        return self.space._binary_num_operation(
            self, other, 'subtract'
        )
    
    def __mul__(self, other):
        """Return ``self * other``."""
        return self.space._binary_num_operation(
            self, other, 'multiply'
        )
    
    def __truediv__(self, other):
        """Implement ``self / other``."""
        with warnings.catch_warnings(record=True) as w:
            result = self.space._binary_num_operation(
                self, other, 'divide'
            )
            for warning in w:
                if issubclass(warning.category, RuntimeWarning):
                    raise RuntimeError(f"Caught a RuntimeWarning: {warning.message}")
            return result
    
    def __floordiv__(self, other):        
        """Implement ``self // other``."""
        return self.space._binary_num_operation(
            self, other, 'floor_divide'
        )

    def __mod__(self, other):        
        """Implement ``self % other``."""
        return self.space._binary_num_operation(
            self, other, 'remainder'
        )
    
    def __pow__(self, other):
        """Implement ``self ** other``, element wise"""
        return self.space._binary_num_operation(
            self, other, 'pow'
        )
    
    ################# Array Operators #################
    def __matmul__(self, other):    
        """Implement ``self @ other``."""
        raise NotImplementedError

    ################# Bitwise Operators #################
    def __invert__(self):
        """Implement ``self.invert``."""
        raise NotImplementedError
    
    def __and__(self, other):
        """Implement ``self.bitwise_and``."""
        raise NotImplementedError
    
    def __or__(self, other):
        """Implement ``self.bitwise_or``."""
        raise NotImplementedError
    
    def __xor__(self, other):
        """Implement ``self.bitwise_xor``."""
        raise NotImplementedError
    
    def __lshift__(self, other):
        """Implement ``self.bitwise_lshift``."""
        raise NotImplementedError
    
    def __rshift__(self, other):
        """Implement ``self.bitwise_rshift``."""
        raise NotImplementedError
    
    ################# Comparison Operators #################
    def __lt__(self, other):
        """Implement ``self < other``."""
        raise NotImplementedError
    
    def __le__(self, other):
        """Implement ``self <= other``."""
        raise NotImplementedError
    
    def __gt__(self, other):
        """Implement ``self > other``."""
        raise NotImplementedError
    
    def __ge__(self, other):
        """Implement ``self >= other``."""
        raise NotImplementedError
    
    def __eq__(self, other):
        """Implement ``self == other``."""
        if other is self:
            return True
        elif other not in self.space:
            return False
        else:
            return (               
                self.shape == other.shape and               
                self.impl == other.impl and
                self.device == other.device and
                self.array_namespace.equal(self, other).all()
                )
    
    def __ne__(self, other):
        """Return ``self != other``."""
        return not self.__eq__(other)
    
     ################# In-place Arithmetic Operators #################
    def __iadd__(self, other):
        """Implement ``self += other``."""
        return self.space._binary_num_operation(
            self, other, 'add', self
        )
    
    def __isub__(self, other):
        """Implement ``self -= other``."""
        return self.space._binary_num_operation(
            self, other, 'subtract', self
        )
    
    def __imul__(self, other):
        """Return ``self *= other``."""
        return self.space._binary_num_operation(
            self, other, 'multiply', self
        )
    
    def __itruediv__(self, other):
        """Implement ``self /= other``."""
        with warnings.catch_warnings(record=True) as w:
            result = self.space._binary_num_operation(
                self, other, 'divide', self
            )
            for warning in w:
                if issubclass(warning.category, RuntimeWarning):
                    raise RuntimeError(f"Caught a RuntimeWarning: {warning.message}")
            return result
    
    def __ifloordiv__(self, other):
        """Implement ``self //= other``."""
        return self.space._binary_num_operation(
            self, other, 'floor_divide', self
        )
    
    def __imod__(self, other):
        """Implement ``self %= other``."""
        return self.space._binary_num_operation(
            self, other, 'remainder', self
        )
    
    def __ipow__(self, other):
        """Implement ``self *= other``, element wise"""
        return self.space._binary_num_operation(
            self, other, 'pow', self
        )
    
    ################# In-place Array Operators #################
    def __imatmul__(self, other):
        """Implement x1 @= x2 """
        raise NotImplementedError
    
    ################# In-place Bitwise Operators #################    
    def __iand__(self, other):
        """Implement ``self.ibitwise_and``."""
        raise NotImplementedError
    
    def __ior__(self, other):
        """Implement ``self.ibitwise_or``."""
        raise NotImplementedError
    
    def __ixor__(self, other):
        """Implement ``self.ibitwise_xor``."""
        raise NotImplementedError
    
    def __lshift__(self, other):
        """Implement ``self.ibitwise_lshift``."""
        raise NotImplementedError
    
    def __irshift__(self, other):
        """Implement ``self.ibitwise_rshift``."""
        raise NotImplementedError

    ################# Reflected Arithmetic Operators #################
    def __radd__(self, other):
        """Return ``other + self``."""
        return self.space._binary_num_operation(
            other, self, 'add'
        )
    
    def __rsub__(self, other):
        """Return ``other - self``."""
        return self.space._binary_num_operation(
            other, self, 'subtract'
        )
 
    def __rmul__(self, other):
        """Return ``other * self``."""
        return self.space._binary_num_operation(
            other, self, 'multiply'
        )
    
    def __rtruediv__(self, other):
        """Implement ``other / self``."""
        return self.space._binary_num_operation(
             other, self, 'divide'
        )
    
    def __rfloordiv__(self, other):
        """Implement ``other // self``."""
        return self.space._binary_num_operation(
            other, self, 'floor_divide'
        )
    
    def __rmod__(self, other):        
        """Implement ``other % self``."""
        return self.space._binary_num_operation(
            other, self, 'remainder'
        )
    
    def __rpow__(self, other):
        """Implement ``other ** self``, element wise"""
        return self.space._binary_num_operation(
            other, self, 'pow'
        )
    
    ################# Reflected Array Operators #################
    def __rmatmul__(self, other):
        """Implement x1 @= x2 """
        raise NotImplementedError
    
    ################# Reflected Bitwise Operators #################    
    def __rand__(self, other):
        """Implement ``self.ibitwise_and``."""
        raise NotImplementedError
    
    def __ror__(self, other):
        """Implement ``self.ibitwise_or``."""
        raise NotImplementedError
    
    def __rxor__(self, other):
        """Implement ``self.ibitwise_xor``."""
        raise NotImplementedError
    
    def __rshift__(self, other):
        """Implement ``self.ibitwise_lshift``."""
        raise NotImplementedError
    
    def __rrshift__(self, other):
        """Implement ``self.ibitwise_rshift``."""
        raise NotImplementedError
        
    ######### private methods #########
    def _assign(self, other, avoid_deep_copy):
        """Assign the values of ``other``, which is assumed to be in the
        same space, to ``self``."""
        if avoid_deep_copy:
            self.__data = other.__data
        else:
            self.__data[:] = other.__data

if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
