# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Base classes for implementations of tensor spaces."""


from types import ModuleType
from numbers import Integral, Number
import warnings
from contextlib import contextmanager
import numpy as np

import odl
from odl.core.set.sets import ComplexNumbers, RealNumbers
from odl.core.set.space import (
    LinearSpace, LinearSpaceElement, LinearSpaceTypeError,
    SupportedNumOperationParadigms, NumOperationParadigmSupport)
from odl.core.array_API_support import ArrayBackend, lookup_array_backend, check_device
from odl.core.util import (
    array_str, indent, is_complex_dtype,
    is_numeric_dtype, is_real_floating_dtype, safe_int_conv,
    signature_string)
from odl.core.util.dtype_utils import(
    is_real_dtype, is_int_dtype,
    is_available_dtype,
    _universal_dtype_identifier,
    is_floating_dtype,
    complex_dtype,
    TYPE_PROMOTION_COMPLEX_TO_REAL, 
    TYPE_PROMOTION_REAL_TO_COMPLEX)
from .weightings.weighting import Weighting, ConstWeighting, ArrayWeighting
from .pspace import ProductSpaceElement

__all__ = ('TensorSpace',)

def default_dtype(array_backend: ArrayBackend | str, field=None):
    """Return the default data type for a given field.

    Parameters
    ----------
    array_backend : `ArrayBackend` or `str`
        The implementation, defining what dtypes are available.
        If a string is given, it is interpreted as an `impl`
        identifier of an array backend from the global registry.
    field : `Field`, optional
        Set of numbers to be represented by a data type.
        Currently supported : `RealNumbers`, `ComplexNumbers`
        The default ``None`` means `RealNumbers`

    Returns
    -------
    dtype :
        Backend data type specifier.
    """
    if not isinstance(array_backend, ArrayBackend):
        array_backend = lookup_array_backend(array_backend)
    if field is None or field == RealNumbers():
        return array_backend.available_dtypes['float64']
    elif field == ComplexNumbers():
        return array_backend.available_dtypes['complex128']
    else:
        raise ValueError(f"no default data type defined for field {field}")


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
    def _init_device(self, device: str):
        """
        Checks that the backend accepts the device passed as an argument.

        Parameters
        ----------
        device : str
            Device identifier
        """
        self.__device = odl.check_device(self.impl, device)

    def _init_dtype(self, dtype: str | int | float | complex):
        """
        Process the dtype argument. This parses the (str or Number) dtype input argument to a backend.dtype and sets two attributes

        self.dtype_identifier (str)      -> Used for passing dtype information from one backend to another
        self.__dtype (backend.dtype) -> Actual dtype of the TensorSpace implementation

        Note:
        The check below is here just in case a user initialise a space directly from this class, which is not recommended
        """

        available_dtypes = self.array_backend.available_dtypes
        identifier = _universal_dtype_identifier(
            dtype, array_backend_selection=[self.array_backend])

        if identifier in available_dtypes.keys():
            self.__dtype_identifier = identifier
            self.__dtype = available_dtypes[identifier]
            # If that fails, we throw an error: the dtype is not a python scalar dtype, not a string describing the dtype or the
            # backend call to parse the dtype has failed.
        else:
            raise ValueError(
                f"The dtype must be in {available_dtypes.keys()} or must be a dtype of the backend, but {dtype} was provided"
            )

    def _init_shape(self, shape, dtype):
        """helper function to handle shape input sanitisation

        Args:
            shape : nonnegative int or sequence of nonnegative ints
            Number of entries of type ``dtype`` per axis in this space. A
            single integer results in a space with rank 1, i.e., 1 axis.

        """
        try:
            shape, shape_in = tuple(safe_int_conv(s) for s in shape), shape
        except TypeError:
            shape, shape_in = (safe_int_conv(shape),), shape
        if any(s < 0 for s in shape):
            raise ValueError(
                f"`shape` must have only nonnegative entries, got {shape_in}"
            )

        # We choose this order in contrast to Numpy, since we usually want
        # to represent discretizations of vector- or tensor-valued functions,
        # i.e., if dtype.shape == (3,) we expect f[0] to have shape `shape`.
        self.__shape = shape

    def _init_field(self):
        """helper function to handle setting the field of a TensorSpace
        """
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
        weighting = kwargs.pop('weighting', None)
        if weighting is None:
            self.__weighting = odl.core.space_weighting(
                impl=self.impl, device=self.device, **kwargs
            )
        else:
            if issubclass(type(weighting), Weighting):
                if weighting.impl != self.impl:
                    raise ValueError(
                        f"`weighting.impl` and space.impl must be consistent, but got \
                        {weighting.impl} and {self.impl}"
                    )
                if isinstance(weighting, ArrayWeighting) and weighting.device != self.device:
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
            elif (
                hasattr(weighting, "__array__")
                or isinstance(weighting, (int, float, tuple, list))
            ):
                self.__weighting = odl.core.space_weighting(
                    impl=self.impl, device=self.device, weight=weighting, **kwargs
                )
            else:
                raise TypeError(
                    "Wrong type of 'weighting' argument. Only floats,array-like and odl.Weightings are accepted"
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

        class TensorSpacebyaxis:
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
                "`complex_dtype` not defined for non-numeric `dtype`")
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
            raise ValueError("`complex_space` not defined for non-numeric `dtype`")
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
        raise NotImplementedError

    @property
    def examples(self):
        """Return example random vectors."""
        # Always return the same numbers
        rand_state = np.random.get_state()
        np.random.seed(1337)

        if is_numeric_dtype(self.dtype):
            yield (
                "Linearly spaced samples",
                self.element(np.linspace(0, 1, self.size).reshape(self.shape)),
            )
            yield (
                "Normally distributed noise",
                self.element(np.random.standard_normal(self.shape)),
            )

        if self.is_real:
            yield (
                "Uniformly distributed noise",
                self.element(np.random.uniform(size=self.shape)),
            )
        elif self.is_complex:
            yield (
                "Uniformly distributed noise",
                self.element(
                    np.random.uniform(size=self.shape)
                    + np.random.uniform(size=self.shape) * 1j
                ),
            )
        else:
            # TODO: return something that always works, like zeros or ones?
            raise NotImplementedError(
                "no examples available for non-numeric data type"
            )

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
        raise NotImplementedError("abstract method")

    @property
    def itemsize(self):
        """Size in bytes of one entry in an element of this space."""
        return int(self.array_backend.array_constructor([], dtype=self.dtype).itemsize)

    @property
    def is_complex(self):
        """True if this is a space of complex tensors."""
        return is_complex_dtype(self.dtype_identifier)

    @property
    def is_real(self):
        """True if this is a space of real tensors."""
        return is_real_floating_dtype(self.dtype_identifier)

    @property
    def is_weighted(self):
        """Return ``True`` if the space is not weighted by constant 1.0."""
        return not (
            isinstance(self.weighting, ConstWeighting) and self.weighting.const == 1.0
        )

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
            raise ValueError("`real_space` not defined for non-numeric `dtype`")
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
            raise ValueError("`None` is not a valid data type")

        available_dtypes = self.array_backend.available_dtypes
        dtype_identifier = _universal_dtype_identifier(
            dtype, array_backend_selection=[self.array_backend]
        )
        if dtype_identifier in available_dtypes:
            dtype = available_dtypes[dtype_identifier]
        else:
            raise ValueError(
                f"Tried to convert space to {dtype}, but this cannot be interpreted as any of"
                + f" {available_dtypes.keys()}, which are all that are available for backend '{self.impl}'."
            )

        if dtype == self.dtype:
            return self

        if is_real_floating_dtype(dtype_identifier) or is_complex_dtype(dtype_identifier):
            if self.dtype_identifier == 'bool':
                return self._astype(dtype_identifier)
            # Caching for real and complex versions (exact dtype mappings)
            elif dtype == self.real_dtype:
                if self.__real_space is None:
                    self.__real_space = self._astype(dtype_identifier)
                return self.__real_space
            elif dtype == self.complex_dtype:
                if self.__complex_space is None:
                    self.__complex_space = self._astype(dtype_identifier)
                return self.__complex_space
            else:
                return self._astype(dtype_identifier)
        else:
            return self._astype(dtype_identifier)

    def to_device(self, device: str):
        """Return a copy of this space with storage on a different computational device.
        Mathematically this is the same space. It also uses the same backend for
        array operations.

        Parameters
        ----------
        device :
            Where elements of this space store their arrays. The default spaces
            store on `'cpu'`. Which alternatives are possible depends on the
            backend (`impl`) and hardware availability.

        Returns
        -------
        newspace : `TensorSpace`
            Version of this space with selected device."""
        _ = check_device(self.impl, device)
        return self._to_device(device)

    def to_impl(self, impl):
        """Return a copy of this space using a different array-backend.
        Mathematically this is the same space, but the computational performance
        can be very different.

        Parameters
        ----------
        impl :
            Identifier of the target backend. Must correspond to a registered
            `ArrayBackend`. See `odl.core.space.entry_points.tensor_space_impl_names`
            for available options.
            Both `impl` and the implementation of the original space must support
            the same device, most typically `'cpu'`. If you want to use GPU storage,
            use a separate call to `TensorSpace.to_device`.

        Returns
        -------
        newspace : `TensorSpace`
            Version of this space with selected backend."""
        _ = check_device(impl, self.device)
        return self._to_impl(impl)
        
    def element(self, inp=None, device=None, copy=None):

        # Most of the cases further below deal with conversions from various array types.
        # This only makes sense for plain arrays and ODL objects based on a single plain
        # array (i.e. `odl.Tensor` subclasses). For other ODL objects, such as product
        # space element, it would result in confusing errors, so we stop this eventuality
        # right here.
        if isinstance(inp, LinearSpaceElement) and not isinstance(inp, Tensor):
            raise TypeError(
                f"Trying to generated a `Tensor` from an ODL object with more structure, {type(inp)=}")

        def wrapped_array(arr):
            if arr.shape != self.shape:
                raise ValueError(
                    f"shape of `inp` not equal to space shape: {arr.shape} != {self.shape}"
                )
            if (is_real_dtype(self.dtype_identifier) and not 
                is_real_dtype(self.array_backend.get_dtype_identifier(array=arr))):
                raise TypeError(f"A real space cannot have complex elements. Got {arr.dtype}")
            elif (is_int_dtype(self.dtype_identifier) and not 
                is_int_dtype(self.array_backend.get_dtype_identifier(array=arr))):
                raise TypeError(f"An integer space can only have integer elements. Got {arr.dtype}")
            
            return self.element_type(self, arr)


        def dlpack_transfer(arr):
            # We check that the object implements the dlpack protocol:
            # assert hasattr(inp, "__dlpack_device__") and hasattr(
            #     arr, "__dlpack__"
            # ), """The input does not support the DLpack framework.
            #     Please convert it to an object that supports it first.
            # (cf:https://data-apis.org/array-api/latest/purpose_and_scope.html)"""
            # We begin by checking that the transfer is actually needed:
            if arr.device == self.device and arr.dtype == self.dtype:
                return self.array_backend.array_constructor(arr, copy=copy)
            return self.array_backend.from_dlpack(arr, device=self.device, copy=copy)

        # Case 1: no input provided
        if inp is None:
            arr = self.array_namespace.empty(
                    self.shape, dtype=self.dtype, device=self.device
                )    
        # Case 2: input is provided
        # Case 2.1: the input is an ODL OBJECT
        # ---> The data of the input is transferred to the space's device and data type AND wrapped into the space.
        elif isinstance(inp, Tensor):
            if inp.space == self and copy != True:
                # If it is already element of the exact space, nothing needs to be done.
                return inp
            arr = dlpack_transfer(inp.data)
        # Case 2.2: the input is an object that implements the python array aPI (np.ndarray, torch.Tensor...)
        # ---> The input is transferred to the space's device and data type AND wrapped into the space.
        elif hasattr(inp, '__array__'):
            arr = dlpack_transfer(inp)
        # Case 2.3: the input is an array like object [[1,2,3],[4,5,6],...]
        # ---> The input is transferred to the space's device and data type AND wrapped into the space.
        elif isinstance(inp, (list, tuple)):
            arr = self.array_backend.array_constructor(
                inp, dtype=self.dtype, device=self.device
            )
        # Case 2.4: the input is a Python Number
        # ---> The input is broadcasted to the space's shape and transferred to the space's device and data type AND wrapped into the space.
        elif isinstance(inp, (int, float, complex)):
            arr = self.broadcast_to(inp)

        else:
            raise ValueError(
                f"The input {inp} with dtype {type(inp)} is not supported by the `element` method."
                + f" The only supported types are int, float, complex, list, tuples, objects with an"
                + f" __array__ attribute of a supported backend (e.g np.ndarray and torch.Tensor) and ODL Tensors."
            )

        return wrapped_array(arr)

    def finfo(self):
        "Machine limits for floating-point data types."
        return self.array_namespace.finfo(self.dtype)

    def iinfo(self):
        "Machine limits for integer data types."
        return self.array_namespace.iinfo(self.dtype)

    def divide(self, x1, x2, out=None):
        """Compute the entry-wise quotient ``x1 / x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `Tensor`
            Dividend and divisor in the quotient.
        out : `Tensor`
            Element to which the result is written.
        """
        return self._divide(x1, x2, out)

    def multiply(self, x1, x2, out=None):
        """Compute the entry-wise product ``out = x1 * x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `Tensor`
            Factors in the product.
        out : `Tensor`
            Element to which the result is written.
        """
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
        return hash(
            (type(self), self.shape, self.dtype, self.device, self.impl, self.weighting))

    def __len__(self):
        """Number of tensor entries along the first axis."""
        return int(self.shape[0])

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim == 1:
            posargs = [self.size]
        else:
            posargs = [self.shape]

        if self.is_real:
            ctor_name = 'rn'
        elif self.is_complex:
            ctor_name = 'cn'
        else:
            ctor_name = 'tensor_space'

        optmod = ''

        if self.device == 'cpu':
            if self.impl == 'numpy':
                if ( ctor_name == 'tensor_space'
                      or not is_numeric_dtype(self.dtype_identifier)
                      or self.dtype != default_dtype(self.array_backend, self.field) ):
                    posargs += [self.dtype_identifier]
                    if is_available_dtype(self.dtype_identifier):
                        optmod = '!s'
            else:
                posargs += [self.dtype_identifier, self.impl]
        else:
            posargs += [self.dtype_identifier, self.impl, self.device]

        inner_str = signature_string(posargs, optargs=[], mod=['', optmod])
        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str

        return f"{ctor_name}({inner_str})"

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

    ########## _underscore methods ##########
    def _astype(self, dtype: str):
        """Internal helper for `astype`.

        Subclasses with differing init parameters should overload this
        method.
        """
        kwargs = {}
        if is_real_dtype(dtype) or is_complex_dtype(dtype):
            # Use weighting only for floating-point types, otherwise, e.g.,
            # `space.astype(bool)` would fail
            weighting = getattr(self, "weighting", None)
            if weighting is not None:
                kwargs["weighting"] = weighting

        return type(self)(self.shape, dtype=dtype, device=self.device, **kwargs)

    def _to_device(self, device: str):
        """Internal helper for `to_device`.

        Subclasses with differing init parameters should overload this
        method.
        """
        kwargs = {}
        weighting = getattr(self, "weighting", None)
        if weighting is not None:
            kwargs["weighting"] = weighting.to_device(device)

        return type(self)(self.shape, dtype=self.dtype, device=device, **kwargs)

    def _to_impl(self, impl: str):
        """Internal helper for `to_impl`.

        Subclasses with structure other than just backend-specific ℝⁿ spaces should
        overload this method.
        """
        # Lazy import to avoid cyclic dependency
        from odl.core.space.space_utils import tensor_space

        kwargs = {}
        weighting = getattr(self, "weighting", None)
        if weighting is not None:
            kwargs["weighting"] = weighting.to_impl(impl)

        return tensor_space(
            shape=self.shape,
            dtype=self.dtype_identifier,
            impl=impl,
            device=self.device,
            **kwargs,
        )

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
        >>> out
        rn(3).element([ 2.,  0.,  2.])
        >>> out.data is result.data
        True
        >>> out = np.zeros((3))
        >>> result = np.divide([2,0,4], [1,1,2], out=out)
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
        >>> result.data is out.data
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
    
    def _elementwise_num_operation(self, operation:str
                                       , x1: LinearSpaceElement | Number
                                       , x2: None | LinearSpaceElement | Number = None
                                       , out=None
                                       , namespace=None
                                       , **kwargs ):
        """
        Internal helper function to implement the __magic_functions__ (such as __add__).

        Parameters
        ----------
        x1 : LinearSpaceElement, Number
            Left operand
        x2 : LinearSpaceElement, Number
            Right operand
        operation: str
            Attribute of the array namespace
        out : TensorSpaceElement, Optional
            LinearSpaceElement for out-of-place operations

        Returns
        -------
        TensorSpaceElement
            The result of the operation `operation` wrapped in a space with the right datatype.

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
            -> the operation is delegated to the general odl.operation which performs the checks on space shape and
            device consistency.

        """
        if namespace is None:
            arr_operation = self.array_backend.lookup_array_operation(operation)
            fn = arr_operation.operation_call
            if arr_operation.supports_out_argument:
                fn_in_place = arr_operation.operation_call
            else:
                # If there is no native `out` argument of the low-level call, an
                # in-place update needs to be emulated in the relevant branches.
                fn_in_place = None
        else:
            fn = getattr(namespace, operation)
            # If an explicit namespace was provided, we have to assume it contains
            # the function in whichever form appropriate for performing the call
            # as requested.
            fn_in_place = fn

        if out is not None:
            assert isinstance(
                out, Tensor
            ), f"The out argument must be an ODL Tensor, got {type(out)}."
            assert (
                self.shape == out.space.shape
            ), f"The shapes of {self} and out {out.space.shape} differ, cannot perform {operation}"
            assert (
                self.device == out.space.device
            ), f"The devices of {self} and out {out.space.device} differ, cannot perform {operation}"

        if x1 is None:
            raise TypeError("The left-hand argument always needs to be provided")

        if x2 is None:
            assert x1 in self,"The left operand is not an element of the space."
            if out is None:
                result_data = fn(x1.data, **kwargs)
            elif fn_in_place is None:
                result_data = fn(x1.data, **kwargs)
                out[:] = result_data
            else:
                result_data = fn_in_place(x1.data, out=out.data, **kwargs)
            return self.astype(
                self.array_backend.get_dtype_identifier(array=result_data)
            ).element(result_data)

        from odl.core.operator import Operator
        if not isinstance(x1, (int, float, complex, Tensor, ProductSpaceElement, Operator)):
            raise TypeError(f"The type of the left operand {type(x1)} is not supported.")
        
        if not isinstance(x2, (int, float, complex, Tensor, ProductSpaceElement, Operator)):
            raise TypeError(f"The type of the right operand {type(x2)} is not supported.")
        
        def _dtype_helper_python_number(x: Tensor, y:int|float|complex):
            # We return the backend-specific dtype
            if isinstance(y, int):
                # Here, we are sure that upcasting y to float will not be a problem
                return x.dtype
            elif isinstance(y, float):
                if is_int_dtype(x.dtype):
                    return type(y)
                elif is_floating_dtype(x.dtype):
                    return x.dtype
                else:
                    raise ValueError(f"The dtype of x {type(x)} is not supported.")
            elif isinstance(y, complex):
                if is_int_dtype(x.dtype) or is_real_dtype(x.dtype):
                    return complex_dtype(x.dtype, backend=x.array_backend)
                elif is_complex_dtype(x.dtype):
                    return x.dtype
                else:
                    raise ValueError(f"The dtype of x {type(x)} is not supported.")
            else:
                raise ValueError(f"The dtype of y {type(y)} is not supported.")
            
        if isinstance(x1, (int, float, complex)) or isinstance(x2, (int, float, complex)):
            if out is None:
                if isinstance(x1, (int, float, complex)):
                    dtype = _dtype_helper_python_number(x2, x1)
                    x1 = self.array_backend.array_constructor(x1, dtype=dtype)
                    result_data = fn(x1, x2.data, **kwargs)

                elif isinstance(x2, (int, float, complex)):
                    dtype = _dtype_helper_python_number(x1, x2)
                    x2 = self.array_backend.array_constructor(x2, dtype=dtype)
                    result_data = fn(x1.data, x2, **kwargs)

            else:
                if isinstance(x1, (int, float, complex)):
                    dtype = _dtype_helper_python_number(x2, x1)
                    x1 = self.array_backend.array_constructor(x1, dtype=dtype)
                    if fn_in_place is None:
                        result_data = fn(x1, x2.data, **kwargs)
                        out[:] = result_data
                    else:
                        result_data = fn_in_place(x1, x2.data, out=out.data, **kwargs)

                elif isinstance(x2, (int, float, complex)):
                    dtype = _dtype_helper_python_number(x1, x2)
                    x2 = self.array_backend.array_constructor(x2, dtype=dtype)
                    if fn_in_place is None:
                        result_data = fn(x1.data, x2, **kwargs)
                        out[:] = result_data
                    else:
                        result_data = fn_in_place(x1.data, x2, out=out.data, **kwargs)

            return self.astype(
                self.array_backend.get_dtype_identifier(array=result_data)
            ).element(result_data)

        if isinstance(x1, ProductSpaceElement):
            if not isinstance(x2, Tensor):
                raise TypeError(f"The right operand is not an ODL Tensor. {type(x2)=}")
            return x1.space._elementwise_num_operation(operation, x1, x2, out, namespace=namespace, **kwargs)

        elif isinstance(x2, ProductSpaceElement):
            if not isinstance(x1, Tensor):
                raise TypeError(f"The left operand is not an ODL Tensor. {type(x1)=}")
            return x2.space._elementwise_num_operation(operation, x1, x2, out, namespace=namespace, **kwargs)
        
        if isinstance(x2, Operator):
            if operation=='multiply':
                warnings.warn("The composition of a LinearSpaceElement and an"
                             +" Operator using the * operator is deprecated"
                             +" and will be removed in future ODL versions."
                             +" Please replace * with @.")
                return x2.__rmul__(x1)
            elif operation =='add':
                return x2.__radd__(x1)
            elif operation =='subtract':
                return x2.__rsub__(x1)
            else:
                raise TypeError(f"Attempted numerical operation {operation}"
                              + " between two incompatible objects"
                              + f" ({type(x1)=}, {type(x2)=})")

        if isinstance(x1, Tensor) and isinstance(x2, Tensor):
            assert (
                self.array_backend.array_type == x2.array_backend.array_type
            ), f"The types of {self.array_backend.array_type} and x2 {x2.array_backend.array_type} differ, cannot perform {operation}"
            assert (
                self.shape == x2.space.shape
            ), f"The shapes of {self} and x2 {x2.space.shape} differ, cannot perform {operation}"
            assert (
                self.device == x2.space.device
            ), f"The devices of {self} and x2 {x2.space.device} differ, cannot perform {operation}"

            if out is None:
                result = fn(x1.data, x2.data)
            elif fn_in_place is None:
                result = fn(x1.data, x2.data)
                out.data[:] = result
            else:
                result = fn(x1.data, x2.data, out=out.data)
    
            # We make sure to return an element of the right type: 
            # for instance, if two spaces have a int dtype, the result of the division 
            # of one of their element by another return should be of float dtype
            return x1.space.astype(x1.space.array_backend.get_dtype_identifier(array=result)).element(result) 
        else:
            raise TypeError(f"Neither x1 nor x2 are odl ODL Tensors. Got {type(x1)} and {type(x2)}")

        

    def _element_reduction(self, operation:str
                               , x: "Tensor"
                               , **kwargs
                               ):
        fn = getattr(self.array_namespace, operation)
        result = fn(x.data, **kwargs)
        try:
            return result.item()
        except AttributeError:
            assert result.shape == ()
            return result[0]
        except (ValueError, RuntimeError):
            # Arises when we are performing the 'reductions' along certains axis only. We can't take the item of an array with several dimensions. 
            # TODO: We should handle that differently than with try and excepts.
            return result
        
        

class Tensor(LinearSpaceElement):

    """Abstract class for representation of `TensorSpace` elements."""
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
        """The backend-specific array representing the data of ``self``."""
        raise NotImplementedError("abstract method")

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
            return real_space.element(self.data.imag, copy=False)
        raise NotImplementedError(
                f"`imag` not defined for non-numeric dtype {self.dtype}"
            )

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
            return real_space.element(self.data.real, copy=False)

        raise NotImplementedError(
                f"`real` not defined for non-numeric dtype {self.dtype}"
            )

    @property
    def shape(self):
        """Number of elements per axis."""
        return self.space.shape

    @property
    def size(self):
        """Total number of entries."""
        return self.space.size

    ######### public methods #########
    def asarray(self, out=None, must_be_contiguous: bool =False):
        """Extract the data of this array as a backend-specific array/tensor.

        This method is invoked when calling `numpy.asarray` on this
        tensor.

        Parameters
        ----------
        out : array_like, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct backend, dtype and device.
        must_be_contiguous: `bool`
            If this is `True`, then the returned array must occupy
            a single block of memory and the axes be ordered
            (in C order). Cf. `numpy.ascontiguousarray`.
            This may require making a copy.
            If `False` is given, the returned array may be a view
            or have transposed axes, if this allows avoiding a copy.
            If an `out` argument is provided, `must_be_contiguous`
            is irrelevant.

        Returns
        -------
        asarray : array_like
            Numpy array, pytorch tensor or similar with the same data type as ``self``.
            If ``out`` was given, the returned object is a reference to it.

        Examples
        --------
        >>> space = odl.rn(3, dtype='float32')
        >>> x = space.element([1, 2, 3])
        >>> x.asarray()
        array([ 1.,  2.,  3.], dtype=float32)
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
            if must_be_contiguous:
                return self.array_backend.make_contiguous(self.data)
            else:
                return self.data
        else:
            out[:] = self.data
            return out

    @contextmanager
    def writable_array(self, must_be_contiguous: bool =False):
        """Context manager that casts `self` to a backend-specific array and saves changes
        made to that array back in `self`.

        Parameters
        ----------
        must_be_contiguous : bool
            Whether the writable array should guarantee standard C order.
            See documentation to `asarray` for the semantics.

        Examples
        --------

        >>> space = odl.uniform_discr(0, 1, 3)
        >>> x = space.element([1, 2, 3])
        >>> with x.writable_array() as arr:
        ...     arr += [1, 1, 1]
        >>> x
        uniform_discr(0.0, 1.0, 3).element([ 2.,  3.,  4.])

        Note that the changes are in general only saved upon exiting the
        context manager. Before, the input object may remain unchanged.
        """
        arr = None
        try:
            # TODO(Justus) it should be possible to avoid making a copy here,
            # and actually just modify `data` in place.
            arr = self.asarray(must_be_contiguous=must_be_contiguous)
            yield arr
        finally:
            if arr is not None:
                self.data[:] = arr

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
        newelem : `Tensor`
            Version of this element with given data type.
        """
        return self.space.astype(dtype).element(self.data.astype(dtype))

    def to_device(self, device: str):
        """Return a copy of this element with the same values stored on
        a different computational device.

        Parameters
        ----------
        device :
            Identifier of the desired storage location. Which ones are
            supported depends on the array backend (`impl`). Always
            allowed is `'cpu'`, but GPU alternatives like `'cuda:0'`
            can offer better performance if available.

        Returns
        -------
        newelem : `Tensor`
            Version of this element with its data array on the desired device.
        """
        return self.space.to_device(device).element(
                  self.array_backend.to_device(self.data, device))
    
    def to_impl(self, impl: str):
        """Return a copy of this element with the same values stored using
        a different array backend.

        Parameters
        ----------
        impl :
            Identifier of the target backend. Must correspond to a registered
            `ArrayBackend`. See `odl.core.space.entry_points.tensor_space_impl_names`
            for available options.
            Both `impl` and the implementation of the original space must support
            the same device, most typically `'cpu'`. If you want to use GPU storage,
            use a separate call to `Tensor.to_device`.

        Returns
        -------
        newelem : `Tensor`
            Version of this element with its data array using the desired backend.
        """
        new_backend = lookup_array_backend(impl)
        new_data = new_backend.array_namespace.from_dlpack(self.data)

        # TODO (Justus) this is a workaround for inconsistent behaviour by
        # DLPack / the array backends. DLPack tries to avoid a copy and makes
        # the result readonly, which is not fully supported and causes various problems.
        # Making an explicit copy avoids this, but is not ideal from a performance
        # perspective. It might make sense to add a `copy` argument that controls
        # this, and/or exception handling.
        # Perhaps in the future it will also just work by leaving it up to DLPack.
        new_data = new_backend.array_constructor(new_data, copy=True)

        assert (
            str(new_data.device) == self.device
        ), (f"Error when transferring array from {self.impl} to {impl}:"
          + f" device changed from {self.device} to {new_data.device}."
          + f" Ensure to use a device supported by both backends.")
        assert (
            _universal_dtype_identifier(new_data.dtype) == self.dtype_identifier
        ), (f"Error when transferring array from {self.impl} to {impl}:"
          + f" dtype changed from {self.dtype} to {new_data.dtype}."
          + f" Ensure to use a dtype supported by both backends.")
        return self.space.to_impl(impl).element(new_data)

    def set_zero(self):
        """Set this element to zero.

        See Also
        --------
        LinearSpace.zero
        """
        self.data[:] = 0
        return self

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
            raise NotImplementedError(
                f"`conj` not defined for non-numeric dtype {self.dtype}"
            )

        if out is None:
            return self.space.element(self.data.conj())
        else:
            if out not in self.space:
                raise LinearSpaceTypeError(
                    f"`out` {out} not in space {self.space}")
            # self.data.conj(out.data)
            out.data = self.array_namespace.conj(self.data)
            return out
    
    @imag.setter
    def imag(self, newimag):
        """Setter for the imaginary part.

        This method is invoked by ``x.imag = other``.

        Parameters
        ----------
        newimag : `Tensor`, array-like, or scalar
            Values to be assigned to the imaginary part of this element.

        Raises
        ------
        ValueError
            If the space is real, i.e., no imagninary part can be set.
        """
        if self.space.is_real:
            raise ValueError("cannot set imaginary part in real spaces")
        if isinstance(newimag, Tensor):
            assert(newimag in self.space.real_space)
        else:
            newimag = self.space.real_space.element(newimag)
        self.data.imag = newimag.data

    @real.setter
    def real(self, newreal):
        """Setter for the real part.

        This method is invoked by ``x.real = other``.

        Parameters
        ----------
        newreal : `Tensor`, array-like, or scalar
            Values to be assigned to the real part of this element.
        """
        if isinstance(newreal, Tensor):
            assert(newreal in self.space.real_space)
        else:
            newreal = self.space.real_space.element(newreal)
        self.data.real = newreal.data
    
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
        odl.core.util.graphics.show_discrete_data : Underlying implementation
        """
        from odl.core.discr import uniform_grid
        from odl.core.util.graphics import show_discrete_data

        # Default to showing x-y slice "in the middle"
        if indices is None and self.ndim >= 3:
            indices = tuple([slice(None)] * 2 + [n // 2 for n in self.space.shape[2:]])

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
            raise ValueError(f"too few axes ({len(indices)} < {self.ndim})")
        if len(indices) > self.ndim:
            raise ValueError(f"too many axes ({len(indices)} > {self.ndim})")

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
            raise ValueError(
                "The truth value of an array with more than one "
                "element is ambiguous. "
                "Use np.any(a) or np.all(a)")
        else:
            return bool(self.asarray())
        
    def __complex__(self):
        """Return ``complex(self)``."""
        assert len(self.data) == 1
        return complex(self.data.item())

    def __float__(self):
        """Return ``float(self)``."""
        assert len(self.data) == 1
        return float(self.data.item())

    def __int__(self):
        """Return ``int(self)``."""
        assert len(self.data) == 1
        return int(self.data.item())

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
        raise NotImplementedError("abstract method")

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
            return f"{self.space}.element({self_str})"
            return f"{self.space}.element(\n{indent(self_str)}\n)"
        
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
        raise NotImplementedError("abstract method")

    def __str__(self):
        """Return ``str(self)``."""
        return array_str(self)


    ####### Arithmetic Operators #######
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
    def __eq__(self, other):
        """Implement ``self == other``."""
        bool_space = self.space.astype(bool)
        if other is self:
            return True
        elif other not in self.space:
            return False
        else:
            return bool(self.array_namespace.all(self.data == other.data))
    
    def __ne__(self, other):
        """Return ``self != other``."""
        return not self.__eq__(other)

    ################# In-place Array Operators #################

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

    ################# Reflected Array Operators #################

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
        raise NotImplementedError("abstract method")

if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests

    run_doctests()
