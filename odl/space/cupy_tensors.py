# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Implementation of tensor spaces using CuPy.

See https://cupy.chainer.org/ or https://github.com/cupy/cupy for details
on the backend.
"""

from __future__ import print_function, division, absolute_import
import numpy as np
import warnings

from odl.set import RealNumbers, ComplexNumbers
from odl.space.base_tensors import TensorSpace, Tensor
from odl.space.weighting import (
    Weighting, ArrayWeighting, ConstWeighting,
    CustomInner, CustomNorm, CustomDist)
from odl.util import (
    array_str, dtype_str, is_floating_dtype, real_dtype,
    signature_string, indent)

try:
    import cupy
except ImportError:
    CUPY_AVAILABLE = False
else:
    _maj = int(cupy.__version__.split('.')[0])
    if _maj < 2:
        raise warnings.warn(
            'your version {} of CuPy is not supported; please upgrade to '
            'version 2.0.0 or higher'.format(cupy.__version__), RuntimeWarning)
    CUPY_AVAILABLE = True


__all__ = ('CupyTensorSpace',)


# --- Space method implementations --- #


if CUPY_AVAILABLE:
    lico = cupy.ElementwiseKernel(in_params='T a, T x, T b, T y',
                                  out_params='T z',
                                  operation='z = a * x + b * y;',
                                  name='lico')


def _fallback_scal(a, x):
    """Fallback implementation of ``scal`` when cuBLAS is not applicable."""
    x *= a
    return x


def _fallback_axpy(a, x, y):
    """Fallback implementation of ``axpy`` when cuBLAS is not applicable."""
    return lico(a, x, 1, y, y)


def _get_flat_inc(arr1, arr2=None):
    """Return flat index increment(s) for cuBLAS, raise if not applicable.

    This function checks if the array stride(s) allow usage of cuBLAS
    and returns the ``incx`` (and ``incy``) parameters needed for the
    cuBLAS functions. If the array strides ar such that cuBLAS cannot
    be applied, a ``ValueError`` is raised, triggering a fallback
    implementation.

    For **1 array**, the conditions to be fulfilled are

    - the strides do not contain 0 and
    - the memory of the array has constant stride.

    The second point applies to

    - contiguous arrays,
    - chunks of arrays along the **slowest-varying axis** (``arr[1:5, ...]``
      for C-contiguous ``arr``), and
    - strided slices along the **fastest-varying axis** (``arr[..., ::2]``
      for C-contiguous ``arr``).

    For **2 arrays**, both arrays must

    - fulfill the "1 array" conditions individually,
    - have the same total size, and
    - the axis order of both must be the same in the sense that the same
      index array sorts the strides of both arrays in ascending order.

    Parameters
    ----------
    arr1 : cupy.core.core.ndarray
        Array to check for compatibility.
    arr2 : cupy.core.core.ndarray, optional
        Second array to check for compatibility, by itself and with ``arr1``.

    Returns
    -------
    flat_inc1 : int
        Memory stride (in terms of elements, not bytes) of ``arr1``.
    flat_inc2 : int or None
        Memory stride of ``arr2`` if provided, otherwise ``None``.

    Raises
    ------
    ValueError
        If the conditions for cuBLAS compatibility are not met.

    Examples
    --------
    >>> arr_c = cupy.zeros((4, 4, 4))
    >>> arr_c.strides
    (128, 32, 8)
    >>> arr_f = cupy.asfortranarray(arr_c)
    >>> arr_f.strides
    (8, 32, 128)

    Contiguous arrays are compatible by and with themselves, but not with
    arrays that are contiguous in a different ordering:

    >>> _get_flat_inc(arr_c)
    1
    >>> _get_flat_inc(arr_c, arr_c)
    (1, 1)
    True
    >>> _get_flat_inc(arr_f)
    1
    >>> _get_flat_inc(arr_f, arr_f)
    (1, 1)
    >>> _get_flat_inc(arr_c, arr_f)
    Traceback (most recent call last):
        ...
    ValueError

    Slicing in the **fastest** axis is allowed as it results in a constant
    stride in the flat memory. Slicing with stride in any other axis is
    results in incompatibility.

    C ordering (last axis fastest):

    >>> half_arr_0_c = cupy.zeros((2, 4, 4))
    >>> half_arr_1_c = cupy.zeros((4, 2, 4))
    >>> half_arr_2_c = cupy.zeros((4, 4, 2))
    >>> _get_flat_inc(arr_c[::2, :, :], half_arr_0_c)
    Traceback (most recent call last):
        ...
    ValueError
    >>> _get_flat_inc(arr_c[:, ::2, :], half_arr_1_c)
    Traceback (most recent call last):
        ...
    ValueError
    >>> _get_flat_inc(arr_c[:, :, ::2], half_arr_2_c)
    (2, 1)

    Fortran ordering (first axis fastest):

    >>> half_arr_0_f = cupy.asfortranarray(half_arr_0_c)
    >>> half_arr_1_f = cupy.asfortranarray(half_arr_1_c)
    >>> half_arr_2_f = cupy.asfortranarray(half_arr_2_c)
    >>> _get_flat_inc(arr_f[::2, :, :], half_arr_0_f)
    (2, 1)
    >>> _get_flat_inc(arr_f[:, ::2, :], half_arr_1_f)
    Traceback (most recent call last):
        ...
    ValueError
    >>> _get_flat_inc(arr_f[:, :, ::2], half_arr_2_f)
    Traceback (most recent call last):
        ...
    ValueError

    Axes swapped (middle axis fastest):

    >>> arr_s = cupy.swapaxes(arr_c, 1, 2)
    >>> arr_s.strides
    (128, 8, 32)
    >>> half_arr_0_s = cupy.swapaxes(half_arr_0_c, 1, 2)
    >>> half_arr_1_s = cupy.swapaxes(half_arr_1_c, 1, 2)
    >>> half_arr_2_s = cupy.swapaxes(half_arr_2_c, 1, 2)
    >>> _get_flat_inc(arr_s[:, :, ::2], half_arr_0_s)
    Traceback (most recent call last):
        ...
    ValueError
    >>> _get_flat_inc(arr_s[:, ::2, :], half_arr_1_s)
    (2, 1)
    >>> _get_flat_inc(arr_s[:, :, ::2], half_arr_2_s)
    Traceback (most recent call last):
        ...
    ValueError
    """
    # Zero strides not allowed
    if 0 in arr1.strides:
        raise ValueError
    if arr2 is not None and 0 in arr2.strides:
        raise ValueError

    # Candidate for flat_inc of array 1
    arr1_flat_inc = min(arr1.strides) // arr1.itemsize

    # Check if the strides are as in a contiguous array (after reordering
    # the axes), except for the fastest axis. We allow arbitrary axis order
    # for operations on the whole array at once, as long as it can be
    # indexed with a single flat index and stride.
    arr1_ax_order = np.argsort(arr1.strides)  # ascending
    arr1_sorted_shape = np.take(arr1.shape, arr1_ax_order)
    arr1_sorted_shape[0] *= arr1_flat_inc
    arr1_elem_strides = np.take(arr1.strides, arr1_ax_order) // arr1.itemsize
    if np.any(np.cumprod(arr1_sorted_shape[:-1]) != arr1_elem_strides[1:]):
        raise ValueError

    if arr2 is None:
        return arr1_flat_inc
    else:
        arr2_flat_inc = _get_flat_inc(arr2)
        if arr1.size != arr2.size:
            raise ValueError
        if np.any(np.diff(np.take(arr2.strides, arr1_ax_order)) < 0):
            # Strides of arr2 are not sorted by axis order of arr1
            raise ValueError
        return arr1_flat_inc, arr2_flat_inc


def _cublas_func(name, dtype):
    """Return the specified cupy.cuda.cublas function for a given dtype.

    Parameters
    ----------
    name : str
        Raw function name without prefix, e.g., ``'axpy'``.
    dtype :
        Numpy dtype specifier for which the cuBLAS function should be
        used. Must be single or double precision float or complex.

    Raises
    ------
    ValueError
        If the data type is not supported by cuBLAS.
    """
    dtype, dtype_in = np.dtype(dtype), dtype
    if dtype == 'float32':
        prefix = 's'
    elif dtype == 'float64':
        prefix = 'd'
    elif dtype == 'complex64':
        prefix = 'c'
    elif dtype == 'complex128':
        prefix = 'z'
    else:
        raise ValueError('dtype {!r} not supported by cuBLAS'.format(dtype_in))

    return getattr(cupy.cuda.cublas, prefix + name)


def _get_scal_axpy(x1, x2):
    """Return implementations of scal and axpy suitable for the inputs.

    If the inputs are suitable, a cuBLAS implementation is returned, otherwise
    a fallback implementation. To be suitable for cuBLAS, both arrays
    must

    - have single or double precision float or complex data type and
    - have a single integer stride when flattened, i.e., be contiguous
      or (this allows using)
    """
    try:
        incx1 = _flat_inc(x1.data)
        incx2 = _flat_inc(x2.data)
    except ValueError:
        use_cublas = False
    else:
        use_cublas = True

    if use_cublas:
        try:
            scal_cublas = _cublas_func('scal', x1.dtype)
        except (ValueError, AttributeError):
            scal = fallback_scal
        else:
            def scal(a, x):
                with cupy.cuda.Device(x1.device) as dev:
                    return scal_cublas(
                        dev.cublas_handle, x.data.size, a, x.data.ptr, incx1)

        try:
            axpy_cublas = _cublas_func('axpy', x1.dtype)
        except (ValueError, AttributeError):
            axpy = fallback_axpy
        else:
            def axpy(a, x, y):
                with cupy.cuda.Device(x1.device) as dev:
                    return axpy_cublas(
                        dev.cublas_handle, x.data.size, a,
                        x.data.ptr, incx1, y.data.ptr, incx2)
    else:
        scal = fallback_scal
        axpy = fallback_axpy

    return scal, axpy


def _lincomb_impl(a, x1, b, x2, out):
    """Linear combination implementation, assuming types have been checked.

    This implementation is a highly optimized, considering all special
    cases of array alignment and special scalar values 0 and 1 separately.
    """
    scal, axpy = _get_scal_axpy(x1, x2)

    if a == 0 and b == 0:
        # out <- 0
        out.data.fill(0)

    elif a == 0:
        # Compute out <- b * x2
        if out is x2:
            # out <- b * out
            if b == 1:
                pass
            else:
                scal(b, out.data)
        else:
            # out <- b * x2
            if b == 1:
                out.data[:] = x2.data
            else:
                cupy.multiply(b, x2.data, out=out.data)

    elif b == 0:
        # Compute out <- a * x1
        if out is x1:
            # out <- a * out
            if a == 1:
                pass
            else:
                scal(a, out.data)
        else:
            # out <- a * x1
            if a == 1:
                out.data[:] = x1.data
            else:
                cupy.multiply(a, x1.data, out=out.data)

    else:
        # Compute out <- a * x1 + b * x2
        # Optimize a number of alignment options. We know that a and b
        # are nonzero.
        if out is x1 and out is x2:
            # out <-- (a + b) * out
            if a + b == 0:
                out.data.fill(0)
            elif a + b == 1:
                pass
            else:
                scal(a + b, out.data)
        elif out is x1 and a == 1:
            # out <-- out + b * x2
            axpy(b, x2.data, out.data)
        elif out is x2 and b == 1:
            # out <-- a * x1 + out
            axpy(a, x1.data, out.data)
        else:
            # out <-- a * x1 + b * x2
            # No optimization for other cases of a and b; alignment doesn't
            # matter anymore.
            lico(a, x1.data, b, x2.data, out.data)


# --- Space and element classes --- #


class CupyTensorSpace(TensorSpace):

    """Tensor space implemented with CUDA arrays using the CuPy library.

    This space implements tensors of arbitrary rank over a `Field` ``F``,
    which is either the real or complex numbers.

    Its elements are represented as instances of the
    `CupyTensor` class.

    See https://github.com/cupy/cupy for details on the backend.
    """

    def __init__(self, shape, dtype='float64', device=None, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        shape : sequence of non-negative ints
            Number entries per dimension.
        dtype :
            Data type for each tuple entry. Can be provided in any
            way the `numpy.dtype` function understands, e.g.,
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            See `available_dtypes` for the list of supported data types.
        device : int, optional
            ID of the GPU device where elements should be created.
            For ``None``, the default device is chosen, which usually
            has ID 0.
        weighting : optional
            Use weighted inner product, norm, and dist. The following
            types are supported:

            `Weighting`: Use this weighting as-is.
            Compatibility with this space's elements is not checked
            during init.

            float: Weighting by a constant

            array-like: Pointwise weighting by an array of the same
            `shape` as the space.

            sequence of 1D array-likes: Per-axis (tensor product) weighting
            using broadcasting multiplication in each axis. ``None``
            entries cause the corresponding axis to be skipped.

            This option cannot be combined with ``dist``,
            ``norm`` or ``inner``.

            Default: no weighting

        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no
            inner product is defined.

            This option is ignored if ``dist``, ``norm`` or
            ``inner`` is given.

            Default: 2.0

        Other Parameters
        ----------------
        dist : callable, optional
            The distance function defining a metric on the space.
            It must accept two `CupyTensor` arguments and
            fulfill the following mathematical conditions for any
            three vectors ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``

            This option cannot be combined with ``weight``,
            ``norm`` or ``inner``.

        norm : callable, optional
            The norm implementation. It must accept an
            `CupyTensor` argument, return a float and satisfy the
            following conditions for all vectors ``x, y`` and scalars
            ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``

            By default, ``norm(x)`` is calculated as ``inner(x, x)``.

            This option cannot be combined with ``weight``,
            ``dist`` or ``inner``.

        inner : callable, optional
            The inner product implementation. It must accept two
            `CupyTensor` arguments, return a element from
            the field of the space (real or complex number) and
            satisfy the following conditions for all vectors
            ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``

            This option cannot be combined with ``weight``,
            ``dist`` or ``norm``.

        kwargs :
            Further keyword arguments are passed to the weighting
            classes.

        Examples
        --------
        Initialization with the class constructor:

        >>> space = CupyTensorSpace(3, 'float')
        >>> space
        rn(3, impl='cupy')
        >>> space.shape
        (3,)
        >>> space.dtype
        dtype('float64')

        A more convenient way is to use the factory functions with the
        ``impl='cupy'`` option:

        >>> space = odl.rn(3, impl='cupy', weighting=[1, 2, 3])
        >>> space
        rn(3, impl='cupy', weighting=[1, 2, 3])
        >>> space = odl.tensor_space((2, 3), impl='cupy', dtype=int)
        >>> space
        tensor_space((2, 3), 'int', impl='cupy')
        """
        super(CupyTensorSpace, self).__init__(shape, dtype)
        if self.dtype.char not in self.available_dtypes():
            raise ValueError('`dtype` {!r} not supported'.format(dtype))

        if device is None:
            self.__device = cupy.cuda.get_device_id()
        else:
            self.__device = int(device)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weighting = kwargs.pop('weighting', None)
        exponent = kwargs.pop('exponent', 2.0)

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if sum(x is None for x in (dist, norm, inner, weighting)) < 3:
            raise ValueError('invalid combination of options `weighting`, '
                             '`dist`, `norm` and `inner`')
        if any(x is not None for x in (dist, norm, inner)) and exponent != 2.0:
            raise ValueError('`exponent` cannot be used together with '
                             '`dist`, `norm` and `inner`')

        # Set the weighting
        if weighting is not None:
            if isinstance(weighting, Weighting):
                if weighting.impl != 'cupy':
                    raise ValueError("`weighting.impl` must be 'cupy', "
                                     '`got {!r}'.format(weighting.impl))
                if weighting.exponent != exponent:
                    raise ValueError('`weighting.exponent` conflicts with '
                                     '`exponent`: {} != {}'
                                     ''.format(weighting.exponent, exponent))
                self.__weighting = weighting
            else:
                self.__weighting = _weighting(weighting, exponent)

            # Check (afterwards) that the weighting input was sane
            if isinstance(self.weighting, CupyTensorSpaceArrayWeighting):
                if not np.can_cast(self.weighting.array.dtype, self.dtype):
                    raise ValueError(
                        'cannot cast from `weighting` data type {} to '
                        'the space `dtype` {}'
                        ''.format(dtype_str(self.weighting.array.dtype),
                                  dtype_str(self.dtype)))
                if self.weighting.array.shape != self.shape:
                    raise ValueError('array-like weights must have same '
                                     'shape {} as this space, got {}'
                                     ''.format(self.shape,
                                               self.weighting.array.shape))

        elif dist is not None:
            self.__weighting = CupyTensorSpaceCustomDist(dist)
        elif norm is not None:
            self.__weighting = CupyTensorSpaceCustomNorm(norm)
        elif inner is not None:
            self.__weighting = CupyTensorSpaceCustomInner(inner)
        else:  # all None -> no weighing
            self.__weighting = CupyTensorSpaceConstWeighting(1.0, exponent)

    @property
    def device(self):
        """The GPU device ID of this tensor space."""
        return self.__device

    @property
    def impl(self):
        """Implementation back-end of this space: ``'cupy'``."""
        return 'cupy'

    @property
    def default_order(self):
        """Default storage order for new elements in this space: ``'C'``."""
        return 'C'

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.__weighting

    @property
    def is_weighted(self):
        """Return ``True`` if the space is not weighted by constant 1.0."""
        return not (
            isinstance(self.weighting, CupyTensorSpaceConstWeighting) and
            self.weighting.const == 1.0)

    @property
    def exponent(self):
        """Exponent of the norm and distance."""
        return self.weighting.exponent

    def element(self, inp=None, order=None):
        """Create a new element.

        Parameters
        ----------
        inp : `array-like`, optional
            Input used to initialize the new element.

            If ``inp`` is `None`, an empty element is created with no
            guarantee of its state (memory allocation only).
            The new element will use ``order`` as storage order if
            provided, otherwise `default_order`.

            Otherwise, a copy is avoided whenever possible. This requires
            correct `shape` and `dtype`, and if ``order`` is provided,
            also contiguousness in that ordering. If any of these
            conditions is not met, a copy is made.

        order : {None, 'C', 'F'}, optional
            Storage order of the returned element. For ``'C'`` and ``'F'``,
            contiguous memory in the respective ordering is enforced.
            The default ``None`` enforces no contiguousness.

        Returns
        -------
        element : `CupyTensor`
            The new element created (from ``inp``).

        Notes
        -----
        This method preserves "array views" of correct size and type,
        see the examples below.

        Examples
        --------
        >>> space = odl.rn((2, 3), impl='cupy')

        Create an empty element:

        >>> empty = space.element()
        >>> empty.shape
        (2, 3)

        Initialization during creation:

        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x
        rn((2, 3), impl='cupy').element(
            [[ 1.,  2.,  3.],
             [ 4.,  5.,  6.]]
        )
        """
        if order is None:
            order_in = order
        else:
            order, order_in = str(order).upper(), order

        if order is not None and order not in ('C', 'F'):
            raise ValueError("`order` {!r} not understood".format(order_in))

        with cupy.cuda.Device(self.device):
            if inp is None:
                if order is None:
                    order = self.default_order
                arr = cupy.empty(self.shape, dtype=self.dtype, order=order)

            else:
                if inp in self and order is None:
                    # Short-circuit for space elements and no enforced ordering
                    return inp

                if hasattr(inp, 'shape') and inp.shape != self.shape:
                    raise ValueError('`inp` must have shape {}, got shape {}'
                                     ''.format(self.shape, inp.shape))

                if isinstance(inp, cupy.ndarray):
                    # Workaround for https://github.com/cupy/cupy/issues/590
                    # TODO: remove when solved
                    if (inp.dtype == self.dtype and
                            inp.device.id == self.device):
                        arr = inp
                    else:
                        arr = inp.astype(self.dtype)
                else:
                    arr = cupy.array(inp, copy=False, dtype=self.dtype,
                                     ndmin=self.ndim, order=order)

                # If the result has a 0 stride, make a copy since it would
                # produce all kinds of nasty problems. This happens for e.g.
                # results of `broadcast_to()`.
                if 0 in arr.strides:
                    arr = arr.copy()

            return self.element_type(self, arr)

    def zero(self):
        """Create a tensor filled with zeros.

        Examples
        --------
        >>> space = odl.rn(3, impl='cupy')
        >>> x = space.zero()
        >>> x
        rn(3, impl='cupy').element([ 0.,  0.,  0.])
        """
        with cupy.cuda.Device(self.device):
            arr = cupy.zeros(self.shape, dtype=self.dtype)
        return self.element(arr)

    def one(self):
        """Create a tensor filled with ones.

        Examples
        --------
        >>> space = odl.rn(3, impl='cupy')
        >>> x = space.one()
        >>> x
        rn(3, impl='cupy').element([ 1.,  1.,  1.])
        """
        with cupy.cuda.Device(self.device):
            arr = cupy.ones(self.shape, dtype=self.dtype)
        return self.element(arr)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is an instance of this space's type
            with the same `shape`,  `dtype`, `device` and
            `weighting`, ``False`` otherwise.

        Examples
        --------
        >>> space = odl.rn(2, impl='cupy')
        >>> same_space = odl.rn(2, exponent=2, impl='cupy')
        >>> same_space == space
        True

        Different `shape`, `exponent`, `dtype` or `impl`
        all result in different spaces:

        >>> diff_space = odl.rn((2, 3), impl='cupy')
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(2, exponent=1, impl='cupy')
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(2, dtype='float32', impl='cupy')
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(2, impl='numpy')
        >>> diff_space == space
        False
        >>> space == object
        False

        A `CupyTensorSpace` with the same properties is considered
        equal:

        >>> same_space = odl.CupyTensorSpace(2, dtype='float64')
        >>> same_space == space
        True
        """
        return (super(CupyTensorSpace, self).__eq__(other) and
                self.device == other.device and
                self.weighting == other.weighting)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super(CupyTensorSpace, self).__hash__(), self.device,
                     self.weighting))

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination of ``x1`` and ``x2``.

        Calculate ``out = a*x1 + b*x2`` using optimized BLAS
        routines if possible.

        Parameters
        ----------
        a, b : `TensorSpace.field` elements
            Scalars to multiply ``x1`` and ``x2`` with.
        x1, x2 : `CupyTensor`
            Summands in the linear combination.
        out : `CupyTensor`
            Tensor to which the result is written.

        Returns
        -------
        None

        Examples
        --------
        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 5, 6])
        >>> out = r3.element()
        >>> result = r3.lincomb(2, x, -1, y, out)
        >>> result
        rn(3, impl='cupy').element([-2., -1.,  0.])
        >>> result is out
        True
        """
        _lincomb_impl(a, x1, b, x2, out)

    def _dist(self, x1, x2):
        """Calculate the distance between two tensors.

        Parameters
        ----------
        x1, x2 : `CupyTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            Distance between the tensors.

        Examples
        --------
        The default case is the Euclidean distance:

        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 2, -1])
        >>> r3.dist(x, y)  # 3^2 + 4^2 = 25
        5.0

        Taking a different exponent or a weighting is also possible
        during space creation:

        >>> r3 = odl.rn(3, impl='cupy', exponent=1)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 2, -1])
        >>> r3.dist(x, y)  # 3 + 4 = 7
        7.0

        >>> r3 = odl.rn(3, impl='cupy', weighting=2, exponent=1)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 2, -1])
        >>> r3.dist(x, y)  # 2*3 + 2*4 = 14
        14.0
        """
        return self.weighting.dist(x1, x2)

    def _norm(self, x):
        """Calculate the norm of a tensor.

        Parameters
        ----------
        x : `CupyTensor`
            The tensor whose norm is calculated.

        Returns
        -------
        norm : float
            Norm of the tensor.

        Examples
        --------
        The default case is the Euclidean norm:

        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([3, 4, 0])
        >>> r3.norm(x)  # 3^2 + 4^2 = 25
        5.0

        Taking a different exponent or a weighting is also possible
        during space creation:

        >>> r3 = odl.rn(3, impl='cupy', exponent=1)
        >>> x = r3.element([3, 4, 0])
        >>> r3.norm(x)  # 3 + 4 = 7
        7.0

        >>> r3 = odl.rn(3, impl='cupy', weighting=2, exponent=1)
        >>> x = r3.element([3, 4, 0])
        >>> r3.norm(x)  # 2*3 + 2*4 = 14
        14.0
        """
        return self.weighting.norm(x)

    def _inner(self, x1, x2):
        """Raw inner product of two tensors.

        Parameters
        ----------
        x1, x2 : `CupyTensor`
            The tensors whose inner product is calculated.

        Returns
        -------
        inner : `field` element
            Inner product of the tensors.

        Examples
        --------
        The default case is the dot product:

        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 0, 1])
        >>> r3.inner(x, y)  # 1*(-1) + 2*0 + 3*1 = 2
        2.0

        Taking a different weighting is also possible during space
        creation:

        >>> r3 = odl.rn(3, impl='cupy', weighting=2)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 0, 1])
        >>> r3.inner(x, y)  # 2 * 1*(-1) + 2 * 2*0 + 2 * 3*1 = 4
        4.0
        """
        return self.weighting.inner(x1, x2)

    def _multiply(self, x1, x2, out):
        """Entry-wise product of two tensors, assigned to out.

        Parameters
        ----------
        x1, x2 : `CupyTensor`
            Factors in the product.
        out : `CupyTensor`
            Tensor to which the result is written.

        Examples
        --------
        Out-of-place evaluation:

        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 0, 1])
        >>> r3.multiply(x, y)
        rn(3, impl='cupy').element([-1.,  0.,  3.])

        In-place:

        >>> out = r3.element()
        >>> result = r3.multiply(x, y, out=out)
        >>> result
        rn(3, impl='cupy').element([-1.,  0.,  3.])
        >>> result is out
        True
        """
        x1.ufuncs.multiply(x2, out=out)

    def _divide(self, x1, x2, out):
        """Entry-wise division of two tensors, assigned to out.

        Parameters
        ----------
        x1, x2 : `CupyTensor`
            Dividend and divisor in the quotient.
        out : `CupyTensor`
            Tensor to which the result is written.

        Examples
        --------
        Out-of-place evaluation:

        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 2, 1])
        >>> r3.divide(x, y)
        rn(3, impl='cupy').element([-1.,  1.,  3.])

        In-place:

        >>> out = r3.element()
        >>> result = r3.divide(x, y, out=out)
        >>> result
        rn(3, impl='cupy').element([-1.,  1.,  3.])
        >>> result is out
        True
        """
        x1.ufuncs.divide(x2, out=out)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim == 1:
            posargs = [self.size]
        else:
            posargs = [self.shape]

        if self.is_real:
            constructor_name = 'rn'
        elif self.is_complex:
            constructor_name = 'cn'
        else:
            constructor_name = 'tensor_space'

        if (constructor_name == 'tensor_space' or
                (not self.is_real and not self.is_complex) or
                self.dtype != self.default_dtype(self.field)):
            posargs.append(dtype_str(self.dtype))

        optargs = [('impl', self.impl, 'numpy'),  # for the helper functions
                   ('device', self.device, cupy.cuda.get_device_id())]
        inner_str = signature_string(posargs, optargs)
        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str

        return '{}({})'.format(constructor_name, inner_str)

    @property
    def element_type(self):
        """`CupyTensor`"""
        return CupyTensor

    @staticmethod
    def available_dtypes():
        """Return the data types available for this space."""
        dtypes = (np.sctypes['float'] +
                  np.sctypes['complex'] +
                  np.sctypes['int']
                  + np.sctypes['uint']
                  + [bool])
        dtypes.remove(np.float128)
        dtypes.remove(np.complex256)
        return tuple(np.dtype(dtype) for dtype in dtypes)

    @staticmethod
    def default_dtype(field=None):
        """Return the default data type of this space type for a given field.

        Parameters
        ----------
        field : `Field`, optional
            Set of numbers to be represented by a data type.
            Currently supported : `RealNumbers`, `ComplexNumbers`.
            Default: `RealNumbers`

        Returns
        -------
        dtype : `numpy.dtype`
            Numpy data type specifier. The returned defaults are:

            - ``RealNumbers()`` or ``None`` : ``np.dtype('float64')``
            - ``ComplexNumbers()`` : ``np.dtype('complex128')``

            These choices correspond to the defaults of the ``cupy``
            library.
        """
        if field is None or field == RealNumbers():
            return np.dtype('float64')
        elif field == ComplexNumbers():
            return np.dtype('complex128')
        else:
            raise ValueError('no default data type defined for field {}.'
                             ''.format(field))


class CupyTensor(Tensor):

    """Representation of an `CupyTensorSpace` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        super(CupyTensor, self).__init__(space)
        self.__data = data

    @property
    def data(self):
        """Raw `cupy.core.core.ndarray` representing the data."""
        return self.__data

    @property
    def ndim(self):
        """Number of axes (=dimensions) of this tensor."""
        return self.space.ndim

    @property
    def device(self):
        """The GPU device on which this tensor lies."""
        return self.space.device

    def asarray(self, out=None):
        """Extract the data of this element as a `numpy.ndarray`.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            Array to which the result should be written.
            Has to be contiguous and of the correct data type.

        Returns
        -------
        asarray : `numpy.ndarray`
            Numpy array of the same `dtype` and `shape` this tensor.
            If ``out`` was given, the returned object is a reference to it.

        Examples
        --------
        By default, a new array is created:

        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([1, 2, 3])
        >>> x.asarray()
        array([ 1.,  2.,  3.])
        >>> int_spc = odl.tensor_space(3, impl='cupy', dtype=int)
        >>> x = int_spc.element([1, 2, 3])
        >>> x.asarray()
        array([1, 2, 3])
        >>> tensors = odl.rn((2, 3), impl='cupy', dtype='float32')
        >>> x = tensors.element([[1, 2, 3],
        ...                      [4, 5, 6]])
        >>> x.asarray()
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)

        Using the out parameter, the array can be filled in-place:

        >>> out = np.empty((2, 3), dtype='float32')
        >>> result = x.asarray(out=out)
        >>> out
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> result is out
        True
        """
        if out is None:
            return cupy.asnumpy(self.data)
        else:
            if out.shape != self.shape:
                raise ValueError('`out` must have shape {}, got shape {}'
                                 ''.format(self.shape, out.shape))
            if out.dtype != self.dtype:
                raise ValueError('`out` must have dtype {}, got dtype {}'
                                 ''.format(self.dtype, out.dtype))
            self.data.data.copy_to_host(
                out.ctypes.data_as(np.ctypeslib.ctypes.c_void_p),
                self.size * self.itemsize)
            return out

    @property
    def data_ptr(self):
        """A raw pointer to the data container.

        Examples
        --------
        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.one()
        >>> x.data_ptr  # doctest: +SKIP
        47259975936
        """
        return self.data.ptr

    def __eq__(self, other):
        """Return ``self == other``.

        Parameters
        ----------
        other :
            Object to be compared with ``self``.

        Returns
        -------
        equals : bool
            ``True`` if all entries of ``other`` are equal to this
            tensor's entries, ``False`` otherwise.

        Notes
        -----
        The element-by-element comparison is performed on the CPU,
        i.e. it involves data transfer to host memory, which is slow.

        Examples
        --------
        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([1, 2, 3])
        >>> same_x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, -2, -3])
        >>> x == same_x
        True
        >>> x == y
        False

        Space membership matters:

        >>> int_spc = odl.tensor_space(3, impl='cupy', dtype=int)
        >>> x_int = int_spc.element([1, 2, 3])
        >>> x == x_int
        False
        """
        if other is self:
            return True
        elif other not in self.space:
            return False
        else:
            return bool((self.data == other.data).all())

    def copy(self):
        """Create an identical (deep) copy of this tensor.

        Returns
        -------
        copy : `CupyTensor`
            A deep copy.

        Examples
        --------
        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([1, 2, 3])
        >>> y = x.copy()
        >>> y
        rn(3, impl='cupy').element([ 1.,  2.,  3.])
        >>> x == y
        True
        >>> x is y
        False
        """
        return self.space.element(self.data.copy())

    def __getitem__(self, indices):
        """Access values of this tensor.

        Parameters
        ----------
        indices : index expression
            The position(s) that should be accessed.

        Returns
        -------
        values : scalar or `cupy.core.core.ndarray`
            The value(s) at the index (indices).

        Examples
        --------
        Indexing rules follow roughly the Numpy style, as far (or "fancy")
        as supported:

        >>> r5 = odl.rn(5, impl='cupy')
        >>> x = r5.element([1, 2, 3, 4, 5])
        >>> x[1:4]
        rn(3, impl='cupy').element([ 2.,  3.,  4.])
        >>> x[::2]
        rn(3, impl='cupy').element([ 1.,  3.,  5.])

        The returned views are writable, so modificatons alter the
        original array:

        >>> view = x[1:4]
        >>> view[:] = -1
        >>> view
        rn(3, impl='cupy').element([-1., -1., -1.])
        >>> x
        rn(5, impl='cupy').element([ 1., -1., -1., -1.,  5.])

        Multi-indexing is also directly supported:

        >>> tensors = odl.rn((2, 3), impl='cupy')
        >>> x = tensors.element([[1, 2, 3],
        ...                      [4, 5, 6]])
        >>> x[1, 2]
        6.0
        >>> x[1]  # row with index 1
        rn(3, impl='cupy').element([ 4.,  5.,  6.])
        >>> view = x[:, ::2]
        >>> view
        rn((2, 2), impl='cupy').element(
            [[ 1.,  3.],
             [ 4.,  6.]]
        )
        >>> view[:] = [[0, 0],
        ...            [0, 0]]
        >>> x
        rn((2, 3), impl='cupy').element(
            [[ 0.,  2.,  0.],
             [ 0.,  5.,  0.]]
        )
        """
        arr = self.data[indices]
        if arr.shape == ():
            if arr.dtype.kind == 'f':
                return float(cupy.asnumpy(arr))
            elif arr.dtype.kind == 'c':
                return complex(cupy.asnumpy(arr))
            elif arr.dtype.kind in ('u', 'i'):
                return int(cupy.asnumpy(arr))
            else:
                raise RuntimeError("no conversion for dtype {}"
                                   "".format(arr.dtype))
        else:
            space = type(self.space)(arr.shape, dtype=self.dtype,
                                     device=self.device)
            return space.element(arr)

    def __setitem__(self, indices, values):
        """Set values of this tensor.

        Parameters
        ----------
        indices : index expression
            The position(s) that should be accessed.
        values : scalar or `array-like`
            The value(s) that are to be assigned.

            If ``indices`` is an int (1D) or a sequence of ints,
            ``value`` must be scalar.

            Otherwise, ``value`` must be broadcastable to the shape of
            the sliced view according to the Numpy broadcasting rules.

        Examples
        --------
        In 1D, Values can be set with scalars or arrays that match the
        shape of the slice:

        >>> r5 = odl.rn(5, impl='cupy')
        >>> x = r5.element([1, 2, 3, 4, 5])
        >>> x[1:4] = 0
        >>> x
        rn(5, impl='cupy').element([ 1.,  0.,  0.,  0.,  5.])
        >>> x[1:4] = [-1, 1, -1]
        >>> x
        rn(5, impl='cupy').element([ 1., -1.,  1., -1.,  5.])
        >>> y = r5.element([5, 5, 5, 8, 8])
        >>> x[:] = y
        >>> x
        rn(5, impl='cupy').element([ 5.,  5.,  5.,  8.,  8.])

        In higher dimensions, broadcasting can be applied to assign
        values:

        >>> tensors = odl.rn((2, 3), impl='cupy')
        >>> x = tensors.element([[1, 2, 3],
        ...                      [4, 5, 6]])
        >>> x[:] = [[6], [3]]  # rhs mimics (2, 1) shape
        >>> x
        rn((2, 3), impl='cupy').element(
            [[ 6.,  6.,  6.],
             [ 3.,  3.,  3.]]
        )

        Be aware of unsafe casts and over-/underflows, there
        will be warnings at maximum.

        >>> int_r3 = odl.tensor_space(3, impl='cupy', dtype='uint32')
        >>> x = int_r3.element([1, 2, 3])
        >>> x[0] = -1
        >>> x[0]
        4294967295
        """
        if isinstance(values, CupyTensor):
            self.data[indices] = values.data
        elif np.isscalar(values):
            self.data[indices] = values
        else:
            values = cupy.array(values, dtype=self.dtype, copy=False)
            self.data[indices] = values

    def __int__(self):
        """Return ``int(self)``.

        Returns
        -------
        int : int
            Integer representing this tensor.

        Raises
        ------
        TypeError
            If the tensor is of `size` != 1.
        """
        if self.size != 1:
            raise TypeError('only size 1 tensors can be converted to int')
        return int(self[(0,) * self.ndim])

    def __long__(self):
        """Return ``long(self)``.

        The `long` method is only available in Python 2.

        Returns
        -------
        long : `long`
            Integer representing this tensor.

        Raises
        ------
        TypeError
            If the tensor is of `size` != 1.
        """
        if self.size != 1:
            raise TypeError('only size 1 tensors can be converted to long')
        return long(self[(0,) * self.ndim])

    def __float__(self):
        """Return ``float(self)``.

        Returns
        -------
        float : float
            Floating point number representing this tensor.

        Raises
        ------
        TypeError
            If the tensor is of `size` != 1.
        """
        if self.size != 1:
            raise TypeError('only size 1 tensors can be converted to float')
        return float(self[(0,) * self.ndim])

    def __complex__(self):
        """Return ``complex(self)``.

        Returns
        -------
        complex : `complex`
            Complex floating point number representing this tensor.

        Raises
        ------
        TypeError
            If the tensor is of `size` != 1.
        """
        if self.size != 1:
            raise TypeError('only size 1 tensors can be converted to complex')
        return complex(self[(0,) * self.ndim])

    def __str__(self):
        """Return ``str(self)``."""
        return str(self.data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Interface to Numpy's ufunc machinery.

        This method is called by Numpy version 1.13 and higher as a single
        point for the ufunc dispatch logic. An object implementing
        ``__array_ufunc__`` takes over control when a `numpy.ufunc` is
        called on it, allowing it to use custom implementations and
        output types.

        This includes handling of in-place arithmetic like
        ``npy_array += custom_obj``. In this case, the custom object's
        ``__array_ufunc__`` takes precedence over the baseline
        `numpy.ndarray` implementation. It will be called with
        ``npy_array`` as ``out`` argument, which ensures that the
        returned object is a Numpy array. For this to work properly,
        ``__array_ufunc__`` has to accept Numpy arrays as ``out`` arguments.
        The same holds analogously for GPU arrays.

        See the `corresponding NEP`_ and the `interface documentation`_
        for further details. See also the `general documentation on
        Numpy ufuncs`_.

        .. warning::
            Apart from ``'__call__'`` (invoked by, e.g., ``np.add(x, y))``,
            CuPy has no native implementation of ufunc methods like
            ``'reduce'`` or ``'accumulate'``. We manually implement the
            mappings (covering most use cases)

            - ``np.add.reduce`` -> ``cupy.sum``
            - ``np.add.accumulate`` -> ``cupy.cumsum``
            - ``np.multiply.reduce`` -> ``cupy.prod``
            - ``np.multiply.reduce`` -> ``cupy.cumprod``.

            **All other such methods will run Numpy code and be slow**!

            Please consult the `CuPy documentation on ufuncs
            <https://docs-cupy.chainer.org/en/stable/reference/ufunc.html>`_
            to check the current state of the library.

        .. note::
            When an ``out`` parameter is specified, and (one of) it has
            type `numpy.ndarray`, the inputs are converted to Numpy
            arrays, and the Numpy ufunc is invoked.

        .. note::
            When using operations that alter the shape (like ``reduce``),
            or the data type (can be any of the methods),
            the resulting array is wrapped in a space of the same
            type as ``self.space``, propagating space properties like
            `exponent` or `weighting` as closely as possible.

        Parameters
        ----------
        ufunc : `numpy.ufunc`
            Ufunc that should be called on ``self``.
        method : str
            Method on ``ufunc`` that should be called on ``self``.
            Possible values:

            ``'__call__'``, ``'accumulate'``, ``'at'``, ``'outer'``,
            ``'reduce'``, ``'reduceat'``

        input1, ..., inputN :
            Positional arguments to ``ufunc.method``.
        kwargs :
            Keyword arguments to ``ufunc.method``.

        Returns
        -------
        ufunc_result : `CupyTensor`, `numpy.ndarray` or tuple
            Result of the ufunc evaluation. If no ``out`` keyword argument
            was given, the result is a `Tensor` or a tuple
            of such, depending on the number of outputs of ``ufunc``.
            If ``out`` was provided, the returned object or tuple entries
            refer(s) to ``out``.

        Examples
        --------
        We apply `numpy.add` to ODL tensors:

        >>> r3 = odl.rn(3, impl='cupy')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, -2, -3])
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        rn(3, impl='cupy').element([ 0.,  0.,  0.])
        >>> np.add(x, y)  # same mechanism for Numpy >= 1.13
        rn(3, impl='cupy').element([ 0.,  0.,  0.])

        As ``out``, a Numpy array or an ODL tensor can be given (wrapped
        in a sequence):

        >>> out = r3.element()
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out,))
        >>> out
        rn(3, impl='cupy').element([ 0.,  0.,  0.])
        >>> res is out
        True
        >>> out_arr = np.empty(3)
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out_arr,))
        >>> out_arr
        array([ 0.,  0.,  0.])
        >>> res is out_arr
        True

        With multiple dimensions:

        >>> r23 = odl.rn((2, 3), impl='cupy')
        >>> x = y = r23.one()
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        rn((2, 3), impl='cupy').element(
            [[ 2.,  2.,  2.],
             [ 2.,  2.,  2.]]
        )

        The ``ufunc.accumulate`` method retains the original `shape` and
        `dtype`. The latter can be changed with the ``dtype`` parameter:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'accumulate', x)
        rn(3, impl='cupy').element([ 1.,  3.,  6.])
        >>> np.add.accumulate(x)  # same mechanism for Numpy >= 1.13
        rn(3, impl='cupy').element([ 1.,  3.,  6.])

        For multi-dimensional tensors, an optional ``axis`` parameter
        can be provided:

        >>> z = r23.one()
        >>> z.__array_ufunc__(np.add, 'accumulate', z, axis=1)
        rn((2, 3), impl='cupy').element(
            [[ 1.,  2.,  3.],
             [ 1.,  2.,  3.]]
        )

        The ``ufunc.at`` method operates in-place. Here we add the second
        operand ``[5, 10]`` to ``x`` at indices ``[0, 2]``:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'at', x, [0, 2], [5, 10])
        >>> x
        rn(3, impl='cupy').element([  6.,   2.,  13.])

        For outer-product-type operations, i.e., operations where the result
        shape is the sum of the individual shapes, the ``ufunc.outer``
        method can be used:

        >>> x = odl.rn(2, impl='cupy').element([0, 3])
        >>> y = odl.rn(3, impl='cupy').element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'outer', x, y)
        rn((2, 3), impl='cupy').element(
            [[ 1.,  2.,  3.],
             [ 4.,  5.,  6.]]
        )
        >>> y.__array_ufunc__(np.add, 'outer', y, x)
        rn((3, 2), impl='cupy').element(
            [[ 1.,  4.],
             [ 2.,  5.],
             [ 3.,  6.]]
        )

        Using ``ufunc.reduce`` produces a scalar, which can be avoided with
        ``keepdims=True``:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'reduce', x)
        6.0
        >>> x.__array_ufunc__(np.add, 'reduce', x, keepdims=True)
        rn(1, impl='cupy').element([ 6.])

        In multiple dimensions, ``axis`` can be provided for reduction over
        selected axes:

        >>> z = r23.element([[1, 2, 3],
        ...                  [4, 5, 6]])
        >>> z.__array_ufunc__(np.add, 'reduce', z, axis=1)
        rn(2, impl='cupy').element([  6.,  15.])

        Finally, ``add.reduceat`` is a combination of ``reduce`` and
        ``at`` with rather flexible and complex semantics (see the
        `reduceat documentation`_ for details):

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'reduceat', x, [0, 1])
        rn(2, impl='cupy').element([ 1.,  5.])

        References
        ----------
        .. _corresponding NEP:
           https://github.com/numpy/numpy/blob/master/doc/neps/\
ufunc-overrides.rst

        .. _interface documentation:
           https://github.com/charris/numpy/blob/master/doc/source/reference/\
arrays.classes.rst#special-attributes-and-methods

        .. _general documentation on Numpy ufuncs:
           https://docs.scipy.org/doc/numpy/reference/ufuncs.html

        .. _reduceat documentation:
           https://docs.scipy.org/doc/numpy/reference/generated/\
numpy.ufunc.reduceat.html
        """
        # --- Process `out` --- #

        # Unwrap out if provided. The output parameters are all wrapped
        # in one tuple, even if there is only one.
        out_tuple = kwargs.pop('out', ())

        # Check number of `out` args, depending on `method`
        if method == '__call__' and len(out_tuple) not in (0, ufunc.nout):
            raise ValueError(
                "need 0 or {} `out` arguments for `method='__call__'`, "
                'got {}'.format(ufunc.nout, len(out_tuple)))
        elif method != '__call__' and len(out_tuple) not in (0, 1):
            raise ValueError(
                "need 0 or 1 `out` arguments for `method={!r}`, "
                'got {}'.format(method, len(out_tuple)))

        # We allow our own tensors, the data container type and
        # `numpy.ndarray` objects as `out` (see docs for reason for the
        # latter)
        valid_types = (type(self), type(self.data), np.ndarray)
        if not all(isinstance(o, valid_types) or o is None
                   for o in out_tuple):
            return NotImplemented

        # Determine native ufunc vs. Numpy ufunc
        if any(isinstance(o, np.ndarray) for o in out_tuple):
            native_ufunc = None
            use_native = False
        else:
            native_ufunc = getattr(cupy, ufunc.__name__, None)
            use_native = (native_ufunc is not None and
                          hasattr(native_ufunc, method))

        # Assign to `out` or `out1` and `out2`, respectively, unwrapping the
        # data container
        out = out1 = out2 = None
        if len(out_tuple) == 1:
            if isinstance(out_tuple[0], type(self)):
                out = out_tuple[0].data
            else:
                out = out_tuple[0]
        elif len(out_tuple) == 2:
            if isinstance(out_tuple[0], type(self)):
                out1 = out_tuple[0].data
            else:
                out1 = out_tuple[0]
            if isinstance(out_tuple[1], type(self)):
                out1 = out_tuple[1].data
            else:
                out1 = out_tuple[1]

        # --- Process `inputs` --- #

        # Pull out the data container of the inputs if necessary
        inputs = tuple(
            inp.data if isinstance(inp, type(self)) else inp
            for inp in inputs)

        if use_native:
            # TODO: remove when upstream issue is fixed
            # For native ufuncs, we turn non-scalar inputs into cupy arrays,
            # as a workaround for https://github.com/cupy/cupy/issues/594
            inputs, orig_inputs = [], inputs
            for inp in orig_inputs:
                if (isinstance(inp, cupy.ndarray) or
                        np.isscalar(inp) or
                        inp is None):
                    inputs.append(inp)
                else:
                    inputs.append(cupy.array(inp))
        elif method != 'at':
            # TODO: remove when upstream issue is fixed
            # For non-native ufuncs (except `at`), we need ot cast our tensors
            # and Cupy arrays to Numpy arrays explicitly, since `__array__`
            # and friends are not implemented. See
            # https://github.com/cupy/cupy/issues/589
            inputs, orig_inputs = [], inputs
            for inp in orig_inputs:
                if isinstance(inp, cupy.ndarray):
                    inputs.append(cupy.asnumpy(inp))
                elif isinstance(inp, CupyTensor):
                    inputs.append(cupy.asnumpy(inp.data))
                else:
                    inputs.append(inp)

        # --- Get some parameters for later --- #

        # Arguments for space constructors
        exponent = self.space.exponent
        weighting = self.space.weighting

        # --- Evaluate ufunc --- #

        if method == '__call__':
            if ufunc.nout == 1:
                if use_native:
                    kwargs['out'] = out  # No tuple packing for cupy
                    res = native_ufunc(*inputs, **kwargs)
                else:
                    kwargs['out'] = (out,)
                    # Everything is cast to Numpy arrays by the parent method;
                    # the result can be a Numpy array or a tensor
                    res = super(CupyTensor, self).__array_ufunc__(
                        ufunc, '__call__', *inputs, **kwargs)

                # Wrap result if necessary (lazily)
                if out is None:
                    if is_floating_dtype(res.dtype):
                        # Weighting contains exponent
                        spc_kwargs = {'weighting': weighting}
                    else:
                        # No `exponent` or `weighting` applicable
                        spc_kwargs = {}
                    out_space = type(self.space)(
                        self.shape, res.dtype, self.device, **spc_kwargs)
                    return out_space.element(res)
                else:
                    # `out` may be the unwrapped version, return the original
                    return out_tuple[0]

            elif ufunc.nout == 2:
                kwargs['out'] = (out1, out2)
                if use_native:
                    res1, res2 = native_ufunc(*inputs, **kwargs)
                else:
                    # Everything is cast to Numpy arrays by the parent method;
                    # the results can be Numpy arrays or tensors
                    res1, res2 = super(CupyTensor, self).__array_ufunc__(
                        ufunc, '__call__', *inputs, **kwargs)

                # Wrap results if necessary (lazily)
                # We don't use exponents or weightings since we don't know
                # how to map them to the spaces
                if out1 is None:
                    res_space = type(self.space)(
                        self.shape, res1.dtype, self.device)
                    result1 = res_space.element(res1)
                else:
                    result1 = out_tuple[0]

                if out2 is None:
                    res_space = type(self.space)(
                        self.shape, res2.dtype, self.device)
                    result2 = res_space.element(res2)
                else:
                    result2 = out_tuple[1]

                return result1, result2

            else:
                raise NotImplementedError('nout = {} not supported'
                                          ''.format(ufunc.nout))

        elif method == 'at':
            native_method = getattr(native_ufunc, 'at', None)
            use_native = (use_native and native_method is not None)

            def eval_at_via_npy(*inputs, **kwargs):
                import ctypes
                cupy_arr = inputs[0]
                npy_arr = cupy.asnumpy(cupy_arr)
                new_inputs = (npy_arr,) + inputs[1:]
                super(CupyTensor, self).__array_ufunc__(
                    ufunc, method, *new_inputs, **kwargs)
                # Workaround for https://github.com/cupy/cupy/issues/593
                # TODO: use cupy_arr[:] = npy_arr when available
                cupy_arr.data.copy_from_host(
                    npy_arr.ctypes.data_as(ctypes.c_void_p), npy_arr.nbytes)

            if use_native:
                # Native method could exist but raise `NotImplementedError`
                # or return `NotImplemented`. We fall back to Numpy also in
                # that situation.
                try:
                    res = native_method(*inputs, **kwargs)
                except NotImplementedError:
                    eval_at_via_npy(*inputs, **kwargs)
                else:
                    if res is NotImplemented:
                        eval_at_via_npy(*inputs, **kwargs)
            else:
                eval_at_via_npy(*inputs, **kwargs)

        else:  # method != '__call__'
            kwargs['out'] = (out,)
            native_method = getattr(native_ufunc, method, None)
            use_native = (use_native and native_method is not None)

            if use_native:
                # Native method could exist but raise `NotImplementedError`
                # or return `NotImplemented`. We fall back to Numpy also in
                # that situation.
                try:
                    res = native_method(*inputs, **kwargs)
                except NotImplementedError:
                    res = super(CupyTensor, self).__array_ufunc__(
                        ufunc, method, *inputs, **kwargs)
                else:
                    if res is NotImplemented:
                        res = super(CupyTensor, self).__array_ufunc__(
                            ufunc, method, *inputs, **kwargs)

            else:
                res = super(CupyTensor, self).__array_ufunc__(
                    ufunc, method, *inputs, **kwargs)

            # Shortcut for scalar or no return value
            if np.isscalar(res) or res is None:
                # The first occurs for `reduce` with all axes,
                # the second for in-place stuff (`at` currently)
                return res

            # Wrap result if necessary (lazily)
            if out is None:
                if is_floating_dtype(res.dtype):
                    if res.shape != self.shape:
                        # Don't propagate weighting if shape changes
                        weighting = CupyTensorSpaceConstWeighting(
                            1.0, exponent)
                    spc_kwargs = {'weighting': weighting}
                else:
                    spc_kwargs = {}

                res_space = type(self.space)(
                    res.shape, res.dtype, self.device, **spc_kwargs)
                result = res_space.element(res)
            else:
                result = out_tuple[0]

            return result

    @property
    def ufuncs(self):
        """Access to NumPy style ufuncs.

        Examples
        --------
        >>> r2 = odl.rn(2, impl='cupy')
        >>> x = r2.element([1, -2])
        >>> x.ufuncs.absolute()
        rn(2, impl='cupy').element([ 1.,  2.])

        These functions can also be used with broadcasting or
        array-like input:

        >>> x.ufuncs.add(3)
        rn(2, impl='cupy').element([ 4.,  1.])
        >>> x.ufuncs.subtract([3, 3])
        rn(2, impl='cupy').element([-2., -5.])

        There is also support for various reductions
        (sum, prod, amin, amax):

        >>> x.ufuncs.sum()
        -1.0
        >>> x.ufuncs.prod()
        -2.0

        They also support an out parameter

        >>> y = r2.element([3, 4])
        >>> out = r2.element()
        >>> result = x.ufuncs.add(y, out=out)
        >>> result
        rn(2, impl='cupy').element([ 4.,  2.])
        >>> result is out
        True

        Notes
        -----
        Those ufuncs which are implemented natively on the GPU incur no
        significant overhead. However, for missing functions, a fallback
        Numpy implementation is used which causes significant overhead
        due to data copies between host and device.
        """
        # TODO: Test with some native ufuncs, then remove this attribute
        return super(CupyTensor, self).ufuncs

    @property
    def real(self):
        """Real part of this tensor.

        Returns
        -------
        real : `CupyTensor` view with real dtype
            The real part of this tensor as an element of an `rn` space.
        """
        if self.space.is_real:
            return self
        else:
            return self.space.real_space.element(self.data.real)

    @real.setter
    def real(self, newreal):
        """Setter for the real part.

        This method is invoked by ``tensor.real = other``.

        Parameters
        ----------
        newreal : `array-like` or scalar
            The new real part for this tensor.
        """
        self.data.real[:] = newreal

    @property
    def imag(self):
        """Imaginary part of this tensor.

        Returns
        -------
        imag : `CupyTensor`
            The imaginary part of this tensor as an element of an `rn` space.
        """
        if self.space.is_real:
            return self.space.zero()
        else:
            return self.space.real_space.element(self.data.imag)

    @imag.setter
    def imag(self, newimag):
        """Setter for the imaginary part.

        This method is invoked by ``tensor.imag = other``.

        Parameters
        ----------
        newimag : `array-like` or scalar
            The new imaginary part for this tensor.
        """
        self.data.imag[:] = newimag

    def conj(self, out=None):
        """Complex conjugate of this tensor.

        Parameters
        ----------
        out : `CupyTensor`, optional
            Tensor to which the complex conjugate is written.
            Must be an element of this tensor's space.

        Returns
        -------
        out : `CupyTensor`
            The complex conjugate tensor. If ``out`` was provided,
            the returned object is a reference to it.
        """
        if out is None:
            if self.space.is_real:
                return self.copy()
            else:
                return self.space.element(self.data.conj())
        else:
            if self.space.is_real:
                self.assign(out)
            else:
                # In-place not available as it seems
                out[:] = self.data.conj()
            return out

    def __ipow__(self, other):
        """Return ``self **= other``."""
        try:
            if other == int(other):
                return super(CupyTensorSpace, self).__ipow__(other)
        except TypeError:
            pass

        self.ufuncs.power(self.data, other, out=self.data)
        return self


# --- Weightings --- #


def _weighting(weights, exponent):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weights):
        weighting = CupyTensorSpaceConstWeighting(weights, exponent=exponent)
    else:
        # TODO: sequence of 1D array-likes, see
        # https://github.com/odlgroup/odl/pull/1238
        weights = cupy.array(weights, copy=False)
        weighting = CupyTensorSpaceArrayWeighting(weights, exponent=exponent)
    return weighting


# Kernels for space functions
#
# T = generic type
# R = real floating point type, usually for norm or dist output
# W = real type for weights, can be floating point or integer
#
# Note: the kernels with an output type that does not also occur as an input
# type must be called with output argument since the output type cannot be
# inferred. The ouptut array must have shape `()` for full reduction.


if CUPY_AVAILABLE:
    dotw = cupy.ReductionKernel(in_params='T x, T y, W w',
                                out_params='T res',
                                map_expr='x * y * w',
                                reduce_expr='a + b',
                                post_map_expr='res = a',
                                identity='0',
                                name='dotw')

    nrm0 = cupy.ReductionKernel(in_params='T x',
                                out_params='R res',
                                map_expr='x != 0',
                                reduce_expr='a + b',
                                post_map_expr='res = a',
                                identity='0',
                                name='nrm0')

    nrm1 = cupy.ReductionKernel(in_params='T x',
                                out_params='R res',
                                map_expr='abs(x)',
                                reduce_expr='a + b',
                                post_map_expr='res = a',
                                identity='0',
                                name='nrm1w')

    nrm1w = cupy.ReductionKernel(in_params='T x, W w',
                                 out_params='R res',
                                 map_expr='abs(x) * w',
                                 reduce_expr='a + b',
                                 post_map_expr='res = a',
                                 identity='0',
                                 name='nrm1w')

    nrm2 = cupy.ReductionKernel(in_params='T x',
                                out_params='R res',
                                map_expr='x * x',
                                reduce_expr='a + b',
                                post_map_expr='res = sqrt(a)',
                                identity='0',
                                name='nrm2')

    nrm2w = cupy.ReductionKernel(in_params='T x, W w',
                                 out_params='R res',
                                 map_expr='x * x * w',
                                 reduce_expr='a + b',
                                 post_map_expr='res = sqrt(a)',
                                 identity='0',
                                 name='nrm2w')

    nrminf = cupy.ReductionKernel(in_params='T x',
                                  out_params='R res',
                                  map_expr='abs(x)',
                                  reduce_expr='a > b ? a : b',
                                  post_map_expr='res = a',
                                  identity='0',
                                  name='nrminf')

    nrmneginf = cupy.ReductionKernel(in_params='T x',
                                     out_params='R res',
                                     map_expr='abs(x)',
                                     reduce_expr='a > b ? b : a',
                                     post_map_expr='res = a',
                                     identity='0',
                                     name='nrmneginf')

    nrmp = cupy.ReductionKernel(in_params='T x, R p',
                                out_params='R res',
                                map_expr='pow(abs(x), p)',
                                reduce_expr='a + b',
                                post_map_expr='res = pow(a, 1 / p)',
                                identity='0',
                                name='nrmp')

    nrmpw = cupy.ReductionKernel(in_params='T x, R p, W w',
                                 out_params='R res',
                                 map_expr='pow(abs(x), p) * w',
                                 reduce_expr='a + b',
                                 post_map_expr='res = pow(a, 1 / p)',
                                 identity='0',
                                 name='nrmpw')

    dist0 = cupy.ReductionKernel(in_params='T x, T y',
                                 out_params='R res',
                                 map_expr='x != y',
                                 reduce_expr='a + b',
                                 post_map_expr='res = a',
                                 identity='0',
                                 name='dist0')

    dist1 = cupy.ReductionKernel(in_params='T x, T y',
                                 out_params='R res',
                                 map_expr='abs(x - y)',
                                 reduce_expr='a + b',
                                 post_map_expr='res = a',
                                 identity='0',
                                 name='dist1')

    dist1w = cupy.ReductionKernel(in_params='T x, T y, W w',
                                  out_params='T res',
                                  map_expr='abs(x - y) * w',
                                  reduce_expr='a + b',
                                  post_map_expr='res = a',
                                  identity='0',
                                  name='dist1w')

    dist2 = cupy.ReductionKernel(in_params='T x, T y',
                                 out_params='R res',
                                 map_expr='abs(x - y) * abs(x - y)',
                                 reduce_expr='a + b',
                                 post_map_expr='res = sqrt(a)',
                                 identity='0',
                                 name='dist2')

    dist2w = cupy.ReductionKernel(in_params='T x, T y, W w',
                                  out_params='R res',
                                  map_expr='abs(x - y) * abs(x - y) * w',
                                  reduce_expr='a + b',
                                  post_map_expr='res = sqrt(a)',
                                  identity='0',
                                  name='dist2w')

    distinf = cupy.ReductionKernel(in_params='T x, T y',
                                   out_params='R res',
                                   map_expr='abs(x - y)',
                                   reduce_expr='a > b ? a : b',
                                   post_map_expr='res = a',
                                   identity='0',
                                   name='distinf')

    distneginf = cupy.ReductionKernel(in_params='T x, T y',
                                      out_params='R res',
                                      map_expr='abs(x - y)',
                                      reduce_expr='a > b ? b : a',
                                      post_map_expr='res = a',
                                      identity='0',
                                      name='distneginf')

    distp = cupy.ReductionKernel(in_params='T x, T y, R p',
                                 out_params='R res',
                                 map_expr='pow(abs(x - y), p)',
                                 reduce_expr='a + b',
                                 post_map_expr='res = pow(a, 1 / p)',
                                 identity='0',
                                 name='distp')

    distpw = cupy.ReductionKernel(in_params='T x, T y, R p, W w',
                                  out_params='R res',
                                  map_expr='pow(abs(x - y), p) * w',
                                  reduce_expr='a + b',
                                  post_map_expr='res = pow(a, 1 / p)',
                                  identity='0',
                                  name='distpw')


class CupyTensorSpaceArrayWeighting(ArrayWeighting):

    """Array weighting for `CupyTensorSpace`.

    See `ArrayWeighting` for further details.
    """

    def __init__(self, array, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        array : `array-like`, one-dim.
            Weighting array of the inner product, norm and distance.
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        if isinstance(array, CupyTensor):
            array = array.data
        elif not isinstance(array, cupy.ndarray):
            array = cupy.array(array, copy=False)
        super(CupyTensorSpaceArrayWeighting, self).__init__(
            array, impl='cupy', exponent=exponent)

    def inner(self, x1, x2):
        """Return the weighted inner product of two tensors.

        Parameters
        ----------
        x1, x2 : `CupyTensor`
            Tensors whose inner product is calculated.

        Returns
        -------
        inner : float or complex
            The inner product of the two provided tensors.
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))
        else:
            return x1.space.field.element(dotw(x1.data, x2.data, self.array))

    def norm(self, x):
        """Return the weighted norm of a tensor.

        Parameters
        ----------
        x : `CupyTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the provided tensor.
        """
        # Define scalar output array
        if is_floating_dtype(x.dtype):
            out_dtype = real_dtype(x.dtype)
        else:
            out_dtype = float
        out = cupy.empty((), dtype=out_dtype)

        # Run reduction kernel (returns the output)
        if self.exponent == 0:
            return float(nrm0(x.data, out))
        elif self.exponent == 1:
            return float(nrm1w(x.data, self.array, out))
        elif self.exponent == 2:
            return float(nrm2w(x.data, self.array, out))
        elif self.exponent == float('inf'):
            return float(nrminf(x.data, out))
        elif self.exponent == -float('inf'):
            return float(nrmneginf(x.data, out))
        else:
            return float(nrmpw(x.data, self.exponent, self.array, out))

    def dist(self, x1, x2):
        """Return the weighted distance of two tensors.

        Parameters
        ----------
        x1, x2 : `CupyTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the provided tensors.
        """
        # Define scalar output array
        if is_floating_dtype(x1.dtype):
            out_dtype = real_dtype(x1.dtype)
        else:
            out_dtype = float
        out = cupy.empty((), dtype=out_dtype)

        # Run reduction kernel (returns the output)
        if self.exponent == 0:
            return float(dist0(x1.data, x2.data, out))
        elif self.exponent == 1:
            return float(dist1w(x1.data, x2.data, self.array, out))
        elif self.exponent == 2:
            return float(dist2w(x1.data, x2.data, self.array, out))
        elif self.exponent == float('inf'):
            return float(distinf(x1.data, x2.data, out))
        elif self.exponent == -float('inf'):
            return float(distneginf(x1.data, x2.data, out))
        else:
            return float(distpw(x1.data, x2.data, self.exponent, self.array,
                                out))

    # TODO: remove repr_part and __repr__ when cupy.ndarray.__array__
    # is implemented. See
    # https://github.com/cupy/cupy/issues/589
    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        maxsize_full_print = 2 * np.get_printoptions()['edgeitems']
        arr_str = array_str(cupy.asnumpy(self.array),
                            nprint=maxsize_full_print)
        optargs = [('weighting', arr_str, ''),
                   ('exponent', self.exponent, 2.0)]
        return signature_string([], optargs, sep=',\n',
                                mod=[[], ['!s', ':.4']])

    def __repr__(self):
        """Return ``repr(self)``."""
        maxsize_full_print = 2 * np.get_printoptions()['edgeitems']
        arr_str = array_str(cupy.asnumpy(self.array),
                            nprint=maxsize_full_print)
        posargs = [arr_str]
        optargs = [('exponent', self.exponent, 2.0)]
        inner_str = signature_string(posargs, optargs, sep=',\n',
                                     mod=['!s', ':.4'])
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))


class CupyTensorSpaceConstWeighting(ConstWeighting):

    """Constant weighting for `CupyTensorSpace`.

    See `ConstWeighting` for further details.
    """

    def __init__(self, constant, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        constant : positive float
            Weighting constant of the inner product.
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        super(CupyTensorSpaceConstWeighting, self).__init__(
            constant, impl='cupy', exponent=exponent)

    def inner(self, x1, x2):
        """Return the weighted inner product of two tensors.

        Parameters
        ----------
        x1, x2 : `CupyTensor`
            Tensors whose inner product is calculated.

        Returns
        -------
        inner : float or complex
            The inner product of the two provided tensors.
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))
        else:
            return x1.space.field.element(
                self.const * cupy.dot(x1.data, x2.data))

    def norm(self, x):
        """Return the constant-weighted norm of a tensor.

        Parameters
        ----------
        x1 : `CupyTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the tensor.
        """
        # Define scalar output array
        if is_floating_dtype(x.dtype):
            out_dtype = real_dtype(x.dtype)
        else:
            out_dtype = float
        out = cupy.empty((), dtype=out_dtype)

        # Run reduction kernel (returns the output)
        if self.exponent == 0:
            return float(nrm0(x.data, out))
        elif self.exponent == 1:
            return float(self.const * nrm1(x.data, out))
        elif self.exponent == 2:
            # We try to use cuBLAS nrm2
            try:
                incx = _flat_inc(x.data)
            except ValueError:
                use_cublas = False
            else:
                use_cublas = True

            if use_cublas:
                try:
                    nrm2_cublas = _cublas_func('nrm2', x.dtype)
                except (ValueError, AttributeError):
                    pass
                else:
                    with cupy.cuda.Device(x.device) as dev:
                        norm = nrm2_cublas(
                            dev.cublas_handle, x.size, x.data_ptr, incx)
                    return float(np.sqrt(self.const) * norm)

            # Cannot use cuBLAS, fall back to custom kernel
            return float(np.sqrt(self.const) * nrm2(x.data, out))
        elif self.exponent == float('inf'):
            return float(nrminf(x.data, out))
        elif self.exponent == -float('inf'):
            return float(nrmneginf(x.data, out))
        else:
            return float(self.const ** (1 / self.exponent) *
                         nrmp(x, self.exponent, out))

    def dist(self, x1, x2):
        """Return the weighted distance between two tensors.

        Parameters
        ----------
        x1, x2 : `CupyTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the tensors.
        """
        # Define scalar output array
        if is_floating_dtype(x1.dtype):
            out_dtype = real_dtype(x1.dtype)
        else:
            out_dtype = float
        out = cupy.empty((), dtype=out_dtype)

        # Run reduction kernel (returns the output)
        if self.exponent == 0:
            return float(dist0(x1.data, x2.data, out))
        elif self.exponent == 1:
            return float(self.const * dist1(x1.data, x2.data, out))
        elif self.exponent == 2:
            # cuBLAS nrm2(x1 - x2) would probably be faster, but would
            # require a copy, so we don't do it
            return float(np.sqrt(self.const) * dist2(x1.data, x2.data, out))
        elif self.exponent == float('inf'):
            return float(distinf(x1.data, x2.data, out))
        elif self.exponent == -float('inf'):
            return float(distneginf(x1.data, x2.data, out))
        else:
            return float(self.const ** (1 / self.exponent) *
                         distp(x1.data, x2.data, self.exponent, out))


class CupyTensorSpaceCustomInner(CustomInner):

    """Class for handling custom inner products in `CupyTensorSpace`."""

    def __init__(self, inner):
        """Initialize a new instance.

        Parameters
        ----------
        inner : callable
            The inner product implementation. It must accept two
            `CupyTensor` arguments, return an element from their space's
            field (real or complex number) and satisfy the following
            conditions for all vectors ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``
        """
        super(CupyTensorSpaceCustomInner, self).__init__(inner, impl='cupy')


class CupyTensorSpaceCustomNorm(CustomNorm):

    """Class for handling a user-specified norm in `CupyTensorSpace`.

    Note that this removes ``inner``.
    """

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : callable
            The norm implementation. It must accept an `CupyTensor`
            argument, return a float and satisfy the following
            conditions for all vectors ``x, y`` and scalars ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``
        """
        super(CupyTensorSpaceCustomNorm, self).__init__(norm, impl='cupy')


class CupyTensorSpaceCustomDist(CustomDist):

    """Class for handling a user-specified distance in `CupyTensorSpace`.

    Note that this removes ``inner`` and ``norm``.
    """

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : callable
            The distance function defining a metric on `CupyTensorSpace`.
            It must accept two `CupyTensor` arguments, return a float and
            fulfill the following mathematical conditions for any three
            vectors ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``
        """
        super(CupyTensorSpaceCustomDist, self).__init__(dist, impl='cupy')


if __name__ == '__main__':
    if CUPY_AVAILABLE:
        from odl.util.testutils import run_doctests
        run_doctests()
