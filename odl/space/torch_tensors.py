# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Implementation of tensor spaces using ``torch``."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

import numpy as np

from odl.set import RealNumbers
from odl.space.base_tensors import TensorSpace, Tensor
from odl.space.weighting import (
    Weighting, ArrayWeighting, ConstWeighting,
    CustomInner, CustomNorm, CustomDist)
from odl.util import (
    dtype_str, is_floating_dtype, signature_string)

try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True


__all__ = ('TorchTensorSpace',)


# --- Evil monkey-patching of torch --- #


def _tensor___array__(self, dtype=None):
    """Return a Numpy array from this tensor.

    Parameters
    ----------
    dtype :
        Specifier for the data type of the output array.

    Returns
    -------
    array : `numpy.ndarray`
    """
    if dtype is None:
        return self.cpu().numpy()
    else:
        return self.cpu().numpy().astype(dtype, copy=False)


def _tensor___array_wrap__(self, array):
    """Return a new tensor wrapping the ``array``.

    Parameters
    ----------
    array : `numpy.ndarray`
        Array to be wrapped.

    Returns
    -------
    wrapper : `Tensor`
        Tensor wrapping ``array``.
    """
    if array.ndim == 0:
        if array.dtype.kind == 'b':
            return bool(array)
        elif array.dtype.kind in ('i', 'u'):
            return int(array)
        elif array.dtype.kind == 'f':
            return float(array)
        elif array.dtype.kind == 'c':
            return complex(array)
        else:
            raise RuntimeError('bad array {!r}'.format(array))
    else:
        if array.dtype == bool:
            # Workaround, torch has no built-in bool tensor
            cls = torch.ByteTensor
            array = array.astype('uint8')
        else:
            cls = _tensor_cls(array.dtype, use_cuda=self.is_cuda)
        return cls(array)


def _tensor___getitem__(self, indices):
    """Implement indexing with lists."""
    if isinstance(indices, list) and not all(np.isscalar(i) for i in indices):
        # For list of lists or lists of slices, remove the "outer list".
        # This makes indexing like `x[[[0, 1], [1, 1]]]` or
        # x[[slice(None), slice(None, None, 2)]]` work like in Numpy.
        # Torch uses the syntax `x[[0, 1], [1, 1]]`.
        indices = tuple(indices)
    return super(type(self), self).__getitem__(indices)


def _tensor___setitem__(self, indices, values):
    """Implement assignment with Numpy arrays and sequences."""
    if isinstance(indices, list) and not all(np.isscalar(i) for i in indices):
        # For list of lists, lists of slices etc., remove the "outer list".
        # This makes things like `x[[list1, list2]] = values` or
        # x[[slice1, slice2]] = values` work like in Numpy.
        # Torch uses the syntax `x[list1, list2] = values`.
        indices = tuple(indices)

    if isinstance(values, np.ndarray):
        # Support assignment from Numpy array
        values = torch.from_numpy(values)
    elif not isinstance(values, torch._TensorBase) and not np.isscalar(values):
        # Wrap non-scalars and non-tensors (e.g. nested lists) in a tensor.
        # This is not perfect since broadcasting won't work, but better
        # than nothing.
        values = type(self)(values)

    def iterable(obj):
        try:
            iter(obj)
        except TypeError:
            return False
        else:
            return True

    if (isinstance(values, torch._TensorBase) and
            (indices in (slice(None), Ellipsis)) or
            (iterable(indices) and all(i in (slice(None), Ellipsis)
                                       for i in indices))):
        # In the special cases `x[:] = values`, `x[:, :] = values` etc,
        # or `x[...] = values` we can use copying to support broadcasting
        self.copy_(values)
    else:
        super(type(self), self).__setitem__(indices, values)


if TORCH_AVAILABLE:
    for cls in torch._tensor_classes:
        setattr(cls, '__array__', _tensor___array__)
        setattr(cls, '__array_wrap__', _tensor___array_wrap__)
        setattr(cls, '__getitem__', _tensor___getitem__)
        setattr(cls, '__setitem__', _tensor___setitem__)


# --- Auxiliary stuff --- #


_TORCH_NAME_FROM_NPY_DTYPE = {
    np.dtype('float16'): 'Half',
    np.dtype('float32'): 'Float',
    np.dtype('float64'): 'Double',
    np.dtype('int8'): 'Char',
    np.dtype('int16'): 'Short',
    np.dtype('int32'): 'Int',
    np.dtype('int64'): 'Long',
    np.dtype('uint8'): 'Byte',
}
_NPY_DTYPE_FROM_TORCH_NAME = {v: k
                              for k, v in _TORCH_NAME_FROM_NPY_DTYPE.items()}


def _storage_name(dtype):
    """Return the name of the torch storage class for a given Numpy dtype."""
    dtype, dtype_in = np.dtype(dtype), dtype
    try:
        return _TORCH_NAME_FROM_NPY_DTYPE[dtype] + 'Storage'
    except KeyError:
        raise ValueError('`dtype` {!r} not supported'.format(dtype_in))


def _tensor_name(dtype):
    """Return the name of the torch tensor class for a given Numpy dtype."""
    dtype, dtype_in = np.dtype(dtype), dtype
    try:
        return _TORCH_NAME_FROM_NPY_DTYPE[dtype] + 'Tensor'
    except KeyError:
        raise ValueError('`dtype` {!r} not supported'.format(dtype_in))


def _numpy_dtype(obj):
    """Return Numpy dtype of a given object, supporting torch tensors."""
    if hasattr(obj, 'dtype'):
        return obj.dtype
    elif isinstance(obj, torch._TensorBase):
        # type() returns the full qualified name, like
        # `torch.cuda.FloatTensor`. Take only the last part.
        tensor_name = obj.type().split('.')[-1]
        dt_name = tensor_name[:-6]  # Remove 'Tensor'
        try:
            return _NPY_DTYPE_FROM_TORCH_NAME[dt_name]
        except KeyError:
            raise TypeError('`torch_tensor` type {!r} not not understood'
                            ''.format(torch.type()))
    else:
        raise TypeError('object of type {} not supported'.format(type(obj)))


def _tensor_cls(dtype, use_cuda):
    """Return a CPU or CUDA tensor class for a given Numpy dtype."""
    name = _tensor_name(dtype)
    if use_cuda:
        return getattr(torch.cuda, name)
    else:
        return getattr(torch, name)


def _empty(shape, dtype='float32', pinned=False, gpu_id=None):
    """Return a merely allocated tensor.

    Parameters
    ----------
    shape : int or sequence of ints
        The desired shape of the tensor.
    dtype : optional
        Numpy dtype that the tensor should use.
    pinned : bool, optional
        If ``True``, use pinned CPU memory.
        Cannot be combined with ``gpu_id``.
    gpu_id : int, optional
        Create a CUDA tensor on the device with this ID. For ``None``,
        CPU memory is used.
        Cannot be combined with ``pinned``.

    Returns
    -------
    empty : `torch.tensor._TensorBase`
        The newly allocated tensor.
    """
    try:
        iter(shape)
    except TypeError:
        shape = (int(shape),)
    else:
        shape = tuple(int(n) for n in shape)

    if gpu_id is None and pinned:
        # Currently no method for direct allocation of pinned memory exists,
        # see https://github.com/pytorch/pytorch/issues/2206
        # TODO: remove the workaround once the issue is closed
        size = int(np.prod(shape))
        stor_cls = getattr(torch, _storage_name(dtype))
        storage = stor_cls(size, allocator=torch.cuda._host_allocator())
        tens_cls = _tensor_cls(dtype, use_cuda=False)
        tens = tens_cls(storage)
        tens.resize_(*shape)

    elif gpu_id is None and not pinned:
        tens_cls = _tensor_cls(dtype, use_cuda=False)
        tens = tens_cls(*shape)

    elif gpu_id is not None and not pinned:
        tens_cls = _tensor_cls(dtype, use_cuda=True)
        tens = tens_cls(*shape, device=gpu_id)

    else:
        raise ValueError('cannot use both `pinned` and `gpu_id`')

    return tens


def _is_in_target_memory(tens, data_loc, gpu_id):
    if data_loc == 'CPU':
        return not tens.is_cuda
    elif data_loc == 'GPU':
        try:
            device = tens.get_device()
        except AttributeError:
            return False
        else:
            return device == gpu_id
    else:
        raise ValueError('`data_loc` {!r} not understood'.format(data_loc))


def _ravel(tensor):
    """Flatten tensor as view if it is contiguous, otherwise make flat copy."""
    if tensor.is_contiguous():
        return tensor.view(-1)
    else:
        return tensor.contiguous().view(-1)


# --- Space method implementations --- #


def _lincomb_impl(a, x1, b, x2, out):
    """Linear combination implementation, assuming types have been checked.

    This implementation is a highly optimized, considering all special
    cases of array alignment and special scalar values 0 and 1 separately.
    """
    if a == 0 and b == 0:
        # out <- 0
        out.data.fill_(0)

    elif a == 0:
        # Compute out <- b * x2
        if out is x2:
            # out <- b * out
            if b == 1:
                pass
            else:
                torch.mul(out.data, b, out=out.data)
        else:
            # out <- b * x2
            if b == 1:
                out.data.copy_(x2.data)
            else:
                torch.mul(x2.data, b, out=out.data)

    elif b == 0:
        # Compute out <- a * x1
        if out is x1:
            # out <- a * out
            if a == 1:
                pass
            else:
                torch.mul(out.data, a, out=out.data)
        else:
            # out <- a * x1
            if a == 1:
                out.data.copy_(x1.data)
            else:
                torch.mul(x1.data, a, out=out.data)

    else:
        # Compute out <- a * x1 + b * x2
        # Optimize a number of alignment options. We know that a and b
        # are nonzero.
        if out is x1 and out is x2:
            # out <-- (a + b) * out
            if a + b == 0:
                out.data.fill_(0)
            elif a + b == 1:
                pass
            else:
                torch.mul(out.data, a + b, out=out.data)
        elif out is x1:
            # out <-- a * out + b * x2
            if a == 1:
                # out <-- out + b * x2
                torch.add(out.data, b, x2.data, out=out.data)
            elif b == 1:
                # out <-- a * out + x2
                torch.add(x2.data, a, out.data, out=out.data)
            else:
                # out <-- a * out + b * x2
                # Makes 1 copy
                torch.add(a * out.data, b, x2.data, out=out.data)
        elif out is x2:
            # out <-- a * x1 + b * out
            if a == 1:
                # out <-- x1 + b * out
                torch.add(x1.data, b, out.data, out=out.data)
            elif b == 1:
                # out <-- a * x1 + out
                torch.add(out.data, a, x1.data, out=out.data)
            else:
                # out <-- a * out + b * x2
                # Makes 1 copy
                torch.add(a * out.data, b, x2.data, out=out.data)
        else:
            # No alignment. Now optimize for some special cases of a and b
            # (a = 0 or b = 0 is already covered).
            if a == 1:
                # out <- x1 + b * x2
                torch.add(x1.data, b, x2.data, out=out.data)
            elif b == 1:
                # out <- a * x1 + x2
                torch.add(x2.data, a, x1.data, out=out.data)
            else:
                # out <- a * x1 + b * x2
                # Makes 1 copy
                torch.add(a * x1.data, b, x2.data, out=out.data)


# --- Space and element classes --- #


class TorchTensorSpace(TensorSpace):

    """Tensor space implemented with Torch tensors.

    This space implements tensors of arbitrary rank over a `Field` ``F``,
    which is either the real or complex numbers.

    Its elements are represented as instances of the
    `TorchTensor` class.
    """

    def __init__(self, shape, dtype='float64', data_loc='CPU', **kwargs):
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
        data_loc : str, optional
            Memory location of elements in this space. The following values
            are possible:

            - ``'CPU'`` : Main `virtual memory`_ on the host.
            - ``'CPU_PINNED'`` : `Pinned (page-locked) memory`_ on the host.
              This is useful for faster and/or asynchronous CPU<->GPU data
              transfer. Since allocating pinned memory is more expensive,
              the break-even point is usually around 100 MB of data.
              See also `this analysis`_.
            - ``'GPU'`` : Device memory on the default GPU.
            - ``'GPUn'`` : Device memory on the GPU with ID ``n``, e.g.,
              ``'GPU0'``.

            .. note ::
                Pinned and virtual CPU memory are compatible and result
                in tensor spaces that will not be distinguished. GPU
                memory is incompatible with those and with GPU memory
                on a different device, so the resulting spaces will be
                considered different.

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
            It must accept two `TorchTensor` arguments and
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
            `TorchTensor` argument, return a float and satisfy the
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
            `TorchTensor` arguments, return a element from
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

        >>> space = TorchTensorSpace(3, 'float32')
        >>> space
        rn(3, impl='torch')
        >>> space.shape
        (3,)
        >>> space.dtype
        dtype('float32')

        A more convenient way is to use the factory functions with the
        ``imp='torch'`` option:

        >>> space = odl.rn(3, impl='torch', weighting=[1, 2, 3])
        >>> space
        rn(3, impl='torch', weighting=[ 1.,  2.,  3.])
        >>> space = odl.tensor_space((2, 3), impl='torch', dtype=int)
        >>> space
        tensor_space((2, 3), 'int', impl='torch')

        References
        ----------
        .. _virtual memory: https://en.wikipedia.org/wiki/Virtual_memory
        .. _Pinned (page-locked) memory:
           https://en.wikipedia.org/wiki/CUDA_Pinned_memory
        .. _this analysis:
           https://www.cs.virginia.edu/~mwb7w/cuda_support/pinned_tradeoff.html
        """
        super(TorchTensorSpace, self).__init__(shape, dtype)
        if self.dtype.char not in self.available_dtypes():
            raise ValueError('`dtype` {!r} not supported'.format(dtype))

        data_loc, data_loc_in = str(data_loc).upper(), data_loc
        if data_loc.startswith('CPU'):
            self.__is_pinned = data_loc[3:].endswith('PINNED')
            self.__gpu_id = None
            self.__data_loc = data_loc[:3]
        elif data_loc.startswith('GPU'):
            gpu_id = data_loc[3:]
            if not gpu_id:
                self.__gpu_id = 0
            else:
                self.__gpu_id = int(gpu_id)
            self.__is_pinned = False
            self.__data_loc = data_loc[:3]
        else:
            raise ValueError('`data_loc` {!r} not understood'
                             ''.format(data_loc_in))

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
                if weighting.impl != 'torch':
                    raise ValueError("`weighting.impl` must be 'torch', "
                                     '`got {!r}'.format(weighting.impl))
                if weighting.exponent != exponent:
                    raise ValueError('`weighting.exponent` conflicts with '
                                     '`exponent`: {} != {}'
                                     ''.format(weighting.exponent, exponent))
                self.__weighting = weighting
            else:
                self.__weighting = _weighting(weighting, self.dtype, exponent)

            # Check (afterwards) that the weighting input was sane
            if isinstance(self.weighting, TorchTensorSpaceArrayWeighting):
                if not np.can_cast(_numpy_dtype(self.weighting.array),
                                   self.dtype):
                    raise ValueError(
                        'cannot cast from `weighting` data type {} to '
                        'the space `dtype` {}'
                        ''.format(
                            dtype_str(_numpy_dtype(self.weighting.array)),
                            dtype_str(self.dtype)))

                if self.weighting.array.shape != self.shape:
                    raise ValueError('array-like weights must have same '
                                     'shape {} as this space, got {}'
                                     ''.format(self.shape,
                                               self.weighting.array.shape))

        elif dist is not None:
            self.__weighting = TorchTensorSpaceCustomDist(dist)
        elif norm is not None:
            self.__weighting = TorchTensorSpaceCustomNorm(norm)
        elif inner is not None:
            self.__weighting = TorchTensorSpaceCustomInner(inner)
        else:  # all None -> no weighing
            self.__weighting = TorchTensorSpaceConstWeighting(1.0, exponent)

    @property
    def impl(self):
        """Implementation back-end of this space: ``'torch'``."""
        return 'torch'

    @property
    def default_order(self):
        """Default storage order for new elements in this space: ``'C'``.

        In fact, only C ordering is supported by torch.
        """
        return 'C'

    @property
    def data_loc(self):
        """Memory location of elements in this space."""
        return self.__data_loc

    @property
    def is_pinned(self):
        """``True`` if pinned CPU memory is used for data, else ``False``."""
        return self.__is_pinned

    @property
    def gpu_id(self):
        """ID of the GPU used for elements in this space.

        If not applicable, i.e., no GPU is used, this attribute is ``None``.
        """
        return self.__gpu_id

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.__weighting

    @property
    def is_weighted(self):
        """Return ``True`` if the space is not weighted by constant 1.0."""
        return not (
            isinstance(self.weighting, TorchTensorSpaceConstWeighting) and
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
            The new element will be C-contiguous.

            Otherwise, a copy is avoided whenever possible. This requires
            correct `shape` and `dtype`, and if ``order='C'`` is provided,
            also contiguousness in that ordering. If any of these
            conditions is not met, a copy is made.

        order : {None, 'C'}, optional
            Storage order of the returned element. For ``'C'``,
            contiguous memory in C (row-major) ordering is enforced.
            The default ``None`` enforces no contiguousness.

        Returns
        -------
        element : `TorchTensor`
            The new element created (from ``inp``).

        Notes
        -----
        This method preserves "array views" of correct size and type,
        see the examples below.

        Examples
        --------
        >>> space = odl.rn((2, 3), impl='torch')

        Create an empty element:

        >>> empty = space.element()
        >>> empty.shape
        (2, 3)

        Initialization during creation:

        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x
        rn((2, 3), impl='torch').element(
            [[ 1.,  2.,  3.],
             [ 4.,  5.,  6.]]
        )
        """
        # --- Handle parameters --- #

        order_in = order
        if order is not None:
            order = str(order).upper()

        if order == 'F':
            raise ValueError("'F' ordering not supported")
        elif order is not None and order != 'C':
            raise ValueError("`order` {!r} not understood".format(order_in))

        tens_cls = _tensor_cls(self.dtype, use_cuda=(self.data_loc == 'GPU'))

        # --- Make element --- #

        if inp is None:
            return self.element_type(
                self,
                _empty(self.shape, self.dtype, self.is_pinned, self.gpu_id))
        else:
            # Optimize for a few cases that don't require work

            # Space element, no ordering enforced -> just return it
            if inp in self and order is None:
                return inp

            # Arguments for the constructor
            if self.gpu_id is None:
                constr_kwargs = {}
            else:
                constr_kwargs = {'device', self.gpu_id}

            if isinstance(inp, tens_cls):
                # Correct tensor class, shape and contiguousness (if required),
                # and in the right memory type -> wrap it
                if inp.shape != self.shape:
                    raise ValueError(
                        'expected `inp` of shape {}, got shape {}'
                        ''.format(self.shape, inp.shape))
                if (_is_in_target_memory(inp, self.data_loc, self.gpu_id) and
                        (order is None or inp.is_contiguous())):
                    return self.element_type(self, inp)

            elif isinstance(inp, np.ndarray):
                # Numpy array, create tensor first, using the ndarray
                # memory if possible
                if inp.shape != self.shape:
                    raise ValueError(
                        'expected `inp` of shape {}, got shape {}'
                        ''.format(self.shape, inp.shape))

                if self.dtype == 'float16':
                    # `from_numpy` not supported for float16, need to
                    # go through a tensor with float32 dtype
                    inp = inp.astype('float32', copy=False)
                    conv_cls = _tensor_cls(
                        'float32', use_cuda=(self.data_loc == 'GPU'))
                    tens = conv_cls(inp, **constr_kwargs)
                    tens = tens.type(tens_cls)

                elif order is None:
                    # TODO: this seems to be really slow sometimes,
                    # make benchmarks!
                    tens = torch.from_numpy(inp)

                else:
                    tens = tens_cls(inp)

                if self.is_pinned:
                    tens = tens.pin_memory()

                return self.element_type(self, tens)

            else:
                # Call the class constructor on the input

                # float16 needs some special care, direct construction of
                # tensors is barely working. We go through a float32 tensor
                # to avoid those issues
                if self.dtype == 'float16':
                    conv_cls = _tensor_cls(
                        'float32', use_cuda=(self.data_loc == 'GPU'))
                else:
                    conv_cls = tens_cls

                if np.isscalar(inp):
                    tens = conv_cls([inp], **constr_kwargs)
                else:
                    tens = conv_cls(inp, **constr_kwargs)

                if self.dtype == 'float16':
                    tens = tens.type(tens_cls)

                if self.is_pinned:
                    tens = tens.pin_memory()

                return self.element_type(self, tens)

    def zero(self):
        """Create a tensor filled with zeros.

        Examples
        --------
        >>> space = odl.rn(3, impl='torch')
        >>> x = space.zero()
        >>> x
        rn(3, impl='torch').element([ 0.,  0.,  0.])
        """
        empty = _empty(self.shape, self.dtype, self.is_pinned, self.gpu_id)
        return self.element(empty.fill_(0))

    def one(self, gpu=False):
        """Create a tensor filled with ones.

        Examples
        --------
        >>> space = odl.rn(3, impl='torch')
        >>> x = space.one()
        >>> x
        rn(3, impl='torch').element([ 1.,  1.,  1.])
        """
        empty = _empty(self.shape, self.dtype, self.is_pinned, self.gpu_id)
        return self.element(empty.fill_(1))

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is an instance of this space's type
            with the same `shape`,  `dtype`, `context` and
            `weighting`, ``False`` otherwise.

        Examples
        --------
        >>> space = odl.rn(2, impl='torch')
        >>> same_space = odl.rn(2, exponent=2, impl='torch')
        >>> same_space == space
        True

        Different `shape`, `exponent`, `dtype` or `impl`
        all result in different spaces:

        >>> diff_space = odl.rn((2, 3), impl='torch')
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(2, exponent=1, impl='torch')
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(2, dtype='float64', impl='torch')
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(2, impl='numpy')
        >>> diff_space == space
        False
        >>> space == object
        False

        A `TorchTensorSpace` with the same properties is considered
        equal:

        >>> same_space = odl.TorchTensorSpace(2, dtype='float32')
        >>> same_space == space
        True
        """
        return (super().__eq__(other) and
                self.data_loc == other.data_loc and
                self.weighting == other.weighting)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super().__hash__(), self.data_loc, self.gpu_id,
                     self.weighting))

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination of ``x1`` and ``x2``.

        Calculate ``out = a*x1 + b*x2`` using optimized BLAS
        routines if possible.

        Parameters
        ----------
        a, b : `TensorSpace.field` elements
            Scalars to multiply ``x1`` and ``x2`` with.
        x1, x2 : `TorchTensor`
            Summands in the linear combination.
        out : `TorchTensor`
            Tensor to which the result is written.

        Returns
        -------
        None

        Examples
        --------
        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 5, 6])
        >>> out = r3.element()
        >>> result = r3.lincomb(2, x, -1, y, out)
        >>> result
        rn(3, impl='torch').element([-2., -1.,  0.])
        >>> result is out
        True
        """
        _lincomb_impl(a, x1, b, x2, out)

    def _dist(self, x1, x2):
        """Calculate the distance between two tensors.

        Parameters
        ----------
        x1, x2 : `TorchTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            Distance between the tensors.

        Examples
        --------
        The default case is the Euclidean distance:

        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 2, -1])
        >>> r3.dist(x, y)  # 3^2 + 4^2 = 25
        5.0

        Taking a different exponent or a weighting is also possible
        during space creation:

        >>> r3 = odl.rn(3, impl='torch', exponent=1)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 2, -1])
        >>> r3.dist(x, y)  # 3 + 4 = 7
        7.0

        >>> r3 = odl.rn(3, impl='torch', weighting=2, exponent=1)
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
        x : `TorchTensor`
            The tensor whose norm is calculated.

        Returns
        -------
        norm : float
            Norm of the tensor.

        Examples
        --------
        The default case is the Euclidean norm:

        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([3, 4, 0])
        >>> r3.norm(x)  # 3^2 + 4^2 = 25
        5.0

        Taking a different exponent or a weighting is also possible
        during space creation:

        >>> r3 = odl.rn(3, impl='torch', exponent=1)
        >>> x = r3.element([3, 4, 0])
        >>> r3.norm(x)  # 3 + 4 = 7
        7.0

        >>> r3 = odl.rn(3, impl='torch', weighting=2, exponent=1)
        >>> x = r3.element([3, 4, 0])
        >>> r3.norm(x)  # 2*3 + 2*4 = 14
        14.0
        """
        return self.weighting.norm(x)

    def _inner(self, x1, x2):
        """Raw inner product of two tensors.

        Parameters
        ----------
        x1, x2 : `TorchTensor`
            The tensors whose inner product is calculated.

        Returns
        -------
        inner : `field` element
            Inner product of the tensors.

        Examples
        --------
        The default case is the dot product:

        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 0, 1])
        >>> r3.inner(x, y)  # 1*(-1) + 2*0 + 3*1 = 2
        2.0

        Taking a different weighting is also possible during space
        creation:

        >>> r3 = odl.rn(3, impl='torch', weighting=2)
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
        x1, x2 : `TorchTensor`
            Factors in the product.
        out : `TorchTensor`
            Tensor to which the result is written.

        Examples
        --------
        Out-of-place evaluation:

        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 0, 1])
        >>> r3.multiply(x, y)
        rn(3, impl='torch').element([-1.,  0.,  3.])

        In-place:

        >>> out = r3.element()
        >>> result = r3.multiply(x, y, out=out)
        >>> result
        rn(3, impl='torch').element([-1.,  0.,  3.])
        >>> result is out
        True
        """
        x1.ufuncs.multiply(x2, out=out)

    def _divide(self, x1, x2, out):
        """Entry-wise division of two tensors, assigned to out.

        Parameters
        ----------
        x1, x2 : `TorchTensor`
            Dividend and divisor in the quotient.
        out : `TorchTensor`
            Tensor to which the result is written.

        Examples
        --------
        Out-of-place evaluation:

        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 2, 1])
        >>> r3.divide(x, y)
        rn(3, impl='torch').element([-1.,  1.,  3.])

        In-place:

        >>> out = r3.element()
        >>> result = r3.divide(x, y, out=out)
        >>> result
        rn(3, impl='torch').element([-1.,  1.,  3.])
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

        optargs = [('impl', self.impl, '')]  # for the helper functions
        if self.is_pinned:
            optargs.append(('data_loc', 'CPU_PINNED', 'CPU'))
        elif self.gpu_id is not None:
            optargs.append(('data_loc', 'GPU{}'.format(self.gpu_id), 'CPU'))

        inner_str = signature_string(posargs, optargs)
        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str

        return '{}({})'.format(constructor_name, inner_str)

    @property
    def element_type(self):
        """`TorchTensor`"""
        return TorchTensor

    @staticmethod
    def available_dtypes():
        """Return the data types available for this space."""
        dtypes = ['float16', 'float32', 'float64',
                  'int8', 'int16', 'int32', 'int64',
                  'uint8']
        return tuple(np.dtype(dt) for dt in dtypes)

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

            ``RealNumbers()`` : ``np.dtype('float32')``

            ``ComplexNumbers()`` : not supported
        """
        if field is None or field == RealNumbers():
            return np.dtype('float32')
        else:
            raise ValueError('no default data type defined for field {}.'
                             ''.format(field))


class TorchTensor(Tensor):

    """Representation of an `TorchTensorSpace` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        super(TorchTensor, self).__init__(space)
        self.__data = data

    @property
    def data(self):
        """Raw `torch.tensor._TensorBase` object representing the data."""
        return self.__data

    @property
    def ndim(self):
        """Number of axes (=dimensions) of this tensor."""
        return self.space.ndim

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

        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([1, 2, 3])
        >>> x.asarray()
        array([ 1.,  2.,  3.], dtype=float32)
        >>> int_spc = odl.tensor_space(3, impl='torch', dtype=int)
        >>> x = int_spc.element([1, 2, 3])
        >>> x.asarray()
        array([1, 2, 3])
        >>> tensors = odl.rn((2, 3), impl='torch', dtype='float32')
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
            if self.dtype == 'float16':
                # numpy() not implemented, need workaround
                self_as_float = self.data.type(torch.FloatTensor)
                return self_as_float.numpy().astype('float16')
            else:
                return self.data.cpu().numpy()
        else:
            # This variant is about 30 % slower on CPU but uses less memory
            # than the alternative `out[:] = self.data.cpu().numpy()`.
            # For GPU data, this is about twice as fast.
            tmp = torch.from_numpy(out)
            tmp.copy_(self.data)
            return out

    @property
    def data_ptr(self):
        """A raw pointer to the data container.

        Examples
        --------
        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.one()
        >>> x.data_ptr  # doctest: +SKIP
        47259975936
        """
        return self.data.data_ptr()

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
        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([1, 2, 3])
        >>> same_x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, -2, -3])
        >>> x == same_x
        True
        >>> x == y
        False

        Space membership matters:

        >>> int_spc = odl.tensor_space(3, impl='torch', dtype=int)
        >>> x_int = int_spc.element([1, 2, 3])
        >>> x == x_int
        False
        """
        if other is self:
            return True
        elif other not in self.space:
            return False
        else:
            return torch.equal(self.data, other.data)

    def copy(self, async=False):
        """Create an identical (deep) copy of this tensor.

        Parameters
        ----------
        async : bool, optional
            Use asynchronous CPU<->GPU copies. See
            `the pytorch documentation
            <http://pytorch.org/docs/master/notes/cuda.html\
#use-pinned-memory-buffers>`_ for more deteils.

        Returns
        -------
        copy : `TorchTensor`
            A deep copy.

        Examples
        --------
        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([1, 2, 3])
        >>> y = x.copy()
        >>> y
        rn(3, impl='torch').element([ 1.,  2.,  3.])
        >>> x == y
        True
        >>> x is y
        False
        """
        new_elem = self.space.element()
        new_elem.data.copy_(self.data)
        return new_elem

    def __getitem__(self, indices):
        """Access values of this tensor.

        Parameters
        ----------
        indices : index expression
            The position(s) that should be accessed.

        Returns
        -------
        values : scalar or `pygpu._array.ndtorch`
            The value(s) at the index (indices).

        Examples
        --------
        Indexing rules follow roughly the Numpy style, as far (or "fancy")
        as supported:

        >>> r5 = odl.rn(5, impl='torch')
        >>> x = r5.element([1, 2, 3, 4, 5])
        >>> x[1:4]
        rn(3, impl='torch').element([ 2.,  3.,  4.])
        >>> x[::2]
        rn(3, impl='torch').element([ 1.,  3.,  5.])

        The returned views are writable, so modificatons alter the
        original array:

        >>> view = x[1:4]
        >>> view[:] = -1
        >>> view
        rn(3, impl='torch').element([-1., -1., -1.])
        >>> x
        rn(5, impl='torch').element([ 1., -1., -1., -1.,  5.])

        Multi-indexing is also directly supported:

        >>> tensors = odl.rn((2, 3), impl='torch')
        >>> x = tensors.element([[1, 2, 3],
        ...                      [4, 5, 6]])
        >>> x[1, 2]
        6.0
        >>> x[1]  # row with index 1
        rn(3, impl='torch').element([ 4.,  5.,  6.])
        >>> view = x[:, ::2]
        >>> view
        rn((2, 2), impl='torch').element(
            [[ 1.,  3.],
             [ 4.,  6.]]
        )
        >>> view[:] = [[0, 0],
        ...            [0, 0]]
        >>> x
        rn((2, 3), impl='torch').element(
            [[ 0.,  2.,  0.],
             [ 0.,  5.,  0.]]
        )
        """
        arr = self.data[indices]
        if np.isscalar(arr):
            return arr
        else:
            space = type(self.space)(arr.shape, dtype=self.dtype)
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

        >>> r5 = odl.rn(5, impl='torch')
        >>> x = r5.element([1, 2, 3, 4, 5])
        >>> x[1:4] = 0
        >>> x
        rn(5, impl='torch').element([ 1.,  0.,  0.,  0.,  5.])
        >>> x[1:4] = [-1, 1, -1]
        >>> x
        rn(5, impl='torch').element([ 1., -1.,  1., -1.,  5.])
        >>> y = r5.element([5, 5, 5, 8, 8])
        >>> x[:] = y
        >>> x
        rn(5, impl='torch').element([ 5.,  5.,  5.,  8.,  8.])

        In higher dimensions, broadcasting can be applied to assign
        values:

        >>> tensors = odl.rn((2, 3), impl='torch')
        >>> x = tensors.element([[1, 2, 3],
        ...                      [4, 5, 6]])
        >>> x[:] = [[6], [3]]  # rhs mimics (2, 1) shape
        >>> x
        rn((2, 3), impl='torch').element(
            [[ 6.,  6.,  6.],
             [ 3.,  3.,  3.]]
        )

        Be aware of unsafe casts and over-/underflows, there
        will be warnings at maximum.

        >>> int_r3 = odl.tensor_space(3, impl='torch', dtype='uint8')
        >>> x = int_r3.element([1, 2, 3])
        >>> x[0] = -1
        >>> x[0]
        255
        """
        if isinstance(values, TorchTensor):
            self.data[indices] = values.data
        else:
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

        .. note::
            This implementation looks for native ufuncs in ``pygpu.ufuncs``
            and falls back to the basic implementation with Numpy arrays
            in case no native ufunc is available. That fallback version
            comes with significant overhead due to data copies between
            host and device.

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
        ufunc_result : `TorchTensor`, `numpy.ndarray` or tuple
            Result of the ufunc evaluation. If no ``out`` keyword argument
            was given, the result is a `Tensor` or a tuple
            of such, depending on the number of outputs of ``ufunc``.
            If ``out`` was provided, the returned object or tuple entries
            refer(s) to ``out``.

        Examples
        --------
        We apply `numpy.add` to ODL tensors:

        >>> r3 = odl.rn(3, impl='torch')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, -2, -3])
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        rn(3, impl='torch').element([ 0.,  0.,  0.])
        >>> np.add(x, y)  # same mechanism for Numpy >= 1.13
        rn(3, impl='torch').element([ 0.,  0.,  0.])

        As ``out``, a Numpy array or an ODL tensor can be given (wrapped
        in a sequence):

        >>> out = r3.element()
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out,))
        >>> out
        rn(3, impl='torch').element([ 0.,  0.,  0.])
        >>> res is out
        True
        >>> out_arr = np.empty(3)
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out_arr,))
        >>> out_arr
        array([ 0.,  0.,  0.])
        >>> res is out_arr
        True

        With multiple dimensions:

        >>> r23 = odl.rn((2, 3), impl='torch')
        >>> x = y = r23.one()
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        rn((2, 3), impl='torch').element(
            [[ 2.,  2.,  2.],
             [ 2.,  2.,  2.]]
        )

        The ``ufunc.accumulate`` method retains the original `shape` and
        `dtype`. The latter can be changed with the ``dtype`` parameter:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'accumulate', x)
        rn(3, impl='torch').element([ 1.,  3.,  6.])
        >>> np.add.accumulate(x)  # same mechanism for Numpy >= 1.13
        rn(3, impl='torch').element([ 1.,  3.,  6.])

        For multi-dimensional tensors, an optional ``axis`` parameter
        can be provided:

        >>> z = r23.one()
        >>> z.__array_ufunc__(np.add, 'accumulate', z, axis=1)
        rn((2, 3), impl='torch').element(
            [[ 1.,  2.,  3.],
             [ 1.,  2.,  3.]]
        )

        The ``ufunc.at`` method operates in-place. Here we add the second
        operand ``[5, 10]`` to ``x`` at indices ``[0, 2]``:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'at', x, [0, 2], [5, 10])
        >>> x
        rn(3, impl='torch').element([  6.,   2.,  13.])

        For outer-product-type operations, i.e., operations where the result
        shape is the sum of the individual shapes, the ``ufunc.outer``
        method can be used:

        >>> x = odl.rn(2, impl='torch').element([0, 3])
        >>> y = odl.rn(3, impl='torch').element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'outer', x, y)
        rn((2, 3), impl='torch').element(
            [[ 1.,  2.,  3.],
             [ 4.,  5.,  6.]]
        )
        >>> y.__array_ufunc__(np.add, 'outer', y, x)
        rn((3, 2), impl='torch').element(
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
        rn(1, impl='torch').element([ 6.])

        In multiple dimensions, ``axis`` can be provided for reduction over
        selected axes:

        >>> z = r23.element([[1, 2, 3],
        ...                  [4, 5, 6]])
        >>> z.__array_ufunc__(np.add, 'reduce', z, axis=1)
        rn(2, impl='torch').element([  6.,  15.])

        Finally, ``add.reduceat`` is a combination of ``reduce`` and
        ``at`` with rather flexible and complex semantics (see the
        `reduceat documentation`_ for details):

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'reduceat', x, [0, 1])
        rn(2, impl='torch').element([ 1.,  5.])

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
        # --- Process `out` and perform checks --- #

        # Unwrap out if provided. The output parameters are all wrapped
        # in one tuple, even if there is only one.
        out_tuple = kwargs.pop('out', ())

        # Check number of `out` args, depending on `method`
        if method == '__call__' and len(out_tuple) not in (0, ufunc.nout):
            raise ValueError(
                "need 0 or {} `out` arguments for `method='__call__'` "
                'in {!r}, got {}'.format(ufunc.nout, ufunc, len(out_tuple)))
        elif method != '__call__' and len(out_tuple) not in (0, 1):
            raise ValueError(
                "need 0 or 1 `out` arguments for `method={!r}` in {!r}, "
                'got {}'.format(method, ufunc, len(out_tuple)))
        elif method == '__call__' and len(inputs) != ufunc.nin:
            arg_txt = 'argument' if ufunc.nin == 1 else 'arguments'
            raise ValueError(
                "need {} `input` {} for `method='__call__'` in {!r}, got {}"
                ''.format(ufunc.nin, arg_txt, ufunc, len(inputs)))

        # We allow our own tensors, the data container type and
        # `numpy.ndarray` objects as `out` (see docs for reason for the
        # latter)
        valid_types = (type(self), type(self.data), np.ndarray)
        if not all(isinstance(o, valid_types) or o is None
                   for o in out_tuple):
            return NotImplemented

        # Determine native ufunc vs. Numpy ufunc
        if (any(isinstance(o, np.ndarray) for o in out_tuple) or
                'order' in kwargs or 'dtype' in kwargs):
            native_ufunc = None
            use_native = False
        else:
            native_ufunc = getattr(torch, ufunc.__name__, None)
            use_native = (native_ufunc is not None)

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

        # Convert non-scalars and non-Torch tensors to elements
        if use_native:
            conv_inputs = []
            for inp in inputs:
                if isinstance(inp, torch._TensorBase) or np.isscalar(inp):
                    conv_inputs.append(inp)
                else:
                    conv_inputs.append(self.space.element(inp))

        # Pull out the data container of the inputs if necessary
        inputs = tuple(
            inp.data if isinstance(inp, type(self)) else inp
            for inp in inputs)

        # --- Get some parameters for later --- #

        # Arguments for space constructors
        exponent = self.space.exponent
        weighting = self.space.weighting

        # --- Evaluate ufunc --- #

        if method == '__call__':
            if ufunc.nout == 1:
                if use_native:
                    # Torch doesn't use out tuples
                    kwargs['out'] = out
                    res = native_ufunc(*inputs, **kwargs)
                else:
                    # Everything is cast to Numpy arrays by the parent method;
                    # the result can be a Numpy array or a tensor
                    kwargs['out'] = (out,)
                    res = super(TorchTensor, self).__array_ufunc__(
                        ufunc, '__call__', *inputs, **kwargs)

                # Wrap result if necessary (lazily)
                if out is None:
                    if is_floating_dtype(_numpy_dtype(res)):
                        # Weighting contains exponent
                        spc_kwargs = {'weighting': weighting}
                    else:
                        # No `exponent` or `weighting` applicable
                        spc_kwargs = {}
                    out_space = type(self.space)(
                        self.shape, _numpy_dtype(res), **spc_kwargs)
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
                    res1, res2 = super(TorchTensor, self).__array_ufunc__(
                        ufunc, '__call__', *inputs, **kwargs)

                # Wrap results if necessary (lazily)
                # We don't use exponents or weightings since we don't know
                # how to map them to the spaces
                if out1 is None:
                    res_space = type(self.space)(
                        self.shape, _numpy_dtype(res1))
                    result1 = res_space.element(res1)
                else:
                    result1 = out_tuple[0]

                if out2 is None:
                    res_space = type(self.space)(
                        self.shape, _numpy_dtype(res2), self.context)
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
                gpu_arr = inputs[0]
                npy_arr = np.asarray(gpu_arr)
                new_inputs = (npy_arr,) + inputs[1:]
                super(TorchTensor, self).__array_ufunc__(
                    ufunc, method, *new_inputs, **kwargs)
                gpu_arr[:] = npy_arr

            if use_native:
                # Native method could exist but raise `NotImplementedError`
                # or return `NotImplemented`, falling back to Numpy case
                # then, too
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
                # or return `NotImplemented`, falling back to base case
                # then, too
                try:
                    res = native_method(*inputs, **kwargs)
                except NotImplementedError:
                    res = super(TorchTensor, self).__array_ufunc__(
                        ufunc, method, *inputs, **kwargs)
                else:
                    if res is NotImplemented:
                        res = super(TorchTensor, self).__array_ufunc__(
                            ufunc, method, *inputs, **kwargs)

            else:
                res = super(TorchTensor, self).__array_ufunc__(
                    ufunc, method, *inputs, **kwargs)

            # Shortcut for scalar or no return value
            if np.isscalar(res) or res is None:
                # The first occurs for `reduce` with all axes,
                # the second for in-place stuff (`at` currently)
                return res

            # Wrap result if necessary (lazily)
            if out is None:
                if is_floating_dtype(_numpy_dtype(res)):
                    if res.shape != self.shape:
                        # Don't propagate weighting if shape changes
                        weighting = TorchTensorSpaceConstWeighting(1.0,
                                                                   exponent)
                    spc_kwargs = {'weighting': weighting}
                else:
                    spc_kwargs = {}

                res_space = type(self.space)(
                    res.shape, _numpy_dtype(res), **spc_kwargs)
                result = res_space.element(res)
            else:
                result = out_tuple[0]

            return result

    @property
    def ufuncs(self):
        """Access to NumPy style ufuncs.

        Examples
        --------
        >>> r2 = odl.rn(2, impl='torch')
        >>> x = r2.element([1, -2])
        >>> x.ufuncs.absolute()
        rn(2, impl='torch').element([ 1.,  2.])

        These functions can also be used with broadcasting or
        array-like input:
        >>> x.ufuncs.add(3)
        rn(2, impl='torch').element([ 4.,  1.])
        >>> x.ufuncs.subtract([3, 3])
        rn(2, 'float', impl='torch').element([-2., -5.])

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
        rn(2, impl='torch').element([ 4.,  2.])
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
        return super(TorchTensor, self).ufuncs

    @property
    def real(self):
        """Real part of this tensor.

        Returns
        -------
        real : `TorchTensor` view with real dtype
            The real part of this tensor as an element of an `rn` space.
        """
        # Only real dtypes currently
        return self

    @real.setter
    def real(self, newreal):
        """Setter for the real part.

        This method is invoked by ``tensor.real = other``.

        Parameters
        ----------
        newreal : `array-like` or scalar
            The new real part for this tensor.
        """
        self.real.data[:] = newreal

    @property
    def imag(self):
        """Imaginary part of this tensor.

        Returns
        -------
        imag : `TorchTensor`
            The imaginary part of this tensor as an element of an `rn` space.
        """
        # Only real dtypes currently
        return self.space.zero()

    @imag.setter
    def imag(self, newimag):
        """Setter for the imaginary part.

        This method is invoked by ``tensor.imag = other``.

        Parameters
        ----------
        newimag : `array-like` or scalar
            The new imaginary part for this tensor.
        """
        raise NotImplementedError('complex dtypes not supported')

    def conj(self, out=None):
        """Complex conjugate of this tensor.

        Parameters
        ----------
        out : `TorchTensor`, optional
            Tensor to which the complex conjugate is written.
            Must be an element of this tensor's space.

        Returns
        -------
        out : `TorchTensor`
            The complex conjugate tensor. If ``out`` was provided,
            the returned object is a reference to it.
        """
        # Only real dtypes currently
        if out is None:
            return self.copy()
        else:
            self.assign(out)
            return out

    def __ipow__(self, other):
        """Return ``self **= other``."""
        try:
            if other == int(other):
                return super(TorchTensorSpace, self).__ipow__(other)
        except TypeError:
            pass

        self.ufuncs.power(self.data, other, out=self.data)
        return self


# --- Weightings --- #


def _weighting(weights, dtype, exponent):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weights):
        weighting = TorchTensorSpaceConstWeighting(weights, exponent=exponent)
    elif isinstance(weights, torch._TensorBase):
        weighting = TorchTensorSpaceArrayWeighting(weights, exponent=exponent)
    else:
        # TODO: sequence of 1D array-likes
        weights = _tensor_cls(dtype, use_cuda=False)(weights)
        weighting = TorchTensorSpaceArrayWeighting(weights, exponent=exponent)
    return weighting


class TorchTensorSpaceArrayWeighting(ArrayWeighting):

    """Array weighting for `TorchTensorSpace`.

    See `ArrayWeighting` for further details.
    """

    def __init__(self, array, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        array : `array-like`
            Weighting array of the inner product, norm and distance.
            Any type other than `torch.tensor._TensorBase` will be
            cast to `torch.FloatTensor`.
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        if isinstance(array, TorchTensor):
            array = array.data
        elif isinstance(array, torch._TensorBase):
            pass
        else:
            array = torch.FloatTensor(array)

        super(TorchTensorSpaceArrayWeighting, self).__init__(
            array, impl='torch', exponent=exponent)

    def inner(self, x1, x2):
        """Calculate the weighted inner product of two tensors.

        Parameters
        ----------
        x1, x2 : `TorchTensor`
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
            inner = torch.dot(_ravel(x1.data), _ravel(self.array * x2.data))
            if x1.space.field is not None:
                inner = x1.space.field.element(inner)
            return inner

    def norm(self, x):
        """Calculate the weighted norm of a tensor.

        Parameters
        ----------
        x : `TorchTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the provided tensor.
        """
        if self.exponent == float('inf'):
            return torch.max(torch.abs(self.array * x.data))
        elif self.exponent == -float('inf'):
            return torch.min(torch.abs(self.array * x.data))
        else:
            return torch.norm(
                x.data * torch.pow(self.array, 1 / self.exponent),
                self.exponent)


class TorchTensorSpaceConstWeighting(ConstWeighting):

    """Constant weighting for `TorchTensorSpace`.

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
        super(TorchTensorSpaceConstWeighting, self).__init__(
            constant, impl='torch', exponent=exponent)

    def inner(self, x1, x2):
        """Calculate the weighted inner product of two tensors.

        Parameters
        ----------
        x1, x2 : `TorchTensor`
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
            inner = self.const * torch.dot(_ravel(x1.data), _ravel(x2.data))
            if x1.space.field is not None:
                inner = x1.space.field.element(inner)
            return inner

    def norm(self, x):
        """Calculate the constant-weighted norm of a tensor.

        Parameters
        ----------
        x1 : `TorchTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the tensor.
        """
        if self.exponent == 0:
            return torch.norm(x.data, 0)
        elif self.exponent == 1:
            return self.const * torch.norm(x.data, 1)
        elif self.exponent == 2:
            return float(np.sqrt(self.const)) * torch.norm(x.data, 2)
        elif self.exponent == float('inf'):
            return self.const * torch.max(torch.abs(x.data))
        elif self.exponent == -float('inf'):
            return self.const * torch.min(torch.abs(x.data))
        else:
            return (self.const ** (1 / self.exponent) *
                    torch.norm(x.data, self.exponent))

    def dist(self, x1, x2):
        """Calculate the weighted distance between two tensors.

        Parameters
        ----------
        x1, x2 : `TorchTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the tensors.
        """
        if self.exponent == 0:
            return torch.dist(x1.data, x2.data, 0)
        elif self.exponent == 1:
            return self.const * torch.dist(x1.data, x2.data, 1)
        elif self.exponent == 2:
            return float(np.sqrt(self.const)) * torch.dist(x1.data, x2.data, 2)
        elif self.exponent == float('inf'):
            return self.const * torch.max(torch.abs(x1.data - x2.data))
        elif self.exponent == -float('inf'):
            return self.const * torch.min(torch.abs(x1.data - x2.data))
        else:
            return (self.const ** (1 / self.exponent) *
                    torch.dist(x1.data, x2.data, self.exponent))


class TorchTensorSpaceCustomInner(CustomInner):

    """Class for handling custom inner products in `TorchTensorSpace`."""

    def __init__(self, inner):
        """Initialize a new instance.

        Parameters
        ----------
        inner : callable
            The inner product implementation. It must accept two
            `TorchTensor` arguments, return an element from their space's
            field (real or complex number) and satisfy the following
            conditions for all vectors ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``
        """
        super(TorchTensorSpaceCustomInner, self).__init__(inner, impl='torch')


class TorchTensorSpaceCustomNorm(CustomNorm):

    """Class for handling a user-specified norm in `TorchTensorSpace`.

    Note that this removes ``inner``.
    """

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : callable
            The norm implementation. It must accept an `TorchTensor`
            argument, return a float and satisfy the following
            conditions for all vectors ``x, y`` and scalars ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``
        """
        super(TorchTensorSpaceCustomNorm, self).__init__(norm, impl='torch')


class TorchTensorSpaceCustomDist(CustomDist):

    """Class for handling a user-specified distance in `TorchTensorSpace`.

    Note that this removes ``inner`` and ``norm``.
    """

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : callable
            The distance function defining a metric on `TorchTensorSpace`.
            It must accept two `TorchTensor` arguments, return a float and
            fulfill the following mathematical conditions for any three
            vectors ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``
        """
        super(TorchTensorSpaceCustomDist, self).__init__(dist, impl='torch')


if __name__ == '__main__':
    if TORCH_AVAILABLE:
        from odl.util.testutils import run_doctests
        run_doctests()
