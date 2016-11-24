# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Implementation of tensor spaces using ``pygpu``."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import os
import numpy as np
import warnings

from odl.set import RealNumbers
from odl.space.base_tensors import TensorSpace, Tensor
from odl.space.weighting import (
    Weighting, ArrayWeighting, ConstWeighting,
    CustomInner, CustomNorm, CustomDist)
from odl.util import (
    dtype_str, is_int_dtype, is_floating_dtype, signature_string)

try:
    import pygpu
except ImportError:
    PYGPU_AVAILABLE = False
else:
    import pygpu.gpuarray as gpuary
    from pygpu import ndgpuarray
    PYGPU_AVAILABLE = True


__all__ = ('GpuTensorSpace',)


# TODO: does this make sense? If yes, document it!
CUDA_DEVICE_NAME = os.environ.get('ODL_CUDA_DEVICE', '')
CUDA_DEFAULT_DEVICE_NAME = 'cuda0'
OPENCL_DEVICE_NAME = os.environ.get('ODL_OPENCL_DEVICE', '')
OPENCL_DEFAULT_DEVICE_NAME = 'opencl0:0'


# --- GPU initialization functions --- #


def _init_gpu_context():
    """Set a default GPU context and return corresponding ``HAVE_*`` flags."""
    HAVE_CUDA_CONTEXT = HAVE_OPENCL_CONTEXT = False

    if PYGPU_AVAILABLE:
        # If a device name for exactly one backend is defined, initialize
        # and set as default a context for that backend without capturing
        # exceptions
        if CUDA_DEVICE_NAME and not OPENCL_DEVICE_NAME:
            CUDA_CONTEXT = pygpu.init(CUDA_DEVICE_NAME)
            pygpu.set_default_context(CUDA_CONTEXT)
            HAVE_CUDA_CONTEXT = True
        elif OPENCL_DEVICE_NAME and not CUDA_DEVICE_NAME:
            OPENCL_CONTEXT = pygpu.init(OPENCL_DEVICE_NAME)
            pygpu.set_default_context(OPENCL_CONTEXT)
            HAVE_OPENCL_CONTEXT = True

        # Otherwise, try CUDA first, and if it fails, try OpenCL. Use
        # specified device names if given, or the defaults.
        else:
            if CUDA_DEVICE_NAME and OPENCL_DEVICE_NAME:
                cuda_dev = CUDA_DEVICE_NAME
                opencl_dev = OPENCL_DEVICE_NAME
            elif not CUDA_DEVICE_NAME and not OPENCL_DEVICE_NAME:
                cuda_dev = CUDA_DEFAULT_DEVICE_NAME
                opencl_dev = OPENCL_DEFAULT_DEVICE_NAME

            try:
                CUDA_CONTEXT = pygpu.init(cuda_dev)
            except (pygpu.gpuarray.GpuArrayException, ValueError):
                # Failed to get CUDA context with the specified device,
                # try OpenCL
                try:
                    OPENCL_CONTEXT = pygpu.init(opencl_dev)
                except (pygpu.gpuarray.GpuArrayException, ValueError):
                    pass
                else:
                    pygpu.set_default_context(OPENCL_CONTEXT)
                    HAVE_OPENCL_CONTEXT = True
            else:
                pygpu.set_default_context(CUDA_CONTEXT)
                HAVE_CUDA_CONTEXT = True

        if not HAVE_CUDA_CONTEXT and not HAVE_OPENCL_CONTEXT:
            # Getting here without failure means that either both or none of
            # the devices were specified
            if CUDA_DEVICE_NAME and OPENCL_DEVICE_NAME:
                # TODO: better to fail in this case?
                warnings.warn(
                    'unable to create a GPU context - both user-defined '
                    'device names {!r} (for CUDA) and {!r} (for OpenCL) are '
                    'invalid'
                    ''.format(CUDA_DEVICE_NAME, OPENCL_DEVICE_NAME),
                    RuntimeWarning)
            else:
                warnings.warn(
                    'unable to create a GPU context - both default device '
                    'names {!r} (for CUDA) and {!r} (for OpenCL) are invalid'
                    ''.format(CUDA_DEFAULT_DEVICE_NAME,
                              OPENCL_DEFAULT_DEVICE_NAME),
                    RuntimeWarning)

    return HAVE_CUDA_CONTEXT, HAVE_OPENCL_CONTEXT


HAVE_CUDA_CONTEXT, HAVE_OPENCL_CONTEXT = _init_gpu_context()
HAVE_GPU_CONTEXT = HAVE_CUDA_CONTEXT or HAVE_OPENCL_CONTEXT



def precompile_kernels():
    """Precompile GPU kernels for linspace methods with most common dtypes."""
    a = 1.0
    b = -1.0
    for dtype in ('float32', 'float64'):
        x = pygpu.zeros(1, dtype=dtype)
        y = pygpu.zeros(1, dtype=dtype)
        out = x._empty_like_me()
        a_arg = pygpu.elemwise.as_argument(a, 'a', read=True)
        b_arg = pygpu.elemwise.as_argument(b, 'b', read=True)
        x_arg = pygpu.elemwise.as_argument(x, 'x', read=True)
        y_arg = pygpu.elemwise.as_argument(y, 'y', read=True)
        out_arg = pygpu.elemwise.as_argument(out, 'out', write=True)

        # scal
        args = [a_arg, x_arg, out_arg]
        oper = 'out = a * x'
        pygpu.elemwise.GpuElemwise(out.context, oper, args)

        # axpy
        args = [a_arg, x_arg, out_arg]
        oper = 'out = a * x + out'
        pygpu.elemwise.GpuElemwise(out.context, oper, args)

        # axpby
        args = [a_arg, x_arg, b_arg, out_arg]
        oper = 'out = a * x + b * out'
        pygpu.elemwise.GpuElemwise(out.context, oper, args)

        # lico
        args = [a_arg, x_arg, b_arg, y_arg, out_arg]
        oper = 'out = a * x + b * y'
        pygpu.elemwise.GpuElemwise(out.context, oper, args)

        # TODO: kernels for inner, norm and dist


# --- GPU kernels for linspace methods --- #


def scal(a, x, out):
    """Implement ``out <-- a * x`` using an elementwise GPU kernel.

    Parameters
    ----------
    a : scalar, `array-like` or `pygpu.gpuarray.GpuArray`
        Factor ``a`` in the scaling. If non-scalar, its shape must be
        broadcastable with ``x.shape``.
    x : `pgpu.gpuarray.GpuArray`
        Array ``x`` in the scaling.
    out : `pgpu.gpuarray.GpuArray`
        Output array. Its shape must be equal to the broadcast shape of
        ``a`` and ``x``.

    See Also
    --------
    numpy.broadcast : helper to calculate properties of broadcast objects
    """
    assert isinstance(x, gpuary.GpuArray)
    assert isinstance(out, gpuary.GpuArray)
    if not isinstance(a, gpuary.GpuArray) and not np.isscalar(a):
        if out.flags.f_contiguous and not out.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'
        a = gpuary.array(a, dtype=out.dtype, order=order, context=x.context)

    a_arg = pygpu.elemwise.as_argument(a, 'a', read=True)
    x_arg = pygpu.elemwise.as_argument(x, 'x', read=True)
    out_arg = pygpu.elemwise.as_argument(out, 'out', write=True)
    args = [a_arg, x_arg, out_arg]

    oper = 'out = a * x'
    # TODO: check what to do with `convert_f16`
    kernel = pygpu.elemwise.GpuElemwise(out.context, oper, args,
                                        convert_f16=True)
    kernel(a, x, out, broadcast=True)


def axpy(a, x, out):
    """Implement ``out <-- a * x + out`` using an elementwise GPU kernel.

    Parameters
    ----------
    a : scalar, `array-like` or `pygpu.gpuarray.GpuArray`
        Factor ``a`` in the scaling. If non-scalar, its shape must be
        broadcastable with ``x.shape``.
    x : `pgpu.gpuarray.GpuArray`
        Array ``x`` in the scaling.
    out : `pgpu.gpuarray.GpuArray`
        Output array. Its shape must be equal to the broadcast shape of
        ``a``, ``x`` and ``out``.

    See Also
    --------
    numpy.broadcast : helper to calculate properties of broadcast objects
    """
    assert isinstance(x, gpuary.GpuArray)
    assert isinstance(out, gpuary.GpuArray)
    if not isinstance(a, gpuary.GpuArray) and not np.isscalar(a):
        if out.flags.f_contiguous and not out.flags.c_contiguous:
            a_order = 'F'
        else:
            a_order = 'C'
        a = gpuary.array(a, dtype=out.dtype, order=a_order,
                         context=out.context)

    a_arg = pygpu.elemwise.as_argument(a, 'a', read=True)
    x_arg = pygpu.elemwise.as_argument(x, 'x', read=True)
    out_arg = pygpu.elemwise.as_argument(out, 'out', read=True, write=True)
    args = [a_arg, x_arg, out_arg]

    oper = 'out = a * x + out'
    # TODO: check what to do with `convert_f16`
    kernel = pygpu.elemwise.GpuElemwise(out.context, oper, args,
                                        convert_f16=True)
    kernel(a, x, out, broadcast=True)


def axpby(a, x, b, out):
    """Implement ``out <-- a * x + b * out`` using an elementwise GPU kernel.

    Parameters
    ----------
    a : scalar, `array-like` or `pygpu.gpuarray.GpuArray`
        Factor ``a`` in the scaling. If non-scalar, its shape must be
        broadcastable with ``x.shape``.
    x : `pgpu.gpuarray.GpuArray`
        Array ``x`` in the scaling.
    b : scalar, `array-like` or `pygpu.gpuarray.GpuArray`
        Factor ``a`` in the scaling. If non-scalar, its shape must be
        broadcastable with ``out.shape``.
    out : `pgpu.gpuarray.GpuArray`
        Output array. Its shape must be equal to the broadcast shape of
        ``a``, ``x``, ``b`` and ``out``.

    See Also
    --------
    numpy.broadcast : helper to calculate properties of broadcast objects
    """
    assert isinstance(x, gpuary.GpuArray)
    assert isinstance(out, gpuary.GpuArray)
    if out.flags.f_contiguous and not out.flags.c_contiguous:
        out_order = 'F'
    else:
        out_order = 'C'
    if not isinstance(a, gpuary.GpuArray) and not np.isscalar(a):
        a = gpuary.array(a, dtype=out.dtype, order=out_order,
                         context=out.context)
    if not isinstance(b, gpuary.GpuArray) and not np.isscalar(b):
        b = gpuary.array(b, dtype=out.dtype, order=out_order,
                         context=out.context)

    a_arg = pygpu.elemwise.as_argument(a, 'a', read=True)
    x_arg = pygpu.elemwise.as_argument(x, 'x', read=True)
    b_arg = pygpu.elemwise.as_argument(b, 'b', read=True)
    out_arg = pygpu.elemwise.as_argument(out, 'out', read=True, write=True)
    args = [a_arg, x_arg, b_arg, out_arg]

    oper = 'out = a * x + b * out'
    # TODO: check what to do with `convert_f16`
    kernel = pygpu.elemwise.GpuElemwise(out.context, oper, args,
                                        convert_f16=True)
    kernel(a, x, b, out, broadcast=True)


def lico(a, x, b, y, out):
    """Implement ``out <-- a * x + b * y`` using an elementwise GPU kernel.

    Parameters
    ----------
    a : scalar, `array-like` or `pygpu.gpuarray.GpuArray`
        Factor ``a`` in the scaling. If non-scalar, its shape must be
        broadcastable with ``x.shape``.
    x : `pgpu.gpuarray.GpuArray`
        Array ``x`` in the scaling.
    b : scalar, `array-like` or `pygpu.gpuarray.GpuArray`
        Factor ``a`` in the scaling. If non-scalar, its shape must be
        broadcastable with ``y.shape``.
    y : `pgpu.gpuarray.GpuArray`
        Array ``y`` in the scaling.
    out : `pgpu.gpuarray.GpuArray`
        Output array. Its shape must be equal to the broadcast shape of
        ``a``, ``x``, ``b`` and ``out``.

    See Also
    --------
    numpy.broadcast : helper to calculate properties of broadcast objects
    """
    assert isinstance(x, gpuary.GpuArray)
    assert isinstance(y, gpuary.GpuArray)
    assert isinstance(out, gpuary.GpuArray)
    if out.flags.f_contiguous and not out.flags.c_contiguous:
        out_order = 'F'
    else:
        out_order = 'C'
    if not isinstance(a, gpuary.GpuArray) and not np.isscalar(a):
        a = gpuary.array(a, dtype=out.dtype, order=out_order,
                         context=out.context)
    if not isinstance(b, gpuary.GpuArray) and not np.isscalar(b):
        b = gpuary.array(b, dtype=out.dtype, order=out_order,
                         context=out.context)

    a_arg = pygpu.elemwise.as_argument(a, 'a', read=True)
    x_arg = pygpu.elemwise.as_argument(x, 'x', read=True)
    b_arg = pygpu.elemwise.as_argument(b, 'b', read=True)
    y_arg = pygpu.elemwise.as_argument(y, 'y', read=True)
    out_arg = pygpu.elemwise.as_argument(out, 'out', write=True)
    args = [a_arg, x_arg, b_arg, y_arg, out_arg]

    oper = 'out = a * x + b * y'
    # TODO: check what to do with `convert_f16`
    kernel = pygpu.elemwise.GpuElemwise(out.context, oper, args,
                                        convert_f16=True)
    kernel(a, x, b, y, out, broadcast=True)


# --- Space method implementations --- #


def _lincomb_impl(a, x1, b, x2, out):
    """Raw linear combination, assuming types have been checked."""
    if x1 is x2 and b != 0:
        # x1 is aligned with x2 =>  out <-- (a + b) * x1
        scal(a + b, x1.data, out.data)
    elif out is x1 and out is x2:
        # All the arrays are aligned =>  out <-- (a + b) * out
        scal(a + b, out.data, out.data)
    elif out is x1:
        # out is aligned with x1 =>  out <-- a * out + b * x2
        axpby(b, x2.data, a, out.data)
    elif out is x2:
        # out is aligned with x2 =>  out <-- a * x1 + b * out
        if a == 0:
            # Use in-place  out <-- b * out directly
            scal(b, out.data, out.data)
        elif b == 0:  # out <-- a * x1
            scal(a, x1.data, out.data)
        else:  # out <-- a * x1 + b * out
            axpby(a, x1.data, b, out.data)
    else:
        # We have exhausted all alignment options, so x1 != x2 != out
        # We now optimize for various values of a and b
        if b == 0:
            if a == 0:  # Zero assignment  out <-- 0
                out.data[:] = 0
            elif a == 1:  # Copy  out <-- x1
                out.data[:] = x1.data
            else:  # Scaled copy  out <-- a * x1
                scal(a, x1.data, out.data)
        else:
            if a == 0:
                if b == 1:
                    # Copy  out <-- x2
                    out.data[:] = x2.data
                else:
                    # Scaled copy  out <-- b * x2
                    scal(b, x2.data, out.data)
            else:  # General case  out <-- a * x1 + b * x2
                lico(a, x1.data, b, x2.data, out.data)


# --- Space and element classes --- #


class GpuTensorSpace(TensorSpace):

    """Tensor space implemented with GPU arrays.

    This space implements tensors of arbitrary rank over a `Field` ``F``,
    which is either the real or complex numbers.

    Its elements are represented as instances of the
    `GpuTensor` class.
    """

    def __init__(self, shape, dtype='float64', context=None, **kwargs):
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
        context : `pygpu.gpuarray.GpuContext`, optional
            GPU context for tensors in this space. If not specified,
            the context returned by `pygpu.get_default_context` is
            used. See Notes for further explanation.
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
            It must accept two `GpuTensor` arguments and
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
            `GpuTensor` argument, return a float and satisfy the
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
            `GpuTensor` arguments, return a element from
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

        >>> space = GpuTensorSpace(3, 'float')
        >>> space
        rn(3, impl='gpuarray')
        >>> space.shape
        (3,)
        >>> space.dtype
        dtype('float64')

        A more convenient way is to use the factory functions with the
        ``imp='gpuarray'`` option:

        >>> space = odl.rn(3, impl='gpuarray', weighting=[1, 2, 3])
        >>> space
        rn(3, impl='gpuarray', weighting=[1, 2, 3])
        >>> space = odl.tensor_space((2, 3), impl='gpuarray', dtype=int)
        >>> space
        tensor_space((2, 3), 'int', impl='gpuarray')

        Notes
        -----
        Most users will want to work with a single context that is
        set as default, and not care about it any more. Such
        a default context is tried to be created at module import time
        by testing some standard device configuration.
        If this fails for your GPU configuration, you need to manually
        set a default context or pass an existing context to the space
        constructor. Here are the steps you may need to take:

            1. Create a context::

                context = pygpu.init(device)

            The ``device`` argument is ``'cuda'`` or ``'cudaN'`` with a
            number N for CUDA, or ``openclM:N'`` with numbers M and N
            for OpenCL. See `this documentation section`_.

            2. Set it as default context::

                pygpu.set_default_context(context)

        References
        ----------
        .. _this documentation section:
           http://deeplearning.net/software/libgpuarray/pyapi.html\
#pygpu.gpuarray.init
        """
        super(GpuTensorSpace, self).__init__(shape, dtype)
        if self.dtype.char not in self.available_dtypes():
            raise ValueError('`dtype` {!r} not supported'.format(dtype))

        if context is None:
            self.__context = pygpu.get_default_context()
            if self.context is None:
                # TODO: find a way to define a default context at
                # import time
                raise TypeError('no default context defined, expected '
                                '`context` to be a `GpuContext` instance')
        else:
            if not isinstance(context, gpuary.GpuContext):
                raise TypeError('`context` must be a `GpuContext` instance, '
                                'got {!r}'.format(context))
            self.__context = context

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
                if weighting.impl != 'gpuarray':
                    raise ValueError("`weighting.impl` must be 'gpuarray', "
                                     '`got {!r}'.format(weighting.impl))
                if weighting.exponent != exponent:
                    raise ValueError('`weighting.exponent` conflicts with '
                                     '`exponent`: {} != {}'
                                     ''.format(weighting.exponent, exponent))
                self.__weighting = weighting
            else:
                self.__weighting = _weighting(weighting, exponent)

            # Check (afterwards) that the weighting input was sane
            if isinstance(self.weighting, GpuTensorSpaceArrayWeighting):
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
            self.__weighting = GpuTensorSpaceCustomDist(dist)
        elif norm is not None:
            self.__weighting = GpuTensorSpaceCustomNorm(norm)
        elif inner is not None:
            self.__weighting = GpuTensorSpaceCustomInner(inner)
        else:  # all None -> no weighing
            self.__weighting = GpuTensorSpaceConstWeighting(1.0, exponent)

    @property
    def context(self):
        """The GPU context of this tensor space."""
        return self.__context

    @property
    def impl(self):
        """Implementation back-end of this space: ``'gpuarray'``."""
        return 'gpuarray'

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
            isinstance(self.weighting, GpuTensorSpaceConstWeighting) and
            self.weighting.const == 1.0)

    @property
    def exponent(self):
        """Exponent of the norm and distance."""
        return self.weighting.exponent

    def element(self, inp=None, data_ptr=None, order=None):
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

        data_ptr : int, optional
            Pointer to the start memory address of a contiguous GPU array
            or an equivalent raw container with the same total number of
            bytes. For this option,
            ``order`` must be either ``'C'`` or ``'F'``.
            The option is also mutually exclusive with ``inp``.
        order : {None, 'C', 'F'}, optional
            Storage order of the returned element. For ``'C'`` and ``'F'``,
            contiguous memory in the respective ordering is enforced.
            The default ``None`` enforces no contiguousness.

        Returns
        -------
        element : `GpuTensor`
            The new element created (from ``inp``).

        Notes
        -----
        This method preserves "array views" of correct size and type,
        see the examples below.

        Examples
        --------
        >>> space = odl.rn((2, 3), impl='gpuarray')

        Create an empty element:

        >>> empty = space.element()
        >>> empty.shape
        (2, 3)

        Initialization during creation:

        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x
        rn((2, 3), impl='gpuarray').element(
            [[ 1.,  2.,  3.],
             [ 4.,  5.,  6.]]
        )
        """
        if order is not None and str(order).upper() not in ('C', 'F'):
            raise ValueError("`order` {!r} not understood".format(order))

        if inp is None and data_ptr is None:
            if order is None:
                arr = gpuary.empty(self.shape, dtype=self.dtype,
                                   order=self.order, cls=ndgpuarray,
                                   context=self.context)
            else:
                arr = gpuary.empty(self.shape, dtype=self.dtype,
                                   order=order, cls=ndgpuarray,
                                   context=self.context)

            return self.element_type(self, arr)

        elif inp is None and data_ptr is not None:
            if order is None:
                raise ValueError('`order` cannot be None for element '
                                 'creation from pointer')

            # TODO: make this work
            if str(order).upper() == 'C':
                strides = [self.dtype.itemsize *
                           int(np.prod(self.shape[self.ndim - i:]))
                           for i in range(self.ndim - 1, -1, -1)]
            else:
                strides = [self.dtype.itemsize *
                           int(np.prod(self.shape[:i]))
                           for i in range(self.ndim)]
            arr = gpuary.from_gpudata(
                data_ptr, 0, self.dtype, self.shape, strides=strides,
                cls=ndgpuarray)
            return self.element_type(self, arr)

        elif inp is not None and data_ptr is None:
            if inp in self and order is None:
                # Short-circuit for space elements and no enforced ordering
                return inp

            # Try to not copy but require dtype and order if given
            # (`order=None` is ok as `array` argument)
            arr = gpuary.array(inp, copy=False, dtype=self.dtype,
                               ndmin=self.ndim, order=order,
                               cls=ndgpuarray, context=self.context)

            # Make sure the result is writeable, if not make copy.
            # This happens for e.g. results of `np.broadcast_to()`.
            if not arr.flags.writeable:
                arr = arr.copy()
            if arr.shape != self.shape:
                raise ValueError('shape of `inp` not equal to space shape: '
                                 '{} != {}'.format(arr.shape, self.shape))
            return self.element_type(self, arr)

        else:
            raise TypeError('cannot provide both `inp` and `data_ptr`')

    def zero(self):
        """Create a tensor filled with zeros.

        Examples
        --------
        >>> space = odl.rn(3, impl='gpuarray')
        >>> x = space.zero()
        >>> x
        rn(3, impl='gpuarray').element([ 0.,  0.,  0.])
        """
        arr = gpuary.zeros(self.shape, dtype=self.dtype,
                           order=self.default_order,
                           context=self.context, cls=ndgpuarray)
        return self.element(arr)

    def one(self):
        """Create a tensor filled with ones.

        Examples
        --------
        >>> space = odl.rn(3, impl='gpuarray')
        >>> x = space.one()
        >>> x
        rn(3, impl='gpuarray').element([ 1.,  1.,  1.])
        """
        arr = gpuary.empty(self.shape, dtype=self.dtype,
                           order=self.default_order,
                           context=self.context, cls=ndgpuarray)
        arr[:] = 1
        return self.element(arr)

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
        >>> space = odl.rn(2, impl='gpuarray')
        >>> same_space = odl.rn(2, exponent=2, impl='gpuarray')
        >>> same_space == space
        True

        Different `shape`, `exponent`, `dtype` or `impl`
        all result in different spaces:

        >>> diff_space = odl.rn((2, 3), impl='gpuarray')
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(2, exponent=1, impl='gpuarray')
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(2, dtype='float32', impl='gpuarray')
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(2, impl='numpy')
        >>> diff_space == space
        False
        >>> space == object
        False

        A `GpuTensorSpace` with the same properties is considered
        equal:

        >>> same_space = odl.GpuTensorSpace(2, dtype='float64')
        >>> same_space == space
        True
        """
        return (super().__eq__(other) and
                self.context == other.context and
                self.weighting == other.weighting)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super().__hash__(), self.context, self.weighting))

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination of ``x1`` and ``x2``.

        Calculate ``out = a*x1 + b*x2`` using optimized BLAS
        routines if possible.

        Parameters
        ----------
        a, b : `TensorSpace.field` elements
            Scalars to multiply ``x1`` and ``x2`` with.
        x1, x2 : `GpuTensor`
            Summands in the linear combination.
        out : `GpuTensor`
            Tensor to which the result is written.

        Returns
        -------
        None

        Examples
        --------
        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 5, 6])
        >>> out = r3.element()
        >>> result = r3.lincomb(2, x, -1, y, out)
        >>> result
        rn(3, impl='gpuarray').element([-2., -1.,  0.])
        >>> result is out
        True
        """
        _lincomb_impl(a, x1, b, x2, out)

    def _dist(self, x1, x2):
        """Calculate the distance between two tensors.

        Parameters
        ----------
        x1, x2 : `GpuTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            Distance between the tensors.

        Examples
        --------
        The default case is the Euclidean distance:

        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 2, -1])
        >>> r3.dist(x, y)  # 3^2 + 4^2 = 25
        5.0

        Taking a different exponent or a weighting is also possible
        during space creation:

        >>> r3 = odl.rn(3, impl='gpuarray', exponent=1)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 2, -1])
        >>> r3.dist(x, y)  # 3 + 4 = 7
        7.0

        >>> r3 = odl.rn(3, impl='gpuarray', weighting=2, exponent=1)
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
        x : `GpuTensor`
            The tensor whose norm is calculated.

        Returns
        -------
        norm : float
            Norm of the tensor.

        Examples
        --------
        The default case is the Euclidean norm:

        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.element([3, 4, 0])
        >>> r3.norm(x)  # 3^2 + 4^2 = 25
        5.0

        Taking a different exponent or a weighting is also possible
        during space creation:

        >>> r3 = odl.rn(3, impl='gpuarray', exponent=1)
        >>> x = r3.element([3, 4, 0])
        >>> r3.norm(x)  # 3 + 4 = 7
        7.0

        >>> r3 = odl.rn(3, impl='gpuarray', weighting=2, exponent=1)
        >>> x = r3.element([3, 4, 0])
        >>> r3.norm(x)  # 2*3 + 2*4 = 14
        14.0
        """
        return self.weighting.norm(x)

    def _inner(self, x1, x2):
        """Raw inner product of two tensors.

        Parameters
        ----------
        x1, x2 : `GpuTensor`
            The tensors whose inner product is calculated.

        Returns
        -------
        inner : `field` element
            Inner product of the tensors.

        Examples
        --------
        The default case is the dot product:

        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 0, 1])
        >>> r3.inner(x, y)  # 1*(-1) + 2*0 + 3*1 = 2
        2.0

        Taking a different weighting is also possible during space
        creation:

        >>> r3 = odl.rn(3, impl='gpuarray', weighting=2)
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
        x1, x2 : `GpuTensor`
            Factors in the product.
        out : `GpuTensor`
            Tensor to which the result is written.

        Examples
        --------
        Out-of-place evaluation:

        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 0, 1])
        >>> r3.multiply(x, y)
        rn(3, impl='gpuarray').element([-1.,  0.,  3.])

        In-place:

        >>> out = r3.element()
        >>> result = r3.multiply(x, y, out=out)
        >>> result
        rn(3, impl='gpuarray').element([-1.,  0.,  3.])
        >>> result is out
        True
        """
        multiply(x1.data, x2.data, out=out.data)

    def _divide(self, x1, x2, out):
        """Entry-wise division of two tensors, assigned to out.

        Parameters
        ----------
        x1, x2 : `GpuTensor`
            Dividend and divisor in the quotient.
        out : `GpuTensor`
            Tensor to which the result is written.

        Examples
        --------
        Out-of-place evaluation:

        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, 2, 1])
        >>> r3.divide(x, y)
        rn(3, impl='gpuarray').element([-1.,  1.,  3.])

        In-place:

        >>> out = r3.element()
        >>> result = r3.divide(x, y, out=out)
        >>> result
        rn(3, impl='gpuarray').element([-1.,  1.,  3.])
        >>> result is out
        True
        """
        # TODO: import from ufuncs
        divide(x1.data, x2.data, out=out.data)

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

        default_ctx = pygpu.get_default_context()
        optargs = [('impl', self.impl, 'numpy'),  # for the helper functions
                   ('context', self.context, default_ctx)]
        inner_str = signature_string(posargs, optargs)
        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str

        return '{}({})'.format(constructor_name, inner_str)

    @property
    def element_type(self):
        """`GpuTensor`"""
        return GpuTensor

    @staticmethod
    def available_dtypes():
        """Return the data types available for this space.

        See Also
        --------
        pygpu.dtypes.NAME_TO_DTYPE

        Notes
        -----
        The available dtypes may depend on the operating system.
        """
        return tuple(set(pygpu.dtypes.NAME_TO_DTYPE.values()))

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

            ``RealNumbers()`` : ``np.dtype('float64')``

            ``ComplexNumbers()`` : not supported
        """
        if field is None or field == RealNumbers():
            return np.dtype('float64')
        else:
            raise ValueError('no default data type defined for field {}.'
                             ''.format(field))


class GpuTensor(Tensor):

    """Representation of an `GpuTensorSpace` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        super(GpuTensor, self).__init__(space)
        self.__data = data

    @property
    def data(self):
        """Raw `pygpu._array.ndgpuarray` representing the data."""
        return self.__data

    @property
    def ndim(self):
        """Number of axes (=dimensions) of this tensor."""
        return self.space.ndim

    @property
    def context(self):
        """The GPU context of this tensor."""
        return self.space.context

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

        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.element([1, 2, 3])
        >>> x.asarray()
        array([ 1.,  2.,  3.])
        >>> int_spc = odl.tensor_space(3, impl='gpuarray', dtype=int)
        >>> x = int_spc.element([1, 2, 3])
        >>> x.asarray()
        array([1, 2, 3])
        >>> tensors = odl.rn((2, 3), impl='gpuarray', dtype='float32')
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
            return np.asarray(self.data)
        else:
            self.data.read(out)
            return out

    @property
    def data_ptr(self):
        """A raw pointer to the data container.

        Examples
        --------
        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.one()
        >>> x.data_ptr  # doctest: +SKIP
        47259975936
        """
        return self.data.gpudata

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
        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.element([1, 2, 3])
        >>> same_x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, -2, -3])
        >>> x == same_x
        True
        >>> x == y
        False

        Space membership matters:

        >>> int_spc = odl.tensor_space(3, impl='gpuarray', dtype=int)
        >>> x_int = int_spc.element([1, 2, 3])
        >>> x == x_int
        False
        """
        if other is self:
            return True
        elif other not in self.space:
            return False
        elif self.size < int(1e6):
            # Not worth launching a kernel
            # TODO: benchmark this
            return np.array_equal(self.data, other.data)
        else:
            # TODO: find a better implementation
            # Create kernel summing the number of equal entries
            args = [pygpu.tools.as_argument(self.data, 'a'),
                    pygpu.tools.as_argument(other.data, 'b')]
            equal_sum_ker = pygpu.reduction.ReductionKernel(
                context=self.data.context, dtype_out=int, neutral='0',
                reduce_expr='a + b', redux=[True] * self.ndim,
                map_expr='a[i] == b[i]', arguments=args)
            num_equal = int(np.asarray(equal_sum_ker(self.data, other.data)))
            return num_equal == self.size

    def copy(self):
        """Create an identical (deep) copy of this tensor.

        Returns
        -------
        copy : `pygpu._array.ndgpuarray`
            A deep copy.

        Examples
        --------
        >>> r3 = odl.rn(3, impl='gpuarray')
        >>> x = r3.element([1, 2, 3])
        >>> y = x.copy()
        >>> y
        rn(3, impl='gpuarray').element([ 1.,  2.,  3.])
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
        values : scalar or `pygpu._array.ndgpuarray`
            The value(s) at the index (indices).

        Examples
        --------
        Indexing rules follow roughly the Numpy style, as far (or "fancy")
        as supported:

        >>> r5 = odl.rn(5, impl='gpuarray')
        >>> x = r5.element([1, 2, 3, 4, 5])
        >>> x[1:4]
        rn(3, impl='gpuarray').element([ 2.,  3.,  4.])
        >>> x[::2]
        rn(3, impl='gpuarray').element([ 1.,  3.,  5.])

        The returned views are writable, so modificatons alter the
        original array:

        >>> view = x[1:4]
        >>> view[:] = -1
        >>> view
        rn(3, impl='gpuarray').element([-1., -1., -1.])
        >>> x
        rn(5, impl='gpuarray').element([ 1., -1., -1., -1.,  5.])

        Multi-indexing is also directly supported:

        >>> tensors = odl.rn((2, 3), impl='gpuarray')
        >>> x = tensors.element([[1, 2, 3],
        ...                      [4, 5, 6]])
        >>> x[1, 2]
        6.0
        >>> x[1]  # row with index 1
        rn(3, impl='gpuarray').element([ 4.,  5.,  6.])
        >>> view = x[:, ::2]
        >>> view
        rn((2, 2), impl='gpuarray').element(
            [[ 1.,  3.],
             [ 4.,  6.]]
        )
        >>> view[:] = [[0, 0],
        ...            [0, 0]]
        >>> x
        rn((2, 3), impl='gpuarray').element(
            [[ 0.,  2.,  0.],
             [ 0.,  5.,  0.]]
        )
        """
        arr = self.data[indices]
        if arr.shape == ():
            if np.issubsctype(arr.dtype, np.floating):
                return float(np.asarray(arr))
            elif np.issubsctype(arr.dtype, np.complexfloating):
                return complex(np.asarray(arr))
            elif np.issubsctype(arr.dtype, np.integer):
                return int(np.asarray(arr))
            else:
                raise RuntimeError("no conversion for dtype '{}'"
                                   "".format(arr.dtype))
        else:
            space = type(self.space)(arr.shape, dtype=self.dtype,
                                     context=self.context)
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

        >>> r5 = odl.rn(5, impl='gpuarray')
        >>> x = r5.element([1, 2, 3, 4, 5])
        >>> x[1:4] = 0
        >>> x
        rn(5, impl='gpuarray').element([ 1.,  0.,  0.,  0.,  5.])
        >>> x[1:4] = [-1, 1, -1]
        >>> x
        rn(5, impl='gpuarray').element([ 1., -1.,  1., -1.,  5.])
        >>> y = r5.element([5, 5, 5, 8, 8])
        >>> x[:] = y
        >>> x
        rn(5, impl='gpuarray').element([ 5.,  5.,  5.,  8.,  8.])

        In higher dimensions, broadcasting can be applied to assign
        values:

        >>> tensors = odl.rn((2, 3), impl='gpuarray')
        >>> x = tensors.element([[1, 2, 3],
        ...                      [4, 5, 6]])
        >>> x[:] = [[6], [3]]  # rhs mimics (2, 1) shape
        >>> x
        rn((2, 3), impl='gpuarray').element(
            [[ 6.,  6.,  6.],
             [ 3.,  3.,  3.]]
        )

        Be aware of unsafe casts and over-/underflows, there
        will be warnings at maximum.

        >>> int_r3 = odl.tensor_space(3, impl='gpuarray', dtype='uint32')
        >>> x = int_r3.element([1, 2, 3])
        >>> x[0] = -1
        >>> x[0]
        4294967295
        """
        if isinstance(values, GpuTensor):
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

    @property
    def ufuncs(self):
        """`GpuTensorSpaceUfuncs`, access to NumPy style ufuncs.

        Examples
        --------
        >>> r2 = odl.rn(2, impl='gpuarray')
        >>> x = r2.element([1, -2])
        >>> x.ufuncs.absolute()
        rn(2, impl='gpuarray').element([ 1.,  2.])

        These functions can also be used with broadcasting or
        array-like input:
        >>> x.ufuncs.add(3)
        rn(2, impl='gpuarray').element([ 4.,  1.])
        >>> x.ufuncs.subtract([3, 3])
        rn(2, impl='gpuarray').element([-2., -5.])

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
        rn(2, impl='gpuarray').element([ 4.,  2.])
        >>> result is out
        True

        Notes
        -----
        These functions are optimized for use with GPU arrays and incur
        no overhead.
        """
        # TODO: import from ufuncs
        return GpuTensorSpaceUfuncs(self)

    @property
    def real(self):
        """Real part of this tensor.

        Returns
        -------
        real : `GpuTensor` view with real dtype
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
        imag : `GpuTensor`
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
        out : `GpuTensor`, optional
            Tensor to which the complex conjugate is written.
            Must be an element of this tensor's space.

        Returns
        -------
        out : `GpuTensor`
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
                return super(GpuTensorSpace, self).__ipow__(other)
        except TypeError:
            pass

        self.ufuncs.power(self.data, other, out=self.data)
        return self


# --- Weightings --- #


def _weighting(weights, exponent):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weights):
        weighting = GpuTensorSpaceConstWeighting(weights, exponent=exponent)
    else:
        # TODO: sequence of 1D array-likes
        weights = pygpu.array(weights, copy=False)
        weighting = GpuTensorSpaceArrayWeighting(weights, exponent=exponent)
    return weighting


def gpuary_weighted_inner(weights):
    """Weighted inner product on `GpuTensorSpace` spaces as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the inner product. A scalar is interpreted as a
        constant weight, an array as pointwise weights.

    Returns
    -------
    inner : callable
        Inner product function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the shapes
        of the weighting and the space must match.

    See Also
    --------
    GpuTensorSpaceConstWeighting
    GpuTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=2.0).inner


def gpuary_weighted_norm(weights, exponent=2.0):
    """Weighted norm on `GpuTensorSpace` spaces as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the inner product. A scalar is interpreted as a
        constant weight, an array as pointwise weights.
    exponent : positive float
        Exponent of the norm.

    Returns
    -------
    norm : callable
        Norm function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the shapes
        of the weighting and the space must match.

    See Also
    --------
    GpuTensorSpaceConstWeighting
    GpuTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=exponent).norm


def gpuary_weighted_dist(weights, exponent=2.0):
    """Weighted distance on `GpuTensorSpace` spaces as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the inner product. A scalar is interpreted as a
        constant weight, an array as pointwise weights.
    exponent : positive float
        Exponent of the norm inducing the distance metric.

    Returns
    -------
    dist : callable
        Distance function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the shapes
        of the weighting and the space must match.

    See Also
    --------
    GpuTensorSpaceConstWeighting
    GpuTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=exponent).dist


def _pnorm(x, p, w=None):
    """Implementation of the (weighted) p-norm.

    Parameters
    ----------
    x : `GpuTensor`
        Tensor whose norm should be computed.
    p : float
        Exponent of the norm.
    w : `pygpu.gpuarray.GpuArray` or `numpy.ndarray`, optional
        Weighting array of the norm. Its shape and dtype must be
        compatible with ``x``.

    Returns
    -------
    pnorm : float
        The p-norm of ``x``.
    """
    if w is None:
        order = 'F' if x.data.flags.f_contiguous else 'C'
    else:
        order = 'F' if all(a.flags.f_contiguous for a in (x.data, w)) else 'C'

    weighted = (w is not None)  # for better readability

    # TODO: benchmark to find a reasonable threshold; the current value is
    # for testing
    if x.size < int(2):
        # Not worth launching a kernel
        if p == 0:
            if weighted:
                xp = np.multiply(x.data, w)
            else:
                xp = x.data
            return float(np.count_nonzero(xp))
        elif p == 1:
            xp = np.abs(x.data)
            if weighted:
                np.multiply(xp, w, out=xp)
            return float(np.sum(xp))
        elif p == 2:
            xp = np.square(x.data)
            if weighted:
                np.multiply(xp, w, out=xp)
            return float(np.sqrt(np.sum(xp)))
        elif p == float('inf'):
            xp = np.abs(x.data)
            if weighted:
                np.multiply(xp, w, out=xp)
            return float(np.max(xp))
        else:
            xp = np.abs(x.data)
            np.power(xp, p, out=xp)
            if weighted:
                np.multiply(xp, w, out=xp)
            return float(np.sum(xp) ** (1 / p))

    else:
        args = [pygpu.tools.as_argument(x.data, 'a')]
        if w is not None:
            # Convert weights to GPU array if necessary
            if not isinstance(w, pygpu.gpuarray.GpuArray):
                w = pygpu.gpuarray.array(w, dtype=x.dtype, order=order,
                                         copy=False, context=x.context)
            args.append(pygpu.tools.as_argument(w, 'w'))

        if p == 0:
            # Return number of non-zeros (weighting shouldn't matter, but
            # we'll use it anyway since we don't check by default that no
            # weight is zero)
            map_expr = 'a[i] != 0'
            if weighted:
                map_expr = 'w[i] * ' + map_expr
            nrm0_ker = pygpu.reduction.ReductionKernel(
                context=x.context, dtype_out=x.dtype, neutral=0,
                reduce_expr='a + b', redux=[True] * x.ndim,
                map_expr=map_expr, arguments=args)
            # GpuArray has no __float__, need to convert to Numpy array
            if weighted:
                return float(np.asarray(nrm0_ker(x.data, w)))
            else:
                return float(np.asarray(nrm0_ker(x.data)))

        elif p == 1:
            # Return the sum of the absolute values
            if x.context.kind == b'opencl' and is_floating_dtype(x.dtype):
                # TODO: check what to do here exactly
                map_expr = 'fabs(a[i])'
            else:
                map_expr = 'abs(a[i])'

            if weighted:
                map_expr = 'w[i] * ' + map_expr
            nrm1_ker = pygpu.reduction.ReductionKernel(
                context=x.context, dtype_out=x.dtype, neutral=0,
                reduce_expr='a + b', redux=[True] * x.ndim,
                map_expr=map_expr, arguments=args)
            # GpuArray has no __float__, need to convert to Numpy array
            if weighted:
                return float(np.asarray(nrm1_ker(x.data, w)))
            else:
                return float(np.asarray(nrm1_ker(x.data)))

        elif p == 2:
            # Return the square root of the sum of squares
            map_expr = 'a[i] * a[i]'
            if weighted:
                map_expr = 'w[i] * ' + map_expr

            nrm2_ker = pygpu.reduction.ReductionKernel(
                context=x.context, dtype_out=x.dtype, neutral=0,
                reduce_expr='a + b', redux=[True] * x.ndim,
                map_expr=map_expr, arguments=args)
            if weighted:
                return float(np.sqrt(nrm2_ker(x.data, w)))
            else:
                return float(np.sqrt(nrm2_ker(x.data)))

        elif p == float('inf'):
            # Return the maximum absolute value
            if x.context.kind == b'opencl' and is_floating_dtype(x.dtype):
                # TODO: check what to do here exactly
                map_expr = 'fabs(a[i])'
            else:
                map_expr = 'abs(a[i])'

            if weighted:
                map_expr = 'w[i] * ' + map_expr
            # Determine neutral element of comparison, which is the minimum
            # possible value for the dtype
            if np.issubsctype(x.dtype, np.integer):
                neutral = np.iinfo(x.dtype).min
            elif np.issubsctype(x.dtype, np.floating):
                neutral = np.finfo(x.dtype).min
            else:
                raise RuntimeError('bad dtype {}'.format(x.dtype))
            nrminf_ker = pygpu.reduction.ReductionKernel(
                context=x.context, dtype_out=x.dtype, neutral=neutral,
                reduce_expr='(a > b) ? a : b', redux=[True] * x.ndim,
                map_expr=map_expr, arguments=args)
            if weighted:
                return float(np.asarray(nrminf_ker(x.data, w)))
            else:
                return float(np.asarray(nrminf_ker(x.data)))

        else:
            # Return the p-th root of the sum of the absolute values'
            # p-th powers

            # Some data type manipulation necessary to find adequate
            # combinations for `pow`
            if is_int_dtype(x.dtype):
                # Using double for integer dtypes
                x_data = x.data.astype(float)
                if p == int(p):
                    p = np.array(p, dtype=int)
                else:
                    p = np.array(p, dtype=float)
            else:
                x_data = x.data
                if p == int(p):
                    p = np.array(p, dtype=int)
                else:
                    p = np.array(p, dtype=x.dtype)

            args = [pygpu.tools.as_argument(x_data, 'a'),
                    pygpu.tools.as_argument(p, 'p')]
            if x.context.kind == b'opencl' and is_floating_dtype(x.dtype):
                # TODO: check what to do here exactly
                map_expr = 'pow(fabs(a[i]), p)'
            else:
                map_expr = 'pow(abs(a[i]), p)'

            if weighted:
                map_expr = 'w[i] * ' + map_expr
            nrmp_ker = pygpu.reduction.ReductionKernel(
                context=x.context, dtype_out=x.dtype, neutral=0,
                reduce_expr='a + b', redux=[True] * x.ndim,
                map_expr=map_expr, arguments=args)
            if weighted:
                return float(np.power(nrmp_ker(x.data, w, p), 1.0 / p))
            else:
                return float(np.power(nrmp_ker(x.data, p), 1.0 / p))


def _pdist(x1, x2, p, w=None):
    """Implementation of the (weighted) p-distance.

    Parameters
    ----------
    x1, x2 : GpuTensor
        Tensors whose mutual distance should be computed. They must have
        the same shape.
    p : float
        Exponent of the norm used to calculate the distance.
    w : `pygpu.gpuarray.GpuArray` or `numpy.ndarray`, optional
        Weighting array of the distance. Its shape and dtype must be
        compatible with ``x1`` and ``x2``.

    Returns
    -------
    pdist : float
        The p-distance from ``x1`` to ``x2``.
    """
    p = float(p)

    if w is None:
        order = ('F' if all(a.flags.f_contiguous
                            for a in (x1.data, x2.data))
                 else 'C')
    else:
        order = ('F' if all(a.flags.f_contiguous
                            for a in (x1.data, x2.data, w))
                 else 'C')

    weighted = (w is not None)  # for better readability

    # TODO: benchmark to find a reasonable threshold
    if x1.size < int(2):
        # Not worth launching a kernel
        if p == 0:
            diffp = np.subtract(x1.data, x2.data)
            if weighted:
                np.multiply(diffp, w, out=diffp)
            return float(np.count_nonzero(diffp))
        elif p == 1:
            diffp = np.abs(np.subtract(x1.data, x2.data))
            if weighted:
                np.multiply(diffp, w, out=diffp)
            return float(np.sum(diffp))
        elif p == 2:
            diffp = np.subtract(x1.data, x2.data)
            np.square(diffp, out=diffp)
            if weighted:
                np.multiply(diffp, w, out=diffp)
            return float(np.sqrt(np.sum(diffp)))
        elif p == float('inf'):
            diffp = np.subtract(x1.data, x2.data)
            np.abs(diffp, out=diffp)
            if weighted:
                np.multiply(diffp, w, out=diffp)
            return float(np.max(diffp))
        else:
            diffp = np.subtract(x1.data, x2.data)
            np.abs(diffp, out=diffp)
            np.power(diffp, p, out=diffp)
            if weighted:
                np.multiply(diffp, w, out=diffp)
            return float(np.sum(diffp) ** (1 / p))

    else:
        args = [pygpu.tools.as_argument(x1.data, 'a'),
                pygpu.tools.as_argument(x2.data, 'b')]
        if w is not None:
            # Convert weights to GPU array if necessary
            if not isinstance(w, pygpu.gpuarray.GpuArray):
                w = pygpu.gpuarray.array(w, dtype=x1.dtype, order=order,
                                         copy=False, context=x1.context)
            args.append(pygpu.tools.as_argument(w, 'w'))

        if p == 0:
            # Return number of non-zero differences (weighting shouldn't
            # matter, but we'll use it anyway since we don't check by default
            # that no weight is zero)
            map_expr = '(a[i] - b[i]) != 0'
            if weighted:
                map_expr = 'w[i] * ' + map_expr
            dist0_ker = pygpu.reduction.ReductionKernel(
                context=x1.context, dtype_out=x1.dtype, neutral=0,
                reduce_expr='a + b', redux=[True] * x1.ndim,
                map_expr=map_expr, arguments=args)
            # GpuArray has no __float__, need to convert to Numpy array
            if weighted:
                return float(np.asarray(dist0_ker(x1.data, x2.data, w)))
            else:
                return float(np.asarray(dist0_ker(x1.data, x2.data)))

        elif p == 1:
            # Return the sum of the absolute differences
            if x1.context.kind == b'opencl' and is_floating_dtype(x1.dtype):
                # TODO: check what to do here exactly
                map_expr = 'fabs(a[i] - b[i])'
            else:
                map_expr = 'abs(a[i] - b[i])'

            if weighted:
                map_expr = 'w[i] * ' + map_expr
            dist1_ker = pygpu.reduction.ReductionKernel(
                context=x1.context, dtype_out=x1.dtype, neutral=0,
                reduce_expr='a + b', redux=[True] * x1.ndim,
                map_expr=map_expr, arguments=args)
            # GpuArray has no __float__, need to convert to Numpy array
            if weighted:
                return float(np.asarray(dist1_ker(x1.data, x2.data, w)))
            else:
                return float(np.asarray(dist1_ker(x1.data, x2.data)))

        elif p == 2:
            # Return the square root of the sum of squared differences
            map_expr = '(a[i] - b[i]) * (a[i] - b[i])'
            if weighted:
                map_expr = 'w[i] * ' + map_expr

            dist2_ker = pygpu.reduction.ReductionKernel(
                context=x1.context, dtype_out=x1.dtype, neutral=0,
                reduce_expr='a + b', redux=[True] * x1.ndim,
                map_expr=map_expr, arguments=args)
            if weighted:
                return float(np.sqrt(dist2_ker(x1.data, x2.data, w)))
            else:
                return float(np.sqrt(dist2_ker(x1.data, x2.data)))

        elif p == float('inf'):
            # Return the maximum absolute value
            if x1.context.kind == b'opencl' and is_floating_dtype(x1.dtype):
                # TODO: check what to do here exactly
                map_expr = 'fabs(a[i] - b[i])'
            else:
                map_expr = 'abs(a[i] - b[i])'

            if weighted:
                map_expr = 'w[i] * ' + map_expr
            # Determine neutral element of comparison, which is the minimum
            # possible value for the dtype
            if np.issubsctype(x1.dtype, np.integer):
                neutral = np.iinfo(x1.dtype).min
            elif np.issubsctype(x1.dtype, np.floating):
                neutral = np.finfo(x1.dtype).min
            else:
                raise RuntimeError('bad dtype {}'.format(x1.dtype))
            distinf_ker = pygpu.reduction.ReductionKernel(
                context=x1.context, dtype_out=x1.dtype, neutral=neutral,
                reduce_expr='(a > b) ? a : b', redux=[True] * x1.ndim,
                map_expr=map_expr, arguments=args)
            if weighted:
                return float(np.asarray(distinf_ker(x1.data, x2.data, w)))
            else:
                return float(np.asarray(distinf_ker(x1.data, x2.data)))

        else:
            # Return the p-th root of the sum of the absolute differences'
            # p-th powers
            # Some data type manipulation necessary to find adequate
            # combinations for `pow`
            if p == int(p):
                p = np.array(p, dtype=int)
            else:
                if is_int_dtype(x1.dtype):
                    p = np.array(p, dtype=float)
                else:
                    p = np.array(p, dtype=x1.dtype)
            args.append(pygpu.tools.as_argument(p, 'p'))
            if x1.context.kind == b'opencl' and is_floating_dtype(x1.dtype):
                # TODO: check what to do here exactly
                map_expr = 'pow(fabs(a[i] - b[i]), p)'
            else:
                map_expr = 'pow(abs(a[i] - b[i]), p)'

            if weighted:
                map_expr = 'w[i] * ' + map_expr
            distp_ker = pygpu.reduction.ReductionKernel(
                context=x1.context, dtype_out=x1.dtype, neutral=0,
                reduce_expr='a + b', redux=[True] * x1.ndim,
                map_expr=map_expr, arguments=args)
            if weighted:
                return float(np.power(distp_ker(x1.data, x2.data, w, p),
                                      1.0 / p))
            else:
                return float(np.power(distp_ker(x1.data, x2.data, p),
                                      1.0 / p))


def _inner(x1, x2, w=None):
    """Implementation of the (weighted) Euclidean inner product.

    Parameters
    ----------
    x1, x2 : GpuTensor
        Tensors whose inner product should be computed. They must have
        the same shape.
    w : `array-like`, optional
        Weighting array of the inner product. Its shape and dtype must be
        compatible with ``x1`` and ``x2``.

    Returns
    -------
    inner : float
        The inner product of ``x1`` and ``x2``.
    """
    if w is None:
        order = ('F' if all(a.flags.f_contiguous
                            for a in (x1.data, x2.data))
                 else 'C')
    else:
        order = ('F' if all(a.flags.f_contiguous
                            for a in (x1.data, x2.data, w))
                 else 'C')

    weighted = (w is not None)  # for better readability

    # TODO: benchmark to find a reasonable threshold
    if x1.size < int(2):
        return np.dot(np.multiply(x2.data, w), x1.data)

    else:
        args = [pygpu.tools.as_argument(x1.data, 'a'),
                pygpu.tools.as_argument(x2.data, 'b')]
        map_expr = 'a[i] * b[i]'
        if w is not None:
            # Convert weights to GPU array if necessary
            if not isinstance(w, pygpu.gpuarray.GpuArray):
                w = pygpu.gpuarray.array(w, dtype=x1.dtype, order=order,
                                         copy=False, context=x1.context)
            args.append(pygpu.tools.as_argument(w, 'w'))
            map_expr = 'w[i] * ' + map_expr
        inner_ker = pygpu.reduction.ReductionKernel(
            context=x1.context, dtype_out=x1.dtype, neutral=0,
            reduce_expr='a + b', redux=[True] * x1.ndim,
            map_expr=map_expr, arguments=args)
        if weighted:
            return float(np.asarray(inner_ker(x1.data, x2.data, w)))
        else:
            return float(np.asarray(inner_ker(x1.data, x2.data)))


class GpuTensorSpaceArrayWeighting(ArrayWeighting):

    """Array weighting for `GpuTensorSpace`.

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
        if isinstance(array, GpuTensor):
            array = array.data
        elif not isinstance(array, gpuary.GpuArray):
            array = gpuary.array(array, copy=False, cls=ndgpuarray)
        super(GpuTensorSpaceArrayWeighting, self).__init__(
            array, impl='gpuarray', exponent=exponent)

    def inner(self, x1, x2):
        """Calculate the weighted inner product of two tensors.

        Parameters
        ----------
        x1, x2 : `GpuTensor`
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
            return x1.space.field.element(_inner(x1, x2, self.array))

    def norm(self, x):
        """Calculate the weighted norm of a tensor.

        Parameters
        ----------
        x : `GpuTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the provided tensor.
        """
        return float(_pnorm(x, self.exponent, self.array))


class GpuTensorSpaceConstWeighting(ConstWeighting):

    """Constant weighting for `GpuTensorSpace`.

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
        super(GpuTensorSpaceConstWeighting, self).__init__(
            constant, impl='gpuarray', exponent=exponent)

    def inner(self, x1, x2):
        """Calculate the weighted inner product of two tensors.

        Parameters
        ----------
        x1, x2 : `GpuTensor`
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
            return x1.space.field.element(self.const * _inner(x1, x2))

    def norm(self, x):
        """Calculate the constant-weighted norm of a tensor.

        Parameters
        ----------
        x1 : `GpuTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the tensor.
        """
        if self.exponent == 0:
            return float(_pnorm(x, p=0))
        elif self.exponent == 1:
            return float(self.const * _pnorm(x, p=1))
        elif self.exponent == 2:
            return float(np.sqrt(self.const) * _pnorm(x, p=2))
        elif self.exponent == float('inf'):
            return float(self.const * _pnorm(x, p=float('inf')))
        else:
            return float(self.const ** (1 / self.exponent) *
                         _pnorm(x, p=self.exponent))

    def dist(self, x1, x2):
        """Calculate the weighted distance between two tensors.

        Parameters
        ----------
        x1, x2 : `GpuTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the tensors.
        """
        if self.exponent == 0:
            return float(_pdist(x1, x2, p=0))
        elif self.exponent == 1:
            return float(self.const * _pdist(x1, x2, p=1))
        elif self.exponent == 2:
            return float(np.sqrt(self.const) * _pdist(x1, x2, p=2))
        elif self.exponent == float('inf'):
            return float(self.const * _pdist(x1, x2, p=float('inf')))
        else:
            return float(self.const ** (1 / self.exponent) *
                         _pdist(x1, x2, p=self.exponent))


class GpuTensorSpaceCustomInner(CustomInner):

    """Class for handling custom inner products in `GpuTensorSpace`."""

    def __init__(self, inner):
        """Initialize a new instance.

        Parameters
        ----------
        inner : callable
            The inner product implementation. It must accept two
            `GpuTensor` arguments, return an element from their space's
            field (real or complex number) and satisfy the following
            conditions for all vectors ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``
        """
        super(GpuTensorSpaceCustomInner, self).__init__(inner, impl='gpuarray')


class GpuTensorSpaceCustomNorm(CustomNorm):

    """Class for handling a user-specified norm in `GpuTensorSpace`.

    Note that this removes ``inner``.
    """

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : callable
            The norm implementation. It must accept an `GpuTensor`
            argument, return a float and satisfy the following
            conditions for all vectors ``x, y`` and scalars ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``
        """
        super(GpuTensorSpaceCustomNorm, self).__init__(norm, impl='gpuarray')


class GpuTensorSpaceCustomDist(CustomDist):

    """Class for handling a user-specified distance in `GpuTensorSpace`.

    Note that this removes ``inner`` and ``norm``.
    """

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : callable
            The distance function defining a metric on `GpuTensorSpace`.
            It must accept two `GpuTensor` arguments, return a float and
            fulfill the following mathematical conditions for any three
            vectors ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``
        """
        super(GpuTensorSpaceCustomDist, self).__init__(dist, impl='gpuarray')


if __name__ == '__main__':
    if PYGPU_AVAILABLE:
        from odl.util.testutils import run_doctests
        run_doctests()
