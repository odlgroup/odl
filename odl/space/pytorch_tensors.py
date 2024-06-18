# Copyright 2024 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""PyTorch implementation of tensor spaces."""

from __future__ import absolute_import, division, print_function
from future.utils import native

import ctypes
from builtins import object
from functools import partial

import numpy as np
import torch

from odl.set.sets import ComplexNumbers, RealNumbers
from odl.set.space import LinearSpaceTypeError
from odl.space.base_tensors import Tensor, TensorSpace
from odl.space.weighting import (
    ArrayWeighting, ConstWeighting, CustomDist, CustomInner, CustomNorm,
    Weighting)
from odl.util import (
    dtype_str, is_floating_dtype, is_numeric_dtype, is_real_dtype, nullcontext,
    signature_string, writable_array)

__all__ = ('PytorchTensorSpace',)


_PYTORCH_DTYPES = {np.dtype('float32'): torch.float32,
                   np.dtype('float64'): torch.float64,
                   np.dtype('complex64'): torch.complex64,
                   np.dtype('complex128'): torch.complex128}

# Define size thresholds to switch implementations
THRESHOLD_SMALL = 100
THRESHOLD_MEDIUM = 50000


class PytorchTensorSpace(TensorSpace):

    """Set of tensors of arbitrary data type, implemented with Pytorch.

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

    This class is implemented using `torch.Tensor`'s as back-end.

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

    def __init__(self, shape, dtype=None, **kwargs):
        r"""Initialize a new instance.

        Parameters
        ----------
        shape : positive int or sequence of positive ints
            Number of entries per axis for elements in this space. A
            single integer results in a space with rank 1, i.e., 1 axis.
        dtype :
            Data type of each element. Can be provided in any
            way the `torch.dtype` function understands, e.g.
            as built-in type or as a string. For ``None``,
            the `default_dtype` of this space (``float64``) is used.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no
            inner product is defined.

            This option has no impact if either ``dist``, ``norm`` or
            ``inner`` is given, or if ``dtype`` is non-numeric.

            Default: 2.0

        Other Parameters
        ----------------
        torch_device : optional, PyTorch device identifier
            Where to store and process data (i.e. arrays) representing elements
            of this space. Should typically be a GPU (cuda) if available, else
            CPU as also used by NumPy.

        weighting : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weighting``:

            ``None``: no weighting, i.e. weighting with ``1.0`` (default).

            `Weighting`: Use this weighting as-is. Compatibility
            with this space's elements is not checked during init.

            ``float``: Weighting by a constant.

            array-like: Pointwise weighting by an array.

            This option cannot be combined with ``dist``,
            ``norm`` or ``inner``. It also cannot be used in case of
            non-numeric ``dtype``.

        dist : callable, optional
            Distance function defining a metric on the space.
            It must accept two `PytorchTensor` arguments and return
            a non-negative real number. See ``Notes`` for
            mathematical requirements.

            By default, ``dist(x, y)`` is calculated as ``norm(x - y)``.

            This option cannot be combined with ``weight``,
            ``norm`` or ``inner``. It also cannot be used in case of
            non-numeric ``dtype``.

        norm : callable, optional
            The norm implementation. It must accept a
            `PytorchTensor` argument, return a non-negative real number.
            See ``Notes`` for mathematical requirements.

            By default, ``norm(x)`` is calculated as ``inner(x, x)``.

            This option cannot be combined with ``weight``,
            ``dist`` or ``inner``. It also cannot be used in case of
            non-numeric ``dtype``.

        inner : callable, optional
            The inner product implementation. It must accept two
            `PytorchTensor` arguments and return an element of the field
            of the space (usually real or complex number).
            See ``Notes`` for mathematical requirements.

            This option cannot be combined with ``weight``,
            ``dist`` or ``norm``. It also cannot be used in case of
            non-numeric ``dtype``.

        kwargs :
            Further keyword arguments are passed to the weighting
            classes.

        See Also
        --------
        odl.space.space_utils.rn : constructor for real tensor spaces
        odl.space.space_utils.cn : constructor for complex tensor spaces
        odl.space.space_utils.tensor_space :
            constructor for tensor spaces of arbitrary scalar data type

        Notes
        -----
        - A distance function or metric on a space :math:`\mathcal{X}`
          is a mapping
          :math:`d:\mathcal{X} \times \mathcal{X} \to \mathbb{R}`
          satisfying the following conditions for all space elements
          :math:`x, y, z`:

          * :math:`d(x, y) \geq 0`,
          * :math:`d(x, y) = 0 \Leftrightarrow x = y`,
          * :math:`d(x, y) = d(y, x)`,
          * :math:`d(x, y) \leq d(x, z) + d(z, y)`.

        - A norm on a space :math:`\mathcal{X}` is a mapping
          :math:`\| \cdot \|:\mathcal{X} \to \mathbb{R}`
          satisfying the following conditions for all
          space elements :math:`x, y`: and scalars :math:`s`:

          * :math:`\| x\| \geq 0`,
          * :math:`\| x\| = 0 \Leftrightarrow x = 0`,
          * :math:`\| sx\| = |s| \cdot \| x \|`,
          * :math:`\| x+y\| \leq \| x\| +
            \| y\|`.

        - An inner product on a space :math:`\mathcal{X}` over a field
          :math:`\mathbb{F} = \mathbb{R}` or :math:`\mathbb{C}` is a
          mapping
          :math:`\langle\cdot, \cdot\rangle: \mathcal{X} \times
          \mathcal{X} \to \mathbb{F}`
          satisfying the following conditions for all
          space elements :math:`x, y, z`: and scalars :math:`s`:

          * :math:`\langle x, y\rangle =
            \overline{\langle y, x\rangle}`,
          * :math:`\langle sx + y, z\rangle = s \langle x, z\rangle +
            \langle y, z\rangle`,
          * :math:`\langle x, x\rangle = 0 \Leftrightarrow x = 0`.

        Examples
        --------
        Explicit initialization with the class constructor:

        >>> space = PytorchTensorSpace(3, float)
        >>> space
        rn(3)
        >>> space.shape
        (3,)
        >>> space.dtype
        dtype('float64')
        """
        super(PytorchTensorSpace, self).__init__(shape, dtype)
        if self.dtype not in self.available_dtypes():
            raise ValueError('`dtype` {!r} not supported'
                             ''.format(dtype_str(dtype)))

        torch_device = kwargs.pop('torch_device', "cpu")
        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weighting = kwargs.pop('weighting', None)
        exponent = kwargs.pop('exponent', getattr(weighting, 'exponent', 2.0))

        self._torch_device = torch.device(torch_device)

        if (not is_numeric_dtype(self.dtype) and
                any(x is not None for x in (dist, norm, inner, weighting))):
            raise ValueError('cannot use any of `weighting`, `dist`, `norm` '
                             'or `inner` for non-numeric `dtype` {}'
                             ''.format(dtype))
        else:
            self._torch_dtype = _PYTORCH_DTYPES[self.dtype]
        if exponent != 2.0 and any(x is not None for x in (dist, norm, inner)):
            raise ValueError('cannot use any of `dist`, `norm` or `inner` '
                             'for exponent != 2')
        # Check validity of option combination (0 or 1 may be provided)
        num_extra_args = sum(a is not None
                             for a in (dist, norm, inner, weighting))
        if num_extra_args > 1:
            raise ValueError('invalid combination of options `weighting`, '
                             '`dist`, `norm` and `inner`')

        # Set the weighting
        if weighting is not None:
            if isinstance(weighting, Weighting):
                if weighting.impl != 'pytorch':
                    raise ValueError("`weighting.impl` must be 'pytorch', "
                                     '`got {!r}'.format(weighting.impl))
                if weighting.exponent != exponent:
                    raise ValueError('`weighting.exponent` conflicts with '
                                     '`exponent`: {} != {}'
                                     ''.format(weighting.exponent, exponent))
                self.__weighting = weighting
            else:
                self.__weighting = _weighting(weighting, exponent)

            # Check (afterwards) that the weighting input was sane
            if isinstance(self.weighting, PytorchTensorSpaceArrayWeighting):
                if self.weighting.array.dtype == object:
                    raise ValueError('invalid `weighting` argument: {}'
                                     ''.format(weighting))
                elif not np.can_cast(self.weighting.array.dtype, self.dtype):
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
            self.__weighting = PytorchTensorSpaceCustomDist(dist)
        elif norm is not None:
            self.__weighting = PytorchTensorSpaceCustomNorm(norm)
        elif inner is not None:
            self.__weighting = PytorchTensorSpaceCustomInner(inner)
        else:
            # No weighting, i.e., weighting with constant 1.0
            self.__weighting = PytorchTensorSpaceConstWeighting(1.0, exponent)

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unknown keyword arguments {}'.format(kwargs))

    @property
    def impl(self):
        """Name of the implementation back-end: ``'pytorch'``."""
        return 'pytorch'

    @property
    def default_order(self):
        """Default (and only) storage order for new elements in this space: ``'C'``."""
        return 'C'

    def is_suitable_scalar(self, s):
        if self._torch_dtype in [torch.complex64, torch.complex128]:
            return type(s) is complex
        else:
            return type(s) is float
        # Singleton-tensor version:
        # if not isinstance(s, torch.Tensor):
        #     return False
        # elif s.dtype != self._torch_dtype:
        #     return False
        # elif s.shape != ():
        #     return False
        # else:
        #     return True

    def as_suitable_scalar(self, s):
        """Try to convert `s` to a type that can be scalar-multiplied with
        torch tensors.
        """
        if self._torch_dtype in [torch.complex64, torch.complex128]:
            return complex(s)
            # Arguably, this would be more appropriate:
            # return torch.tensor(complex(s), dtype=self._torch_dtype)
            # But this results in wrong PyTorch multiplication functions
            # being called.
        else:
            return float(s)
            # return torch.tensor(float(s), dtype=self._torch_dtype)

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.__weighting

    @property
    def is_weighted(self):
        """Return ``True`` if the space is not weighted by constant 1.0."""
        return not (
            isinstance(self.weighting, PytorchTensorSpaceConstWeighting) and
            self.weighting.const == 1.0)

    @property
    def exponent(self):
        """Exponent of the norm and the distance."""
        return self.weighting.exponent

    def element(self, inp=None, data_ptr=None, order=None):
        """Create a new element.

        Parameters
        ----------
        inp : `array-like`, optional
            Input used to initialize the new element.

            If ``inp`` is `None`, an empty element is created with no
            guarantee of its state (memory allocation only).
            All tensors use row-major storage (corrsponding to
            `order='C'` in NumPy).

            Otherwise, a copy is avoided whenever possible. This requires
            correct `shape` and `dtype`, and if ``order`` is provided,
            also contiguousness in that ordering. If any of these
            conditions is not met, a copy is made.

        data_ptr : int, optional
            Pointer to the start memory address of a contiguous PyTorch array
            or an equivalent raw container with the same total number of
            bytes.
            The option is also mutually exclusive with ``inp``.

        Returns
        -------
        element : `PytorchTensor`
            The new element, created from ``inp`` or from scratch.

        Examples
        --------
        Without arguments, an uninitialized element is created. With an
        array-like input, the element can be initialized:

        >>> space = odl.rn(3)           # TODO adapt / test
        >>> empty = space.element()
        >>> empty.shape
        (3,)
        >>> empty.space
        rn(3)
        >>> x = space.element([1, 2, 3])
        >>> x
        rn(3).element([ 1.,  2.,  3.])

        If the input already is a `torch.Tensor` of correct `dtype`, it
        will merely be wrapped, i.e., both array and space element access
        the same memory, such that mutations will affect both:

        >>> arr = torch.Tensor([1, 2, 3], dtype=float)  # TODO test
        >>> elem = odl.rn(3).element(arr)
        >>> elem[0] = 0
        >>> elem
        rn(3).element([ 0.,  2.,  3.])
        >>> arr
        array([ 0.,  2.,  3.])

        Elements can also be constructed from a data pointer, resulting
        again in shared memory:

        >>> int_space = odl.tensor_space((2, 3), dtype=int)
        >>> arr = torch.Tensor([[1, 2, 3],
        ...                     [4, 5, 6]], dtype=int, order='F')
        >>> ptr = arr.ctypes.data
        >>> y = int_space.element(data_ptr=ptr, order='F')
        >>> y
        tensor_space((2, 3), dtype=int).element(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        >>> y[0, 1] = -1
        >>> arr
        array([[ 1, -1,  3],
               [ 4,  5,  6]])
        """
        if order is not None and str(order).upper() not in ('C'):
            raise ValueError(f"Only row-major order supported ('C'), not '{order}'.")

        if inp is None and data_ptr is None:
            arr = torch.empty(self.shape, dtype=self._torch_dtype, device=self._torch_device)

            return self.element_type(self, arr)

        elif inp is None and data_ptr is not None:
            if order is None:
                raise ValueError('`order` cannot be None for element '
                                 'creation from pointer')

            ctype_array_def = ctypes.c_byte * self.nbytes
            as_ctype_array = ctype_array_def.from_address(data_ptr)
            as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
            arr = as_numpy_array.view(dtype=self._torch_dtype)
            arr = arr.reshape(self.shape, order=order)
            return self.element_type(self, torch.Tensor(arr))

        elif inp is not None and data_ptr is None:
            if inp in self and order is None:
                # Short-circuit for space elements and no enforced ordering
                return inp

            # TODO avoid copy when it's not necessary
            arr = torch.tensor(inp, dtype=self._torch_dtype, device=self._torch_device)

            if arr.shape != self.shape:
                raise ValueError('shape of `inp` not equal to space shape: '
                                 '{} != {}'.format(arr.shape, self.shape))
            return self.element_type(self, arr)

        else:
            raise TypeError('cannot provide both `inp` and `data_ptr`')

    def zero(self):
        """Return a tensor of all zeros.

        Examples
        --------
        >>> space = odl.rn(3)  # TODO adapt
        >>> x = space.zero()
        >>> x
        rn(3).element([ 0.,  0.,  0.])
        """
        return self.element(torch.zeros(self.shape, dtype=self._torch_dtype))

    def one(self):
        """Return a tensor of all ones.

        Examples
        --------
        >>> space = odl.rn(3)  # TODO adapt
        >>> x = space.one()
        >>> x
        rn(3).element([ 1.,  1.,  1.])
        """
        return self.element(torch.ones(self.shape, dtype=self._torch_dtype))

    @staticmethod
    def available_dtypes():
        """Return the set of data types available in this implementation.

        Notes
        -----
        Currently only a conservative selection of the types supported
        by Pytorch.
        """
        return [np.float32, np.float64, np.complex64, np.complex128]

    @staticmethod
    def default_dtype(field=None):
        """Return the default data type of this class for a given field.

        Parameters
        ----------
        field : `Field`, optional
            Set of numbers to be represented by a data type.
            Currently supported : `RealNumbers`, `ComplexNumbers`
            The default ``None`` means `RealNumbers`

        Returns
        -------
        dtype : `torch.dtype`
            Pytorch data type specifier. The returned defaults are:

                ``RealNumbers()`` : ``np.dtype('float64')``

                ``ComplexNumbers()`` : ``np.dtype('complex128')``
        """
        # Note that we're using the NumPy versions of the types, rather
        # than the equivalent Pytorch ones. This is for compatibility
        # with the rest of ODL, which is not aware of Pytorch.
        if field is None or field == RealNumbers():
            return np.float64
        elif field == ComplexNumbers():
            return np.complex128
        else:
            raise ValueError('no default data type defined for field {}'
                             ''.format(field))

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
        x1, x2 : `PytorchTensor`
            Summands in the linear combination.
        out : `PytorchTensor`
            Tensor to which the result is written.

        Examples
        --------
        >>> space = odl.rn(3)    # TODO adapt
        >>> x = space.element([0, 1, 1])
        >>> y = space.element([0, 0, 1])
        >>> out = space.element()
        >>> result = space.lincomb(1, x, 2, y, out)
        >>> result
        rn(3).element([ 0.,  1.,  3.])
        >>> result is out
        True
        """
        torch.add(input=a*x1.data, other=x2.data, alpha=b, out=out.data)

    def _dist(self, x1, x2):
        """Return the distance between ``x1`` and ``x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `PytorchTensor`
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
        return self.weighting.dist(x1, x2)

    def _norm(self, x):
        """Return the norm of ``x``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x : `PytorchTensor`
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
        return self.weighting.norm(x)

    def _inner(self, x1, x2):
        """Return the inner product of ``x1`` and ``x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `PytorchTensor`
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
        return self.weighting.inner(x1, x2)

    def _multiply(self, x1, x2, out):
        """Compute the entry-wise product ``out = x1 * x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `PytorchTensor`
            Factors in the product.
        out : `PytorchTensor`
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
        torch.mul(x1.data, x2.data, out=out.data)

    def _divide(self, x1, x2, out):
        """Compute the entry-wise quotient ``x1 / x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `PytorchTensor`
            Dividend and divisor in the quotient.
        out : `PytorchTensor`
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
        torch.div(x1.data, x2.data, out=out.data)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            True if ``other`` is an instance of ``type(self)``
            with the same `PytorchTensorSpace.shape`, `PytorchTensorSpace.dtype`
            and `PytorchTensorSpace.weighting`, otherwise False.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> same_space = odl.rn(3, exponent=2)
        >>> same_space == space
        True

        Different `shape`, `exponent` or `dtype` all result in different
        spaces:

        >>> diff_space = odl.rn((3, 4))
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(3, exponent=1)
        >>> diff_space == space
        False
        >>> diff_space = odl.rn(3, dtype='float32')
        >>> diff_space == space
        False
        >>> space == object
        False
        """
        if other is self:
            return True

        return (super(PytorchTensorSpace, self).__eq__(other) and
                self.weighting == other.weighting)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super(PytorchTensorSpace, self).__hash__(),
                     self.weighting))

    @property
    def byaxis(self):
        """Return the subspace defined along one or several dimensions.

        Examples
        --------
        Indexing with integers or slices:

        >>> space = odl.rn((2, 3, 4))  # TODO adapt
        >>> space.byaxis[0]
        rn(2)
        >>> space.byaxis[1:]
        rn((3, 4))

        Lists can be used to stack spaces arbitrarily:

        >>> space.byaxis[[2, 1, 2]]
        rn((4, 3, 4))
        """
        space = self

        class PytorchTensorSpacebyaxis(object):

            """Helper class for indexing by axis."""

            def __getitem__(self, indices):
                """Return ``self[indices]``."""
                try:
                    iter(indices)
                except TypeError:
                    newshape = space.shape[indices]
                else:
                    newshape = tuple(space.shape[i] for i in indices)

                if isinstance(space.weighting, ArrayWeighting):
                    new_array = np.asarray(space.weighting.array[indices])
                    weighting = PytorchTensorSpaceArrayWeighting(
                        new_array, space.weighting.exponent)
                else:
                    weighting = space.weighting

                return type(space)(newshape, space.dtype, weighting=weighting)

            def __repr__(self):
                """Return ``repr(self)``."""
                return repr(space) + '.byaxis'

        return PytorchTensorSpacebyaxis()

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim == 1:
            posargs = [self.size]
        else:
            posargs = [self.shape]

        if self.is_real:
            ctor_name = 'rn'  # TODO adapt
        elif self.is_complex:
            ctor_name = 'cn'
        else:
            ctor_name = 'tensor_space'

        if (ctor_name == 'tensor_space' or
                not is_numeric_dtype(self.dtype) or
                self.dtype != self.default_dtype(self.field)):
            optargs = [('dtype', dtype_str(self.dtype), '')]
            if self.dtype in (float, complex, int, bool):
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

    @property
    def element_type(self):
        """Type of elements in this space: `PytorchTensor`."""
        return PytorchTensor


class PytorchTensor(Tensor):

    """Representation of a `PytorchTensorSpace` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        Tensor.__init__(self, space)
        self.__data = data

    @property
    def data(self):
        """The `torch.Tensor` representing the data of ``self``."""
        return self.__data

    def asarray(self, out=None):
        """Extract the data of this array as a ``torch.Tensor``.

        This method is invoked when calling `torch.tensor` on this
        tensor.

        Parameters
        ----------
        out : `np.ndarray`, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype.

        Returns
        -------
        asarray : `torch.Tensor`
            Pytorch array with the same data type as ``self``. If
            ``out`` was given, the returned object is a reference
            to it.

        Examples
        --------
        >>> space = odl.rn(3, dtype='float32')  # TODO adapt
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
        newelem : `PytorchTensor`
            Version of this element with given data type.
        """
        return self.space.astype(dtype).element(self.data.astype(dtype))

    @property
    def data_ptr(self):
        """A raw pointer to the data container of ``self``.

        Examples
        --------
        >>> import ctypes
        >>> space = odl.tensor_space(3, dtype='uint16')   # TODO check example
        >>> x = space.element([1, 2, 3])
        >>> arr_type = ctypes.c_uint16 * 3  # C type "array of 3 uint16"
        >>> buffer = arr_type.from_address(x.data_ptr)
        >>> arr = np.frombuffer(buffer, dtype='uint16')
        >>> arr
        array([1, 2, 3], dtype=uint16)

        In-place modification via pointer:

        >>> arr[0] = 42    # TODO doubtful if this actually works
        >>> x
        tensor_space(3, dtype='uint16').element([42,  2,  3])
        """
        return self.data.data_ptr()

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            True if all entries of ``other`` are equal to this
            the entries of ``self``, False otherwise.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> x = space.element([1, 2, 3])
        >>> y = space.element([1, 2, 3])
        >>> x == y
        True

        >>> y = space.element([-1, 2, 3])
        >>> x == y
        False
        >>> x == object
        False

        Space membership matters:

        >>> space2 = odl.tensor_space(3, dtype='int64')
        >>> y = space2.element([1, 2, 3])
        >>> x == y or y == x
        False
        """
        if other is self:
            return True
        elif other not in self.space:
            return False
        else:
            return torch.equal(self.data, other.data)

    def copy(self):
        """Return an identical (deep) copy of this tensor.

        Parameters
        ----------
        None

        Returns
        -------
        copy : `PytorchTensor`
            The deep copy

        Examples
        --------
        >>> space = odl.rn(3)   # TODO adapt
        >>> x = space.element([1, 2, 3])
        >>> y = x.copy()
        >>> y == x
        True
        >>> y is x
        False
        """
        return self.space.element(self.data.clone())

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

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be accessed.

        Returns
        -------
        values : `PytorchTensorSpace.dtype` or `PytorchTensor`
            The value(s) at the given indices. Note that the returned
            object is a writable view into the original tensor, except
            for the case when ``indices`` is a list.

        Examples
        --------
        For one-dimensional spaces, indexing is as in linear arrays:

        >>> space = odl.rn(3)
        >>> x = space.element([1, 2, 3])
        >>> x[0]
        1.0
        >>> x[1:]
        rn(2).element([ 2.,  3.])

        In higher dimensions, the i-th index expression accesses the
        i-th axis:

        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x[0, 1]
        2.0
        >>> x[:, 1:]
        rn((2, 2)).element(
            [[ 2.,  3.],
             [ 5.,  6.]]
        )

        Slices can be assigned to, except if lists are used for indexing:

        >>> y = x[:, ::2]  # view into x
        >>> y[:] = -9
        >>> x
        rn((2, 3)).element(
            [[-9.,  2., -9.],
             [-9.,  5., -9.]]
        )
        >>> y = x[[0, 1], [1, 2]]  # not a view, won't modify x
        >>> y
        rn(2).element([ 2., -9.])
        >>> y[:] = 0
        >>> x
        rn((2, 3)).element(
            [[-9.,  2., -9.],
             [-9.,  5., -9.]]
        )
        """
        # Lazy implementation: index the array and deal with it
        if isinstance(indices, PytorchTensor):
            indices = indices.data
        arr = self.data[indices]

        if arr.shape == ():  # scalar
            if self.space.field is not None:
                return self.space.field.element(arr)
            else:
                return arr
        else:
            if is_numeric_dtype(self.dtype):
                weighting = self.space.weighting
            else:
                weighting = None
            space = type(self.space)(
                arr.shape, dtype=self.dtype, exponent=self.space.exponent,
                weighting=weighting)
            return space.element(arr)

    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be written to.
        values : scalar, array-like or `PytorchTensor`
            The value(s) that are to be assigned.

            If ``index`` is an integer, ``value`` must be a scalar.

            If ``index`` is a slice or a sequence of slices, ``value``
            must be broadcastable to the shape of the slice.

        Examples
        --------
        For 1d spaces, entries can be set with scalars or sequences of
        correct shape:

        >>> space = odl.rn(3)
        >>> x = space.element([1, 2, 3])
        >>> x[0] = -1
        >>> x[1:] = (0, 1)
        >>> x
        rn(3).element([-1.,  0.,  1.])

        It is also possible to use tensors of other spaces for
        casting and assignment:

        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x[0, 1] = -1
        >>> x
        rn((2, 3)).element(
            [[ 1., -1.,  3.],
             [ 4.,  5.,  6.]]
        )
        >>> short_space = odl.tensor_space((2, 2), dtype='short')
        >>> y = short_space.element([[-1, 2],
        ...                          [0, 0]])
        >>> x[:, :2] = y
        >>> x
        rn((2, 3)).element(
            [[-1.,  2.,  3.],
             [ 0.,  0.,  6.]]
        )

        The Numpy assignment and broadcasting rules apply:

        >>> x[:] = np.array([[0, 0, 0],
        ...                  [1, 1, 1]])
        >>> x
        rn((2, 3)).element(
            [[ 0.,  0.,  0.],
             [ 1.,  1.,  1.]]
        )
        >>> x[:, 1:] = [7, 8]
        >>> x
        rn((2, 3)).element(
            [[ 0.,  7.,  8.],
             [ 1.,  7.,  8.]]
        )
        >>> x[:, ::2] = -2.
        >>> x
        rn((2, 3)).element(
            [[-2.,  7., -2.],
             [-2.,  7., -2.]]
        )
        """
        if isinstance(indices, type(self)):
            indices = indices.data
        if isinstance(values, type(self)):
            values = values.data

        self.data[indices] = values

    @property
    def real(self):
        """Real part of ``self``.

        Returns
        -------
        real : `PytorchTensor`
            Real part of this element as a member of a
            `PytorchTensorSpace` with corresponding real data type.

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

    @property
    def imag(self):
        """Imaginary part of ``self``.

        Returns
        -------
        imag : `PytorchTensor`
            Imaginary part this element as an element of a
            `PytorchTensorSpace` with real data type.

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

    def conj(self, out=None):
        """Return the complex conjugate of ``self``.

        Parameters
        ----------
        out : `PytorchTensor`, optional
            Element to which the complex conjugate is written.
            Must be an element of ``self.space``.

        Returns
        -------
        out : `PytorchTensor`
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

    def __ipow__(self, other):
        """Return ``self **= other``."""
        try:
            if other == int(other):
                return super(PytorchTensor, self).__ipow__(other)
        except TypeError:
            pass

        torch.pow(self.data, other, out=self.data)
        return self

    def __rmul__(self, other):
        result = self.space.element(other * self.data)
        return result

    def __int__(self):
        """Return ``int(self)``."""
        return int(self.data)

    def __long__(self):
        """Return ``long(self)``.

        This method is only useful in Python 2.
        """
        return long(self.data)

    def __float__(self):
        """Return ``float(self)``."""
        return float(self.data)

    def __complex__(self):
        """Return ``complex(self)``."""
        if self.size != 1:
            raise TypeError('only size-1 tensors can be converted to '
                            'Python scalars')
        return complex(self.data.ravel()[0])




def _weighting(weights, exponent):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weights) or weights.shape == ():
        weighting = PytorchTensorSpaceConstWeighting(weights, exponent)
    elif weights is None:
        weighting = PytorchTensorSpaceConstWeighting(1.0, exponent)
    else:  # last possibility: make an array
        arr = torch.tensor(weights)
        weighting = PytorchTensorSpaceArrayWeighting(arr, exponent)
    return weighting


def pytorch_weighted_inner(weights):
    """Weighted inner product on `TensorSpace`'s as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the inner product. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting vector.

    Returns
    -------
    inner : `callable`
        Inner product function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See Also
    --------
    PytorchTensorSpaceConstWeighting
    PytorchTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=2.0).inner


def pytorch_weighted_norm(weights, exponent=2.0):
    """Weighted norm on `TensorSpace`'s as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the norm. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting vector.
    exponent : positive `float`
        Exponent of the norm.

    Returns
    -------
    norm : `callable`
        Norm function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See Also
    --------
    PytorchTensorSpaceConstWeighting
    PytorchTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=exponent).norm


def pytorch_weighted_dist(weights, exponent=2.0):
    """Weighted distance on `TensorSpace`'s as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the distance. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting vector.
    exponent : positive `float`
        Exponent of the norm.

    Returns
    -------
    dist : `callable`
        Distance function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See Also
    --------
    PytorchTensorSpaceConstWeighting
    PytorchTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=exponent).dist


def _norm_default(x):
    """Default Euclidean norm implementation."""

    return x.data.norm(p=2)


def _pnorm_default(x, p):
    """Default p-norm implementation."""
    return x.data.norm(p=p)


def _pnorm_diagweight(x, p, w):
    """Diagonally weighted p-norm implementation."""
    xp = torch.abs(x.data)
    if p == float('inf'):
        xp *= w
        return torch.max(xp)
    else:
        torch.pow(xp, p, out=xp)
        xp *= w
        return torch.sum(xp) ** (1 / p)


def _inner_default(x1, x2):
    """Default Euclidean inner product implementation."""

    if is_real_dtype(x1.dtype):
        return torch.dot(x1.data, x2.data)
    else:
        # x2 as first argument because we want linearity in x1
        return torch.vdot(x2.data, x1.data)



class PytorchTensorSpaceArrayWeighting(ArrayWeighting):

    """Weighting of a `PytorchTensorSpace` by an array.

    This class defines a weighting by an array that has the same shape
    as the tensor space. Since the space is not known to this class,
    no checks of shape or data type are performed.
    See ``Notes`` for mathematical details.
    """

    def __init__(self, array, exponent=2.0):
        r"""Initialize a new instance.

        Parameters
        ----------
        array : `array-like`, one-dim.
            Weighting array of the inner product, norm and distance.
            All its entries must be positive, however this is not
            verified during initialization.
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, no inner
            product is defined.

        Notes
        -----
        - For exponent 2.0, a new weighted inner product with array
          :math:`W` is defined as

          .. math::
              \langle A, B\rangle_W :=
              \langle W \odot A, B\rangle =
              \langle w \odot a, b\rangle =
              b^{\mathrm{H}} (w \odot a),

          where :math:`a, b, w` are the "flattened" counterparts of
          tensors :math:`A, B, W`, respectively, :math:`b^{\mathrm{H}}`
          stands for transposed complex conjugate and :math:`w \odot a`
          for element-wise multiplication.

        - For other exponents, only norm and dist are defined. In the
          case of exponent :math:`\infty`, the weighted norm is

          .. math::
              \| A\|_{W, \infty} :=
              \| W \odot A\|_{\infty} =
              \| w \odot a\|_{\infty},

          otherwise it is (using point-wise exponentiation)

          .. math::
              \| A\|_{W, p} :=
              \| W^{1/p} \odot A\|_{p} =
              \| w^{1/p} \odot a\|_{\infty}.

        - Note that this definition does **not** fulfill the limit
          property in :math:`p`, i.e.

          .. math::
              \| A\|_{W, p} \not\to
              \| A\|_{W, \infty} \quad (p \to \infty)

          unless all weights are equal to 1.

        - The array :math:`W` may only have positive entries, otherwise
          it does not define an inner product or norm, respectively. This
          is not checked during initialization.
        """
        if isinstance(array, PytorchTensor):
            array = array.data
        elif not isinstance(array, torch.Tensor):
            array = torch.tensor(array)
        super(PytorchTensorSpaceArrayWeighting, self).__init__(
            array, impl='pytorch', exponent=exponent)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), hash(self.array), self.exponent))

    def inner(self, x1, x2):
        """Return the weighted inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `PytorchTensor`
            Tensors whose inner product is calculated.

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors.
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))
        else:
            inner = _inner_default(x1 * self.array, x2)
            if is_real_dtype(x1.dtype):
                return float(inner)
            else:
                return complex(inner)

    def norm(self, x):
        """Return the weighted norm of ``x``.

        Parameters
        ----------
        x : `PytorchTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the provided tensor.
        """
        if self.exponent == 2.0:
            norm_squared = self.inner(x, x).real  # TODO: optimize?!
            if norm_squared < 0:
                norm_squared = 0.0  # Compensate for numerical error
            return float(np.sqrt(norm_squared))
        else:
            return float(_pnorm_diagweight(x, self.exponent, self.array))


class PytorchTensorSpaceConstWeighting(ConstWeighting):

    """Weighting of a `PytorchTensorSpace` by a constant.

    See ``Notes`` for mathematical details.
    """

    def __init__(self, const, exponent=2.0):
        r"""Initialize a new instance.

        Parameters
        ----------
        const : positive float
            Weighting constant of the inner product, norm and distance.
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.

        Notes
        -----
        - For exponent 2.0, a new weighted inner product with constant
          :math:`c` is defined as

          .. math::
              \langle a, b\rangle_c :=
              c \, \langle a, b\rangle_c =
              c \, b^{\mathrm{H}} a,

          where :math:`b^{\mathrm{H}}` standing for transposed complex
          conjugate.

        - For other exponents, only norm and dist are defined. In the
          case of exponent :math:`\infty`, the weighted norm is defined
          as

          .. math::
              \| a \|_{c, \infty} :=
              c\, \| a \|_{\infty},

          otherwise it is

          .. math::
              \| a \|_{c, p} :=
              c^{1/p}\, \| a \|_{p}.

        - Note that this definition does **not** fulfill the limit
          property in :math:`p`, i.e.

          .. math::
              \| a\|_{c, p} \not\to
              \| a \|_{c, \infty} \quad (p \to \infty)

          unless :math:`c = 1`.

        - The constant must be positive, otherwise it does not define an
          inner product or norm, respectively.
        """
        super(PytorchTensorSpaceConstWeighting, self).__init__(
            const, impl='pytorch', exponent=exponent)

    def inner(self, x1, x2):
        """Return the weighted inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `PytorchTensor`
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
            inner = self.const * _inner_default(x1, x2)
            if x1.space.field is None:
                return inner
            else:
                return x1.space.field.element(inner)

    def norm(self, x):
        """Return the weighted norm of ``x``.

        Parameters
        ----------
        x1 : `PytorchTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the tensor.
        """
        if self.exponent == 2.0:
            return float(np.sqrt(self.const) * _norm_default(x))
        elif self.exponent == float('inf'):
            return float(self.const * _pnorm_default(x, self.exponent))
        else:
            return float((self.const ** (1 / self.exponent) *
                          _pnorm_default(x, self.exponent)))

    def dist(self, x1, x2):
        """Return the weighted distance between ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `PytorchTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the tensors.
        """
        if self.exponent == 2.0:
            return float(np.sqrt(self.const) * _norm_default(x1 - x2))
        elif self.exponent == float('inf'):
            return float(self.const * _pnorm_default(x1 - x2, self.exponent))
        else:
            return float((self.const ** (1 / self.exponent) *
                          _pnorm_default(x1 - x2, self.exponent)))


class PytorchTensorSpaceCustomInner(CustomInner):

    """Class for handling a user-specified inner product."""

    def __init__(self, inner):
        """Initialize a new instance.

        Parameters
        ----------
        inner : `callable`
            The inner product implementation. It must accept two
            `Tensor` arguments, return an element from their space's
            field (real or complex number) and satisfy the following
            conditions for all vectors ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``
        """
        super(PytorchTensorSpaceCustomInner, self).__init__(inner, impl='pytorch')


class PytorchTensorSpaceCustomNorm(CustomNorm):

    """Class for handling a user-specified norm.

    Note that this removes ``inner``.
    """

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : `callable`
            The norm implementation. It must accept a `Tensor`
            argument, return a `float` and satisfy the following
            conditions for all any two elements ``x, y`` and scalars
            ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``
        """
        super(PytorchTensorSpaceCustomNorm, self).__init__(norm, impl='pytorch')


class PytorchTensorSpaceCustomDist(CustomDist):

    """Class for handling a user-specified distance in `TensorSpace`.

    Note that this removes ``inner`` and ``norm``.
    """

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : `callable`
            The distance function defining a metric on `TensorSpace`. It
            must accept two `Tensor` arguments, return a `float` and
            fulfill the following mathematical conditions for any three
            elements ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``
        """
        super(PytorchTensorSpaceCustomDist, self).__init__(dist, impl='pytorch')


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
