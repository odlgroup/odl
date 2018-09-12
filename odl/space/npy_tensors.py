# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""NumPy implementation of tensor spaces."""

from __future__ import print_function, division, absolute_import
from builtins import object
from future.utils import native
from future.moves.itertools import zip_longest
import ctypes
from functools import partial
import numpy as np

from odl.set.sets import RealNumbers, ComplexNumbers
from odl.set.space import LinearSpaceTypeError
from odl.space.base_tensors import TensorSpace, Tensor
from odl.space.weighting import (
    Weighting, ArrayWeighting, ConstWeighting, PerAxisWeighting,
    CustomInner, CustomNorm, CustomDist)
from odl.util import (
    dtype_str, signature_string, is_real_dtype, is_numeric_dtype, array_str,
    array_hash, indent, fast_1d_tensor_mult, writable_array, is_floating_dtype,
    simulate_slicing, normalized_index_expression)


__all__ = ('NumpyTensorSpace',)


_BLAS_DTYPES = (np.dtype('float32'), np.dtype('float64'),
                np.dtype('complex64'), np.dtype('complex128'))

# Define size thresholds to switch implementations
THRESHOLD_SMALL = 100
THRESHOLD_MEDIUM = 50000


class NumpyTensorSpace(TensorSpace):

    """Set of tensors of arbitrary data type, implemented with NumPy.

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

    This class is implemented using `numpy.ndarray`'s as back-end.

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
            way the `numpy.dtype` function understands, e.g.
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
        weighting : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weighting``:

            - ``None``: No weighting, i.e. weighting with ``1.0`` (default).
            - ``float``: Weighting by a constant.
            - sequence of length ``ndim``: Separate weighting per axis.
              Entries can be constants or 1D arrays.
            - array-like: Pointwise weighting by an array of the same
              ``shape`` as this space.
            - `Weighting`: Use this weighting as-is. Compatibility
              with this space is not checked during init.

            This option cannot be combined with ``dist``,
            ``norm`` or ``inner``. It also cannot be used in case of
            non-numeric ``dtype``.

        dist : callable, optional
            Distance function defining a metric on the space.
            It must accept two `NumpyTensor` arguments and return
            a non-negative real number. See ``Notes`` for
            mathematical requirements.

            By default, ``dist(x, y)`` is calculated as ``norm(x - y)``.

            This option cannot be combined with ``weight``,
            ``norm`` or ``inner``. It also cannot be used in case of
            non-numeric ``dtype``.

        norm : callable, optional
            The norm implementation. It must accept a
            `NumpyTensor` argument, return a non-negative real number.
            See ``Notes`` for mathematical requirements.

            By default, ``norm(x)`` is calculated as ``inner(x, x)``.

            This option cannot be combined with ``weight``,
            ``dist`` or ``inner``. It also cannot be used in case of
            non-numeric ``dtype``.

        inner : callable, optional
            The inner product implementation. It must accept two
            `NumpyTensor` arguments and return an element of the field
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
        - A distance function or metric on a space :math:`X`
          is a mapping
          :math:`d:X \times X \to \mathbb{R}`
          satisfying the following conditions for all space elements
          :math:`x, y, z`:

          * :math:`d(x, y) \geq 0`,
          * :math:`d(x, y) = 0 \Leftrightarrow x = y`,
          * :math:`d(x, y) = d(y, x)`,
          * :math:`d(x, y) \leq d(x, z) + d(z, y)`.

        - A norm on a space :math:`X` is a mapping
          :math:`\| \cdot \|:X \to \mathbb{R}`
          satisfying the following conditions for all
          space elements :math:`x, y`: and scalars :math:`s`:

          * :math:`\| x\| \geq 0`,
          * :math:`\| x\| = 0 \Leftrightarrow x = 0`,
          * :math:`\| sx\| = |s| \cdot \| x \|`,
          * :math:`\| x+y\| \leq \| x\| +
            \| y\|`.

        - An inner product on a space :math:`X` over a field
          :math:`\mathbb{F} = \mathbb{R}` or :math:`\mathbb{C}` is a
          mapping
          :math:`\langle\cdot, \cdot\rangle: X \times
          X \to \mathbb{F}`
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

        >>> space = NumpyTensorSpace(3, float)
        >>> space
        rn(3)
        >>> space.shape
        (3,)
        >>> space.dtype
        dtype('float64')

        A more convenient way is to use factory functions:

        >>> space = odl.rn(3, weighting=[1, 2, 3])
        >>> space
        rn(3, weighting=[1, 2, 3])
        >>> space = odl.tensor_space((2, 3), dtype=int)
        >>> space
        tensor_space((2, 3), dtype=int)
        """
        super(NumpyTensorSpace, self).__init__(shape, dtype)
        # Check if dtype is supported; to include variable-size dtypes
        # (mostly string types) we check `dtype.char`
        if np.dtype(self.dtype.char) not in self.available_dtypes():
            raise ValueError('`dtype` {!r} not supported'
                             ''.format(dtype_str(dtype)))

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weighting = kwargs.pop('weighting', None)
        exponent = kwargs.pop('exponent', getattr(weighting, 'exponent', 2.0))

        if (not is_numeric_dtype(self.dtype) and
                any(x is not None for x in (dist, norm, inner, weighting))):
            raise ValueError('cannot use any of `weighting`, `dist`, `norm` '
                             'or `inner` for non-numeric `dtype` {}'
                             ''.format(dtype))
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
                if weighting.impl != 'numpy':
                    raise ValueError("`weighting.impl` must be 'numpy', "
                                     '`got {!r}'.format(weighting.impl))
                if weighting.exponent != exponent:
                    raise ValueError('`weighting.exponent` conflicts with '
                                     '`exponent`: {} != {}'
                                     ''.format(weighting.exponent, exponent))
                self.__weighting = weighting

            elif np.isscalar(weighting):
                if self.ndim == 1:
                    # Prefer per-axis weighting if possible, it behaves better
                    self.__weighting = NumpyTensorSpacePerAxisWeighting(
                        [weighting], exponent)
                else:
                    self.__weighting = NumpyTensorSpaceConstWeighting(
                        weighting, exponent)

            elif len(weighting) == self.ndim:
                self.__weighting = NumpyTensorSpacePerAxisWeighting(
                    weighting, exponent)

            else:
                array = np.asarray(weighting)
                if array.dtype == object:
                    raise ValueError(
                        'invalid `weighting` provided; valid inputs are '
                        '`None`, constants, sequences of length `ndim` '
                        "and array-like objects of this space's `shape`")
                self.__weighting = NumpyTensorSpaceArrayWeighting(
                    array, exponent)

            # Check (afterwards) that the weighting input was sane
            if isinstance(self.__weighting, NumpyTensorSpaceArrayWeighting):
                if not np.can_cast(self.__weighting.array.dtype, self.dtype):
                    raise ValueError(
                        'cannot cast from `weighting` data type {} to '
                        'the space `dtype` {}'
                        ''.format(dtype_str(self.__weighting.array.dtype),
                                  dtype_str(self.dtype)))
                if self.__weighting.array.shape != self.shape:
                    raise ValueError('array-like weights must have same '
                                     'shape {} as this space, got {}'
                                     ''.format(self.shape,
                                               self.__weighting.array.shape))

            elif isinstance(self.__weighting,
                            NumpyTensorSpacePerAxisWeighting):
                if len(self.__weighting.factors) != self.ndim:
                    raise ValueError(
                        'per-axis weighting must have {} (=ndim) factors, '
                        'got {}'.format(self.ndim,
                                        len(self.__weighting.factors)))

        elif dist is not None:
            self.__weighting = NumpyTensorSpaceCustomDist(dist)
        elif norm is not None:
            self.__weighting = NumpyTensorSpaceCustomNorm(norm)
        elif inner is not None:
            self.__weighting = NumpyTensorSpaceCustomInner(inner)
        else:
            # No weighting, i.e., weighting with constant 1.0
            self.__weighting = NumpyTensorSpaceConstWeighting(1.0, exponent)

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unknown keyword arguments {}'.format(kwargs))

    @property
    def impl(self):
        """Name of the implementation back-end: ``'numpy'``."""
        return 'numpy'

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
            isinstance(self.weighting, NumpyTensorSpaceConstWeighting) and
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
            The new element will use ``order`` as storage order if
            provided, otherwise `default_order`.

            Otherwise, a copy is avoided whenever possible. This requires
            correct `shape` and `dtype`, and if ``order`` is provided,
            also contiguousness in that ordering. If any of these
            conditions is not met, a copy is made.

        data_ptr : int, optional
            Pointer to the start memory address of a contiguous Numpy array
            or an equivalent raw container with the same total number of
            bytes. For this option, ``order`` must be either ``'C'`` or
            ``'F'``.
            The option is also mutually exclusive with ``inp``.
        order : {None, 'C', 'F'}, optional
            Storage order of the returned element. For ``'C'`` and ``'F'``,
            contiguous memory in the respective ordering is enforced.
            The default ``None`` enforces no contiguousness.

        Returns
        -------
        element : `NumpyTensor`
            The new element, created from ``inp`` or from scratch.

        Examples
        --------
        Without arguments, an uninitialized element is created. With an
        array-like input, the element can be initialized:

        >>> space = odl.rn(3)
        >>> empty = space.element()
        >>> empty.shape
        (3,)
        >>> empty.space
        rn(3)
        >>> x = space.element([1, 2, 3])
        >>> x
        rn(3).element([ 1.,  2.,  3.])

        If the input already is a `numpy.ndarray` of correct `dtype`, it
        will merely be wrapped, i.e., both array and space element access
        the same memory, such that mutations will affect both:

        >>> arr = np.array([1, 2, 3], dtype=float)
        >>> elem = odl.rn(3).element(arr)
        >>> elem[0] = 0
        >>> elem
        rn(3).element([ 0.,  2.,  3.])
        >>> arr
        array([ 0.,  2.,  3.])

        Elements can also be constructed from a data pointer, resulting
        again in shared memory:

        >>> int_space = odl.tensor_space((2, 3), dtype=int)
        >>> arr = np.array([[1, 2, 3],
        ...                 [4, 5, 6]], dtype=int, order='F')
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
        if order is not None and str(order).upper() not in ('C', 'F'):
            raise ValueError("`order` {!r} not understood".format(order))

        if inp is None and data_ptr is None:
            if order is None:
                arr = np.empty(self.shape, dtype=self.dtype,
                               order=self.default_order)
            else:
                arr = np.empty(self.shape, dtype=self.dtype, order=order)

            return self.element_type(self, arr)

        elif inp is None and data_ptr is not None:
            if order is None:
                raise ValueError('`order` cannot be None for element '
                                 'creation from pointer')

            ctype_array_def = ctypes.c_byte * self.nbytes
            as_ctype_array = ctype_array_def.from_address(data_ptr)
            as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
            arr = as_numpy_array.view(dtype=self.dtype)
            arr = arr.reshape(self.shape, order=order)
            return self.element_type(self, arr)

        elif inp is not None and data_ptr is None:
            if inp in self and order is None:
                # Short-circuit for space elements and no enforced ordering
                return inp

            # Try to not copy but require dtype and order if given
            # (`order=None` is ok as np.array argument)
            arr = np.array(inp, copy=False, dtype=self.dtype, ndmin=self.ndim,
                           order=order)
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

    def _astype(self, dtype):
        """Internal helper for `astype`.

        Subclasses with different constructor signature should override this
        method.
        """
        kwargs = {}
        dtype = np.dtype(dtype)

        # Use weighting only for floating-point types, otherwise, e.g.,
        # `space.astype(bool)` would fail
        if is_floating_dtype(dtype) and dtype.shape == ():
            # Standard case, basically pass-through
            weighting = getattr(self, 'weighting', None)
            if weighting is not None:
                kwargs['weighting'] = weighting

        elif is_floating_dtype(dtype) and dtype.shape != ():
            # Got nontrivial `dtype.shape`, make new axes accordingly
            weighting_slc = (
                (None,) * len(dtype.shape) + (slice(None),) * self.ndim
            )
            weighting = slice_weighting(
                self.weighting, self.shape, weighting_slc)
            kwargs['weighting'] = weighting

        return type(self)(dtype.shape + self.shape, dtype=dtype.base, **kwargs)

    def zero(self):
        """Return a tensor of all zeros.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> x = space.zero()
        >>> x
        rn(3).element([ 0.,  0.,  0.])
        """
        return self.element(np.zeros(self.shape, dtype=self.dtype,
                                     order=self.default_order))

    def one(self):
        """Return a tensor of all ones.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> x = space.one()
        >>> x
        rn(3).element([ 1.,  1.,  1.])
        """
        return self.element(np.ones(self.shape, dtype=self.dtype,
                                    order=self.default_order))

    @staticmethod
    def available_dtypes():
        """Return the set of data types available in this implementation.

        Returns
        -------
        available_dtypes : set

        Notes
        -----
        This set includes all Numpy dtypes, except for ``object`` and
        ``void``. See ``numpy.sctypes`` for more information.

        The available dtypes can depend on the operating system and the
        Numpy version.
        """
        all_dtypes = []
        for lst in np.sctypes.values():
            for dtype in lst:
                if dtype not in (np.object, np.void):
                    all_dtypes.append(np.dtype(dtype))
        # Need to add these manually since np.sctypes['others'] will only
        # contain one of them (depending on Python version)
        all_dtypes.extend([np.dtype('S'), np.dtype('U')])
        return tuple(sorted(set(all_dtypes)))

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
        dtype : `numpy.dtype`
            Numpy data type specifier. The returned defaults are:

                ``RealNumbers()`` : ``np.dtype('float64')``

                ``ComplexNumbers()`` : ``np.dtype('complex128')``
        """
        if field is None or field == RealNumbers():
            return np.dtype('float64')
        elif field == ComplexNumbers():
            return np.dtype('complex128')
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
        _lincomb_impl(a, x1.data, b, x2.data, out.data)

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
        return self.weighting.dist(x1, x2)

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
        return self.weighting.norm(x)

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
        return self.weighting.inner(x1, x2)

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
        np.multiply(x1.data, x2.data, out=out.data)

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
        np.divide(x1.data, x2.data, out=out.data)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            True if ``other`` is an instance of ``type(self)``
            with the same `NumpyTensorSpace.shape`, `NumpyTensorSpace.dtype`
            and `NumpyTensorSpace.weighting`, otherwise False.

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

        return (super(NumpyTensorSpace, self).__eq__(other) and
                self.weighting == other.weighting)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super(NumpyTensorSpace, self).__hash__(),
                     self.weighting))

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        For all supported cases, indexing is implemented such that for an
        element ``x in space``, ::

            x[indices] in space[indices]

        Space indexing works with

        - integers,
        - `slice` objects,
        - index arrays, i.e., 1D array-like objects containing integers,

        and combinations of the above. It does not work with boolean arrays
        or more advanced "fancy indexing".

        .. note::
            This method is a default implementation that propagates only
            ``shape`` and ``dtype``. Subclasses with more properties
            need to override the method.

        Examples
        --------
        A single integer slices along the first axis (index does not
        matter as long as it lies within the bounds):

        >>> rn = odl.rn((3, 4, 5, 6))
        >>> rn[0]
        rn((4, 5, 6))
        >>> rn[2]
        rn((4, 5, 6))

        Multiple indices slice into corresponding axes from the left:

        >>> rn[0, 0]
        rn((5, 6))
        >>> rn[0, 1, 1]
        rn(6)

        Ellipsis (``...``) and ``slice(None)`` (``:``) can be used to keep
        one or several axes intact:

        >>> rn[0, :, 1, :]
        rn((4, 6))
        >>> rn[..., 0, 0]
        rn((3, 4))

        With slices, parts of axes can be selected:

        >>> rn[0, :3, 1:4, ::2]
        rn((3, 3, 3))

        Array-like objects (must all have the same 1D shape) of integers are
        treated as follows: if their common length is ``n``, a new axis
        of length ``n`` is created at the position of the leftmost index
        array, and all index array axes are collapsed (Note: this is
        not so useful for spaces, more so for elements):

        >>> rn[0, 0, [0, 1, 0, 2], :]
        rn((4, 6))
        >>> rn[:, [1, 1], [0, 2], :]
        rn((3, 2, 6))
        >>> rn[[2, 0], [3, 3], [0, 1], [5, 2]]
        rn(2)
        """
        new_shape, removed_axes, _, _ = simulate_slicing(self.shape, indices)
        weighting = slice_weighting(self.weighting, self.shape, indices)
        return type(self)(shape=new_shape, dtype=self.dtype,
                          weighting=weighting, exponent=self.exponent)

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

        class NpyTensorSpaceByAxis(object):

            """Helper class for indexing by axis."""

            def __getitem__(self, indices):
                """Return ``self[indices]``."""
                try:
                    iter(indices)
                except TypeError:
                    # Integer, slice or Ellipsis
                    indices = list(range(space.ndim))[indices]
                    if not isinstance(indices, list):
                        indices = [indices]
                else:
                    indices = [int(i) for i in indices]

                new_shape = tuple(space.shape[i] for i in indices)
                new_weighting = slice_weighting_by_axis(space.weighting,
                                                        space.shape, indices)
                return type(space)(new_shape, space.dtype,
                                   weighting=new_weighting,
                                   exponent=space.exponent)

            def __repr__(self):
                """Return ``repr(self)``."""
                return repr(space) + '.byaxis'

        return NpyTensorSpaceByAxis()

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
        """Type of elements in this space: `NumpyTensor`."""
        return NumpyTensor


class NumpyTensor(Tensor):

    """Representation of a `NumpyTensorSpace` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        super(Tensor, self).__init__(space)
        self.__data = data

    @property
    def data(self):
        """The `numpy.ndarray` representing the data of ``self``."""
        return self.__data

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
        dtype = np.dtype(dtype)
        if dtype.shape != ():
            raise ValueError('`dtype` with shape not allowed')
        return self.space.astype(dtype).element(self.data.astype(dtype))

    @property
    def data_ptr(self):
        """A raw pointer to the data container of ``self``.

        Examples
        --------
        >>> import ctypes
        >>> space = odl.tensor_space(3, dtype='uint16')
        >>> x = space.element([1, 2, 3])
        >>> arr_type = ctypes.c_uint16 * 3  # C type "array of 3 uint16"
        >>> buffer = arr_type.from_address(x.data_ptr)
        >>> arr = np.frombuffer(buffer, dtype='uint16')
        >>> arr
        array([1, 2, 3], dtype=uint16)

        In-place modification via pointer:

        >>> arr[0] = 42
        >>> x
        tensor_space(3, dtype='uint16').element([42,  2,  3])
        """
        return self.data.ctypes.data

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
            return np.array_equal(self.data, other.data)

    def copy(self):
        """Return an identical (deep) copy of this tensor.

        Parameters
        ----------
        None

        Returns
        -------
        copy : `NumpyTensor`
            The deep copy

        Examples
        --------
        >>> space = odl.rn(3)
        >>> x = space.element([1, 2, 3])
        >>> y = x.copy()
        >>> y == x
        True
        >>> y is x
        False
        """
        return self.space.element(self.data.copy())

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
        values : `NumpyTensorSpace.dtype` or `NumpyTensor`
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

        More advanced indexing:

        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x[[0, 0, 1], [2, 1, 0]]  # entries (0, 2), (0, 1) and (1, 0)
        rn(3).element([ 3.,  2.,  4.])
        >>> bool_elem = x.ufuncs.less(3)
        >>> x[bool_elem]
        rn(2).element([ 1.,  2.])
        """
        if isinstance(indices, NumpyTensor):
            indices = indices.data
        res_arr = self.data[indices]

        if np.isscalar(res_arr):
            if self.space.field is not None:
                return self.space.field.element(res_arr)
            else:
                return res_arr
        else:
            # If possible, ensure that
            # `self.space[indices].shape == self[indices].shape`
            try:
                res_space = self.space[indices]
            except TypeError:
                # No weighting
                res_space = type(self.space)(
                    res_arr.shape, dtype=self.dtype,
                    exponent=self.space.exponent)
            return res_space.element(res_arr)

    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be written to.
        values : scalar, array-like or `NumpyTensor`
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

    def __ipow__(self, other):
        """Return ``self **= other``."""
        try:
            if other == int(other):
                return super(NumpyTensor, self).__ipow__(other)
        except TypeError:
            pass

        np.power(self.data, other, out=self.data)
        return self

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

        See the `corresponding NEP`_ and the `interface documentation`_
        for further details. See also the `general documentation on
        Numpy ufuncs`_.

        .. note::
            This basic implementation casts inputs and
            outputs to Numpy arrays and evaluates ``ufunc`` on those.
            For `numpy.ndarray` based data storage, this incurs no
            significant overhead compared to direct usage of Numpy arrays.

            For other (in particular non-local) implementations, e.g.,
            GPU arrays or distributed memory, overhead is significant due
            to copies to CPU main memory. In those classes, the
            ``__array_ufunc__`` mechanism should be overridden to use
            native implementations if possible.

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
        ufunc_result : `Tensor`, `numpy.ndarray` or tuple
            Result of the ufunc evaluation. If no ``out`` keyword argument
            was given, the result is a `Tensor` or a tuple
            of such, depending on the number of outputs of ``ufunc``.
            If ``out`` was provided, the returned object or tuple entries
            refer(s) to ``out``.

        Examples
        --------
        We apply `numpy.add` to ODL tensors:

        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, -2, -3])
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        rn(3).element([ 0.,  0.,  0.])
        >>> np.add(x, y)  # same mechanism for Numpy >= 1.13
        rn(3).element([ 0.,  0.,  0.])

        As ``out``, a Numpy array or an ODL tensor can be given (wrapped
        in a sequence):

        >>> out = r3.element()
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out,))
        >>> out
        rn(3).element([ 0.,  0.,  0.])
        >>> res is out
        True
        >>> out_arr = np.empty(3)
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out_arr,))
        >>> out_arr
        array([ 0.,  0.,  0.])
        >>> res is out_arr
        True

        With multiple dimensions:

        >>> r23 = odl.rn((2, 3))
        >>> x = y = r23.one()
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        rn((2, 3)).element(
            [[ 2.,  2.,  2.],
             [ 2.,  2.,  2.]]
        )

        The ``ufunc.accumulate`` method retains the original `shape` and
        `dtype`. The latter can be changed with the ``dtype`` parameter:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'accumulate', x)
        rn(3).element([ 1.,  3.,  6.])
        >>> np.add.accumulate(x)  # same mechanism for Numpy >= 1.13
        rn(3).element([ 1.,  3.,  6.])
        >>> x.__array_ufunc__(np.add, 'accumulate', x, dtype=complex)
        cn(3).element([ 1.+0.j,  3.+0.j,  6.+0.j])

        For multi-dimensional tensors, an optional ``axis`` parameter
        can be provided:

        >>> z = r23.one()
        >>> z.__array_ufunc__(np.add, 'accumulate', z, axis=1)
        rn((2, 3)).element(
            [[ 1.,  2.,  3.],
             [ 1.,  2.,  3.]]
        )

        The ``ufunc.at`` method operates in-place. Here we add the second
        operand ``[5, 10]`` to ``x`` at indices ``[0, 2]``:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'at', x, [0, 2], [5, 10])
        >>> x
        rn(3).element([  6.,   2.,  13.])

        For outer-product-type operations, i.e., operations where the result
        shape is the sum of the individual shapes, the ``ufunc.outer``
        method can be used:

        >>> x = odl.rn(2).element([0, 3])
        >>> y = odl.rn(3).element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'outer', x, y)
        rn((2, 3)).element(
            [[ 1.,  2.,  3.],
             [ 4.,  5.,  6.]]
        )
        >>> y.__array_ufunc__(np.add, 'outer', y, x)
        rn((3, 2)).element(
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
        rn(1).element([ 6.])

        In multiple dimensions, ``axis`` can be provided for reduction over
        selected axes:

        >>> z = r23.element([[1, 2, 3],
        ...                  [4, 5, 6]])
        >>> z.__array_ufunc__(np.add, 'reduce', z, axis=1)
        rn(2).element([  6.,  15.])

        Finally, ``add.reduceat`` is a combination of ``reduce`` and
        ``at`` with rather flexible and complex semantics (see the
        `reduceat documentation`_ for details):

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'reduceat', x, [0, 1])
        rn(2).element([ 1.,  5.])

        References
        ----------
        .. _corresponding NEP:
           https://docs.scipy.org/doc/numpy/neps/ufunc-overrides.html

        .. _interface documentation:
           https://docs.scipy.org/doc/numpy/reference/arrays.classes.html\
#numpy.class.__array_ufunc__

        .. _general documentation on Numpy ufuncs:
           https://docs.scipy.org/doc/numpy/reference/ufuncs.html

        .. _reduceat documentation:
           https://docs.scipy.org/doc/numpy/reference/generated/\
numpy.ufunc.reduceat.html
        """
        # Remark: this method differs from the parent implementation only
        # in the propagation of additional space properties.

        # --- Process `out` --- #

        # Unwrap out if provided. The output parameters are all wrapped
        # in one tuple, even if there is only one.
        out_tuple = kwargs.pop('out', ())

        # Check number of `out` args, depending on `method`
        if method == '__call__' and len(out_tuple) not in (0, ufunc.nout):
            raise ValueError(
                "ufunc {}: need 0 or {} `out` arguments for "
                "`method='__call__'`, got {}"
                ''.format(ufunc.__name__, ufunc.nout, len(out_tuple)))
        elif method != '__call__' and len(out_tuple) not in (0, 1):
            raise ValueError(
                'ufunc {}: need 0 or 1 `out` arguments for `method={!r}`, '
                'got {}'.format(ufunc.__name__, method, len(out_tuple)))

        # We allow our own tensors, the data container type and
        # `numpy.ndarray` objects as `out` (see docs for reason for the
        # latter)
        valid_types = (type(self), type(self.data), np.ndarray)
        if not all(isinstance(o, valid_types) or o is None
                   for o in out_tuple):
            return NotImplemented

        # Assign to `out` or `out1` and `out2`, respectively
        out = out1 = out2 = None
        if len(out_tuple) == 1:
            out = out_tuple[0]
        elif len(out_tuple) == 2:
            out1 = out_tuple[0]
            out2 = out_tuple[1]

        # --- Get some parameters for later --- #

        # Arguments for `writable_array` and/or space constructors
        dtype_out = kwargs.get('dtype', None)
        if dtype_out is None:
            array_kwargs = {}
        else:
            array_kwargs = {'dtype': dtype_out}

        exponent = self.space.exponent
        weighting = self.space.weighting
        try:
            weighting2 = inputs[1].space.weighting
        except (IndexError, AttributeError):
            weighting2 = None

        # --- Process `inputs` --- #

        # Convert inputs that are ODL tensors to Numpy arrays so that the
        # native Numpy ufunc is called later
        inputs = tuple(
            inp.asarray() if isinstance(inp, type(self)) else inp
            for inp in inputs)

        # --- Evaluate ufunc --- #

        # Trivial context used to create a single code path for the ufunc
        # evaluation. For `None` output parameter(s), this is used instead of
        # `writable_array`.
        class CtxNone(object):
            """Trivial context manager class.

            When used as ::

                with CtxNone() as obj:
                    # do stuff with `obj`

            the returned ``obj`` is ``None``.
            """
            __enter__ = __exit__ = lambda *_: None

        if method == '__call__':
            if ufunc.nout == 1:
                # Make context for output (trivial one returns `None`)
                if out is None:
                    out_ctx = CtxNone()
                else:
                    out_ctx = writable_array(out, **array_kwargs)

                # Evaluate ufunc
                with out_ctx as out_arr:
                    kwargs['out'] = out_arr
                    res = ufunc(*inputs, **kwargs)

                # Wrap result if necessary (lazily)
                if out is None:
                    if is_floating_dtype(res.dtype):
                        # Weighting contains exponent
                        spc_kwargs = {'weighting': weighting}
                    else:
                        # No `exponent` or `weighting` applicable
                        spc_kwargs = {}
                    out_space = type(self.space)(self.shape, res.dtype,
                                                 **spc_kwargs)
                    out = out_space.element(res)

                return out

            elif ufunc.nout == 2:
                # Make contexts for outputs (trivial ones return `None`)
                if out1 is not None:
                    out1_ctx = writable_array(out1, **array_kwargs)
                else:
                    out1_ctx = CtxNone()
                if out2 is not None:
                    out2_ctx = writable_array(out2, **array_kwargs)
                else:
                    out2_ctx = CtxNone()

                # Evaluate ufunc
                with out1_ctx as out1_arr, out2_ctx as out2_arr:
                    kwargs['out'] = (out1_arr, out2_arr)
                    res1, res2 = ufunc(*inputs, **kwargs)

                # Wrap results if necessary (lazily)
                # We don't use exponents or weightings since we don't know
                # how to map them to the spaces
                if out1 is None:
                    out1_space = type(self.space)(self.shape, res1.dtype)
                    out1 = out1_space.element(res1)
                if out2 is None:
                    out2_space = type(self.space)(self.shape, res2.dtype)
                    out2 = out2_space.element(res2)

                return out1, out2

            else:
                raise NotImplementedError('nout = {} not supported'
                                          ''.format(ufunc.nout))

        else:  # method != '__call__'
            # Make context for output (trivial one returns `None`)
            if out is None:
                out_ctx = CtxNone()
            else:
                out_ctx = writable_array(out, **array_kwargs)

            # Evaluate ufunc method
            with out_ctx as out_arr:
                if method != 'at':
                    # No kwargs allowed for 'at'
                    kwargs['out'] = out_arr
                res = getattr(ufunc, method)(*inputs, **kwargs)

            # Shortcut for scalar or no return value
            if np.isscalar(res) or res is None:
                # The first occurs for `reduce` with all axes,
                # the second for in-place stuff (`at` currently)
                return res

            # Wrap result if necessary (lazily)
            if out is None:
                if is_floating_dtype(res.dtype):

                    # For weighting, we need to check which axes remain in
                    # the result and slice the weighting accordingly
                    if method == 'reduce':
                        axis = kwargs.get('axis', 0)
                        try:
                            iter(axis)
                        except TypeError:
                            axis = [axis]

                        if kwargs.get('keepdims', False):
                            reduced_axes = []
                        else:
                            reduced_axes = [i for i in range(self.ndim)
                                            if i not in axis]
                        weighting = slice_weighting_by_axis(
                            weighting, self.shape, reduced_axes)

                    elif method == 'outer':
                        # Stack the weightings as well if possible. Stacking
                        # makes sense for per-axis weighting and constant
                        # weighting for 1D inputs. An input that has no
                        # weighting results in per-axis weighting with
                        # constants 1.
                        can_stack = True
                        if isinstance(weighting,
                                      NumpyTensorSpacePerAxisWeighting):
                            factors = weighting.factors
                        elif (isinstance(weighting,
                                         NumpyTensorSpaceConstWeighting) and
                              inputs[0].ndim == 1):
                            factors = (weighting.const,)
                        else:
                            can_stack = False

                        if weighting2 is None:
                            factors = (1.0,) * (res.ndim - inputs[0].ndim)
                        elif isinstance(weighting2,
                                        NumpyTensorSpacePerAxisWeighting):
                            factors2 = weighting2.factors
                        elif (isinstance(weighting2,
                                         NumpyTensorSpaceConstWeighting) and
                              inputs[1].ndim == 1):
                            factors2 = (weighting2.const,)
                        elif (isinstance(weighting2,
                                         NumpyTensorSpaceConstWeighting) and
                              weighting2.const == 1):
                            factors2 = (1.0,) * inputs[1].ndim
                        else:
                            can_stack = False

                        if can_stack:
                            weighting = NumpyTensorSpacePerAxisWeighting(
                                factors + factors2,
                                exponent=weighting.exponent)
                        else:
                            weighting = NumpyTensorSpaceConstWeighting(
                                1.0, exponent)

                    elif (res.shape != self.shape and
                          not isinstance(weighting,
                                         NumpyTensorSpaceConstWeighting)):
                        # For other cases, preserve constant weighting, and
                        # preserve other weightings if the shape is unchanged
                        weighting = NumpyTensorSpaceConstWeighting(
                            1.0, exponent)

                    spc_kwargs = {'weighting': weighting}

                else:  # not is_floating_dtype(res.dtype)
                    spc_kwargs = {}

                out_space = type(self.space)(res.shape, res.dtype,
                                             **spc_kwargs)
                out = out_space.element(res)

            return out


# --- Implementations of low-level functions --- #


def _blas_is_applicable(*args):
    """Whether BLAS routines can be applied or not.

    BLAS routines are available for single and double precision
    float or complex data only. If the arrays are non-contiguous,
    BLAS methods are usually slower, and array-writing routines do
    not work at all. Hence, only contiguous arrays are allowed.

    Parameters
    ----------
    x1,...,xN : `numpy.ndarray`
        The arrays to be tested for BLAS conformity.

    Returns
    -------
    blas_is_applicable : bool
        ``True`` if all mentioned requirements are met, ``False`` otherwise.
    """
    if any(x.dtype != args[0].dtype for x in args[1:]):
        return False
    elif any(x.dtype not in _BLAS_DTYPES for x in args):
        return False
    elif not (all(x.flags.f_contiguous for x in args) or
              all(x.flags.c_contiguous for x in args)):
        return False
    elif any(x.size > np.iinfo('int32').max for x in args):
        # Temporary fix for 32 bit int overflow in BLAS
        # TODO: use chunking instead
        return False
    else:
        return True


def _lincomb_impl(a, x1, b, x2, out):
    """Optimized implementation of ``out[:] = a * x1 + b * x2``."""
    # Lazy import to improve `import odl` time
    import scipy.linalg

    size = native(x1.size)
    if size < THRESHOLD_SMALL:
        # Faster for small arrays
        out[:] = a * x1 + b * x2
        return

    elif (size < THRESHOLD_MEDIUM or
          not _blas_is_applicable(x1, x2, out)):

        def fallback_axpy(x1, x2, n, a):
            """Fallback axpy implementation avoiding copy."""
            if a != 0:
                x2 /= a
                x2 += x1
                x2 *= a
            return x2

        def fallback_scal(a, x, n):
            """Fallback scal implementation."""
            x *= a
            return x

        def fallback_copy(x1, x2, n):
            """Fallback copy implementation."""
            x2[...] = x1[...]
            return x2

        axpy, scal, copy = (fallback_axpy, fallback_scal, fallback_copy)

    else:
        # Need flat data for BLAS, otherwise in-place does not work.
        # Raveling must happen in fixed order for non-contiguous out,
        # otherwise 'A' is applied to arrays, which makes the outcome
        # dependent on their respective contiguousness.
        if out.flags.f_contiguous:
            ravel_order = 'F'
        else:
            ravel_order = 'C'

        x1 = x1.ravel(order=ravel_order)
        x2 = x2.ravel(order=ravel_order)
        out = out.ravel(order=ravel_order)
        axpy, scal, copy = scipy.linalg.blas.get_blas_funcs(
            ['axpy', 'scal', 'copy'], arrays=(x1, x2, out))

    if x1 is x2 and b != 0:
        # x1 is aligned with x2 -> out = (a+b)*x1
        _lincomb_impl(a + b, x1, 0, x1, out)
    elif out is x1 and out is x2:
        # All the vectors are aligned -> out = (a+b)*out
        if (a + b) != 0:
            scal(a + b, out, size)
        else:
            out[:] = 0
    elif out is x1:
        # out is aligned with x1 -> out = a*out + b*x2
        if a != 1:
            scal(a, out, size)
        if b != 0:
            axpy(x2, out, size, b)
    elif out is x2:
        # out is aligned with x2 -> out = a*x1 + b*out
        if b != 1:
            scal(b, out, size)
        if a != 0:
            axpy(x1, out, size, a)
    else:
        # We have exhausted all alignment options, so x1 is not x2 is not out
        # We now optimize for various values of a and b
        if b == 0:
            if a == 0:  # Zero assignment -> out = 0
                out[:] = 0
            else:  # Scaled copy -> out = a*x1
                copy(x1, out, size)
                if a != 1:
                    scal(a, out, size)

        else:  # b != 0
            if a == 0:  # Scaled copy -> out = b*x2
                copy(x2, out, size)
                if b != 1:
                    scal(b, out, size)

            elif a == 1:  # No scaling in x1 -> out = x1 + b*x2
                copy(x1, out, size)
                axpy(x2, out, size, b)
            else:  # Generic case -> out = a*x1 + b*x2
                copy(x2, out, size)
                if b != 1:
                    scal(b, out, size)
                axpy(x1, out, size, a)


def _norm_impl(x):
    """Default Euclidean norm implementation."""
    # Lazy import to improve `import odl` time
    import scipy.linalg

    if _blas_is_applicable(x):
        nrm2 = scipy.linalg.blas.get_blas_funcs('nrm2', dtype=x.dtype)
        norm = partial(nrm2, n=native(x.size))
    else:
        norm = np.linalg.norm
    return norm(x.ravel())


def _pnorm_impl(x, p):
    """Default p-norm implementation."""
    return np.linalg.norm(x.ravel(), ord=p)


def _pnorm_diagweight_impl(x, p, w):
    """Diagonally weighted p-norm implementation."""
    # Ravel both in the same order (w is a numpy array)
    order = 'F' if all(a.flags.f_contiguous for a in (x, w)) else 'C'

    # This is faster than first applying the weights and then summing with
    # BLAS dot or nrm2
    xp = np.abs(x.ravel(order))
    if p == float('inf'):
        return np.max(xp)
    elif p == -float('inf'):
        return np.min(xp)
    else:
        xp = np.power(xp, p, out=xp)
        xp *= w.ravel(order)
        return np.sum(xp) ** (1 / p)


def _inner_impl(x1, x2):
    """Default Euclidean inner product implementation."""
    # Ravel both in the same order
    order = 'F' if all(a.flags.f_contiguous for a in (x1, x2)) else 'C'

    if is_real_dtype(x1.dtype):
        if x1.size > THRESHOLD_MEDIUM:
            # This is as fast as BLAS dotc
            return np.tensordot(x1, x2, [range(x1.ndim)] * 2)
        else:
            # Several times faster for small arrays
            return np.dot(x1.ravel(order), x2.ravel(order))
    else:
        # x2 as first argument because we want linearity in x1
        return np.vdot(x2.ravel(order), x1.ravel(order))


# --- Weightings --- #


def slice_weighting(weighting, space_shape, indices):
    """Return a weighting for a space after indexing.

    The different types of weightings behave as follows:

    - ``ConstWeighting``: preserved
    - ``PerAxisWeighting``: preserved, where factors are discarded for
      removed axes and sliced for other axes in which an array is used
    - ``ArrayWeighting``: preserved, using the sliced array for the
      new weighting
    - Other: not preserved, mapped to ``None``.
    """
    indices = normalized_index_expression(indices, space_shape)
    new_shape, removed_axes, new_axes, _ = simulate_slicing(
        space_shape, indices)

    if isinstance(weighting, NumpyTensorSpaceConstWeighting):
        new_weighting = weighting

    elif isinstance(weighting, NumpyTensorSpacePerAxisWeighting):
        # Determine factors without `None` components
        factors = []
        indices_no_none = [i for i in indices if i is not None]
        for i, (fac, idx) in enumerate(zip_longest(weighting.factors,
                                                   indices_no_none,
                                                   fillvalue=slice(None))):
            if i in removed_axes:
                continue

            if fac.ndim == 0:
                factors.append(fac)
            else:
                factors.append(fac[idx])

        # Add 1.0 for new axes
        for newax in new_axes:
            factors.insert(newax, 1.0)

        new_weighting = NumpyTensorSpacePerAxisWeighting(
            factors, weighting.exponent)

    elif isinstance(weighting, NumpyTensorSpaceArrayWeighting):
        array = weighting.array[indices]
        new_weighting = NumpyTensorSpaceArrayWeighting(
            array, weighting.exponent)
    else:
        new_weighting = None

    return new_weighting


def slice_weighting_by_axis(weighting, space_shape, indices):
    """Return a weighting for a space after indexing by axis.

    The different types of weightings behave as follows:

    - ``ConstWeighting``: preserved
    - ``PerAxisWeighting``: preserved, where factors are discarded for
      removed axes, repeated for repeated axes, and set to 1 for new
      axes
    - ``ArrayWeighting``: not preserved, no meaningful way to slice by axis
    - Other: not preserved, mapped to ``None``.
    """
    try:
        iter(indices)
    except TypeError:
        # Integer, slice or Ellipsis
        indices = list(range(len(space_shape)))[indices]
        if not isinstance(indices, list):
            indices = [indices]
    else:
        indices = [int(i) for i in indices]

    if isinstance(weighting, NumpyTensorSpaceConstWeighting):
        new_weighting = weighting

    elif isinstance(weighting, NumpyTensorSpacePerAxisWeighting):
        factors = [weighting.factors[i] for i in indices]
        new_weighting = NumpyTensorSpacePerAxisWeighting(
            factors, weighting.exponent)

    else:
        new_weighting = None

    return new_weighting


class NumpyTensorSpaceArrayWeighting(ArrayWeighting):

    """Weighting of a `NumpyTensorSpace` by an array.

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

        - For other exponents, only norm and dist are defined. In the case
          of finite exponent, it is (using point-wise exponentiation)

          .. math::
              \| A\|_{W, p} :=
              \| W^{1/p} \odot A\|_p =
              \| w^{1/p} \odot a\|_p,

          and for :math:`\pm \infty` we have

          .. math::
              \| A\|_{W, \pm \infty} :=
              \| W \odot A\|_{\pm \infty} =
              \| w \odot a\|_{\pm \infty}.

          Note that this definition is chosen such that the limit
          property in :math:`p` holds, i.e.

          .. math::
              \| A\|_{W, p} \to
              \| A\|_{W, \infty} \quad (p \to \infty).

        - The array :math:`W` may only have positive entries, otherwise
          it does not define an inner product or norm, respectively. This
          is not checked during initialization.
        """
        if isinstance(array, NumpyTensor):
            array = array

        array = np.asarray(array)
        if array.dtype == object:
            raise ValueError('got invalid `array` as input')

        super(NumpyTensorSpaceArrayWeighting, self).__init__(
            array, impl='numpy', exponent=exponent)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash(
            (type(self), array_hash(self.array), self.exponent)
        )

    def inner(self, x1, x2):
        """Return the weighted inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
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
            inner = _inner_impl(x1.data * self.array, x2.data)
            if is_real_dtype(x1.dtype):
                return float(inner)
            else:
                return complex(inner)

    def norm(self, x):
        """Return the weighted norm of ``x``.

        Parameters
        ----------
        x : `NumpyTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the provided tensor.
        """
        if self.exponent == 2.0:
            norm_squared = self.inner(x, x).real  # TODO: optimize?
            if norm_squared < 0:
                norm_squared = 0.0  # Compensate for numerical error
            return float(np.sqrt(norm_squared))
        else:
            return float(_pnorm_diagweight_impl(x.data, self.exponent,
                                                self.array))


class NumpyTensorSpacePerAxisWeighting(PerAxisWeighting):

    """Weighting of a space with one weight per axis.

    See Notes for mathematical details.
    """

    def __init__(self, factors, exponent=2.0):
        r"""Initialize a new instance.

        Parameters
        ----------
        factors : sequence of `array-like`
            Weighting factors, one per axis. The factors can be constants
            or one-dimensional array-like objects.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.

        Notes
        -----
        We consider a tensor space :math:`\mathbb{F}^n` of shape
        :math:`n \in \mathbb{N}^d` over a field :math:`\mathbb{F}`.
        If :math:`0 < v^{(i)} \in \mathbb{R}^{n_i}` are vectors of length
        equal to the corresponding axis, their outer product

        .. math::
            v_{k} = \prod_{i=0}^d v^{(i)}_{k_i}

        defines a tensor of shape :math:`n`. With this tensor
        we can define a weighted space as follows.

        - For exponent 2.0, a new weighted inner product is given as

          .. math::
              \langle a, b\rangle_v :=
              \langle v \odot a, b\rangle =
              b^{\mathrm{H}} (v \odot a),

          where :math:`b^{\mathrm{H}}` stands for transposed complex
          conjugate and ":math:`\odot`" for pointwise product.

        - For other exponents, only norm and dist are defined. In the case
          of finite exponent, it is (using point-wise exponentiation)

          .. math::
              \| a \|_{v, p} := \| v^{1/p} \odot a \|_{p},

          and for :math:`\pm \infty` we have

          .. math::
              \| a \|_{v, \pm \infty} := \| a \|_{\pm \infty},

          Note that this definition is chosen such that the limit
          property in :math:`p` holds, i.e.

          .. math::
              \| a\|_{v, p} \to
              \| a \|_{v, \infty} \quad (p \to \infty).
        """
        # TODO: allow 3-tuples for `bdry, inner, bdry` type factors
        conv_factors = []
        for factor in factors:
            factor = np.asarray(factor)
            if factor.ndim not in (0, 1):
                raise ValueError(
                    '`factors` must all be scalar or 1-dim. vectors, got '
                    '{}-dim. entries'.format(factor.ndim))

            conv_factors.append(factor)

        super(NumpyTensorSpacePerAxisWeighting, self).__init__(
            conv_factors, impl='numpy', exponent=exponent)

    @property
    def const_axes(self):
        """Tuple of indices in which the factors are constants."""
        return tuple(i for i, fac in enumerate(self.factors) if fac.ndim == 0)

    @property
    def consts(self):
        """Tuple containing those factors that are constants."""
        return tuple(fac for fac in self.factors if fac.ndim == 0)

    @property
    def array_axes(self):
        """Tuple of indices in which the factors are arrays."""
        return tuple(i for i, fac in enumerate(self.factors) if fac.ndim == 1)

    @property
    def arrays(self):
        """Tuple containing those factors that are arrays."""
        return tuple(fac for fac in self.factors if fac.ndim == 1)

    def inner(self, x1, x2):
        """Return the weighted inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
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

        const = np.prod(self.consts)
        x1_w = fast_1d_tensor_mult(x1.data, self.arrays, axes=self.array_axes)
        inner = const * _inner_impl(x1_w, x2.data)
        if x1.space.field is None:
            return inner
        else:
            return x1.space.field.element(inner)

    def norm(self, x):
        """Return the weighted norm of ``x``.

        Parameters
        ----------
        x1 : `NumpyTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the tensor.
        """
        const = np.prod(self.consts)

        if self.exponent == 2.0:
            arrays = [np.sqrt(arr) for arr in self.arrays]
            x_w = fast_1d_tensor_mult(x.data, arrays, axes=self.array_axes)
            return float(np.sqrt(const) * _norm_impl(x_w))
        elif self.exponent == float('inf'):
            return float(_pnorm_impl(x.data, float('inf')))
        else:
            arrays = [np.power(arr, 1 / self.exponent) for arr in self.arrays]
            x_w = fast_1d_tensor_mult(x.data, arrays, axes=self.array_axes)
            return float(const ** (1 / self.exponent) *
                         _pnorm_impl(x_w, self.exponent))

    def dist(self, x1, x2):
        """Return the weighted distance between ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the tensors.
        """
        const = np.prod(self.consts)
        diff = (x1 - x2).data

        if self.exponent == 2.0:
            arrays = [np.sqrt(arr) for arr in self.arrays]
            fast_1d_tensor_mult(diff, arrays, axes=self.array_axes, out=diff)
            return float(np.sqrt(const) * _norm_impl(diff))
        elif self.exponent == float('inf'):
            return float(_pnorm_impl(diff, float('inf')))
        else:
            arrays = [np.power(arr, 1 / self.exponent) for arr in self.arrays]
            fast_1d_tensor_mult(diff, arrays, axes=self.array_axes, out=diff)
            return float(const ** (1 / self.exponent) *
                         _pnorm_impl(diff, self.exponent))

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if ``other`` is a `NumpyTensorSpacePerAxisWeighting`
            instance with the same factors (equal for constants,
            *identical* for arrays), ``False`` otherwise.
        """
        if other is self:
            return True

        if isinstance(other, ConstWeighting):
            # Consider per-axis weighting in 1 axis with a constant to
            # be equal to constant weighting
            return (len(self.factors) == 1 and
                    self.factors[0].ndim == 0 and
                    self.factors[0] == other.const)
        elif not isinstance(other, NumpyTensorSpacePerAxisWeighting):
            return False

        same_const_idcs = (other.const_axes == self.const_axes)
        consts_equal = (self.consts == other.consts)
        arrs_ident = (a is b for a, b in zip(self.arrays, other.arrays))

        return (super(NumpyTensorSpacePerAxisWeighting, self).__eq__(other) and
                same_const_idcs and
                consts_equal and
                arrs_ident)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash(
            (super(NumpyTensorSpacePerAxisWeighting, self).__hash__(),) +
            tuple(array_hash(fac) for fac in self.factors)
        )

    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        max_elems = 2 * np.get_printoptions()['edgeitems']
        precision = np.get_printoptions()['precision']

        def factors_repr(factors):
            """Return repr string for the weighting factors part."""
            factor_strs = []
            for fac in factors:
                if fac.ndim == 0:
                    fmt = '{{:.{}}}'.format(precision)
                    factor_strs.append(fmt.format(float(fac)))
                else:
                    factor_strs.append(array_str(fac, nprint=max_elems))
            if len(factor_strs) == 1:
                return factor_strs[0]
            else:
                return '({})'.format(', '.join(factor_strs))

        optargs = []
        optmod = []
        if not all(fac.ndim == 0 and fac == 1.0 for fac in self.factors):
            optargs.append(('weighting', factors_repr(self.factors), ''))
            optmod.append('!s')

        optargs.append(('exponent', self.exponent, 2.0))
        optmod.append('')

        return signature_string([], optargs, mod=[[], optmod])

    def __repr__(self):
        """Return ``repr(self)``."""
        max_elems = 2 * np.get_printoptions()['edgeitems']
        precision = np.get_printoptions()['precision']

        def factors_repr(factors):
            """Return repr string for the weighting factors part."""
            factor_strs = []
            for fac in factors:
                if fac.ndim == 0:
                    fmt = '{{:.{}}}'.format(precision)
                    factor_strs.append(fmt.format(float(fac)))
                else:
                    factor_strs.append(array_str(fac, nprint=max_elems))
            return '({})'.format(', '.join(factor_strs))

        posargs = [factors_repr(self.factors)]
        optargs = [('exponent', self.exponent, 2.0)]

        inner_str = signature_string(posargs, optargs, mod=['!s', ''])
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class NumpyTensorSpaceConstWeighting(ConstWeighting):

    """Weighting of a `NumpyTensorSpace` by a constant.

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

        - For other exponents, only norm and dist are defined. In the case
          of finite exponent, it is

          .. math::
              \| a \|_{c, p} :=
              c^{1/p}\, \| a \|_{p},

          and for :math:`\pm \infty` we have

          .. math::
              \| a \|_{c, \pm \infty} :=
              \| a \|_{\pm \infty}.

          Note that this definition is chosen such that the limit
          property in :math:`p` holds, i.e.

          .. math::
              \| a\|_{c, p} \to
              \| a \|_{c, \infty} \quad (p \to \infty)

        - The constant must be positive, otherwise it does not define an
          inner product or norm, respectively.
        """
        super(NumpyTensorSpaceConstWeighting, self).__init__(
            const, impl='numpy', exponent=exponent)

    def inner(self, x1, x2):
        """Return the weighted inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
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

        inner = self.const * _inner_impl(x1.data, x2.data)
        if x1.space.field is None:
            return inner
        else:
            return x1.space.field.element(inner)

    def norm(self, x):
        """Return the weighted norm of ``x``.

        Parameters
        ----------
        x1 : `NumpyTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the tensor.
        """
        if self.exponent == 2.0:
            return float(np.sqrt(self.const) * _norm_impl(x.data))
        elif self.exponent == float('inf'):
            return float(_pnorm_impl(x.data, self.exponent))
        else:
            return float(self.const ** (1 / self.exponent) *
                         _pnorm_impl(x.data, self.exponent))

    def dist(self, x1, x2):
        """Return the weighted distance between ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the tensors.
        """
        if self.exponent == 2.0:
            return float(np.sqrt(self.const) * _norm_impl((x1 - x2).data))
        elif self.exponent == float('inf'):
            return float(_pnorm_impl((x1 - x2).data, self.exponent))
        else:
            return float((self.const ** (1 / self.exponent) *
                          _pnorm_impl((x1 - x2).data, self.exponent)))


class NumpyTensorSpaceCustomInner(CustomInner):

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
        super(NumpyTensorSpaceCustomInner, self).__init__(inner, impl='numpy')


class NumpyTensorSpaceCustomNorm(CustomNorm):

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
        super(NumpyTensorSpaceCustomNorm, self).__init__(norm, impl='numpy')


class NumpyTensorSpaceCustomDist(CustomDist):

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
        super(NumpyTensorSpaceCustomDist, self).__init__(dist, impl='numpy')


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
