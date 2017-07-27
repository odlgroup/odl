# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""NumPy implementation of tensor spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
from future.utils import native
standard_library.install_aliases()

import ctypes
from functools import partial
from numbers import Integral
import numpy as np
import scipy.linalg as linalg

from odl.set.sets import RealNumbers, ComplexNumbers
from odl.set.space import LinearSpaceTypeError
from odl.space.base_tensors import TensorSpace, Tensor
from odl.space.weighting import (
    Weighting, ArrayWeighting, ConstWeighting, NoWeighting,
    CustomInner, CustomNorm, CustomDist)
from odl.util import (
    dtype_str, signature_string, is_real_dtype, is_numeric_dtype,
    writable_array, is_floating_dtype)


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
    See also [Hac2012]_ "Part I Algebraic Tensors" for a rigorous
    treatment of tensors with a definition close to this one.

    References
    ----------
    [Hac2012] Hackbusch, W. *Tensor Spaces and Numerical Tensor Calculus*.
    Springer, 2012.

    .. _Wikipedia article on tensors: https://en.wikipedia.org/wiki/Tensor
    """

    def __init__(self, shape, dtype=None, order='A', **kwargs):
        """Initialize a new instance.

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
        order : {'A', 'C', 'F'}, optional
            Axis ordering of the data storage. Only relevant for more
            than 1 axis.
            For ``'C'`` and ``'F'``, elements are forced to use
            contiguous memory in the respective ordering.
            For ``'A'`` ("any") no contiguousness is enforced.
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

            ``None``: no weighting, i.e. weighting with ``1.0`` (default).

            `Weighting`: Use this weighting as-is. Compatibility
            with this space's elements is not checked during init.

            ``float``: Weighting by a constant

            array-like: Pointwise weighting by an array

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
        rn : constructor for real tensor spaces
        cn : constructor for complex tensor spaces
        tensor_space :
            constructor for tensor spaces of arbitrary scalar
            data type

        Notes
        -----
        - A distance function or metric on a space :math:`\mathcal{X}`
          is a mapping
          :math:`d:\mathcal{X} \\times \mathcal{X} \\to \mathbb{R}`
          satisfying the following conditions for all space elements
          :math:`x, y, z`:

          * :math:`d(x, y) \geq 0`,
          * :math:`d(x, y) = 0 \Leftrightarrow x = y`,
          * :math:`d(x, y) = d(y, x)`,
          * :math:`d(x, y) \\leq d(x, z) + d(z, y)`.

        - A norm on a space :math:`\mathcal{X}` is a mapping
          :math:`\| \cdot \|:\mathcal{X} \\to \mathbb{R}`
          satisfying the following conditions for all
          space elements :math:`x, y`: and scalars :math:`s`:

          * :math:`\| x\| \geq 0`,
          * :math:`\| x\| = 0 \Leftrightarrow x = 0`,
          * :math:`\| sx\| = |s| \cdot \| x \|`,
          * :math:`\| x+y\| \\leq \| x\| +
            \| y\|`.

        - An inner product on a space :math:`\mathcal{X}` over a field
          :math:`\mathbb{F} = \mathbb{R}` or :math:`\mathbb{C}` is a
          mapping
          :math:`\\langle\cdot, \cdot\\rangle: \mathcal{X} \\times
          \mathcal{X} \\to \mathbb{F}`
          satisfying the following conditions for all
          space elements :math:`x, y, z`: and scalars :math:`s`:

          * :math:`\\langle x, y\\rangle =
            \overline{\\langle y, x\\rangle}`,
          * :math:`\\langle sx + y, z\\rangle = s \\langle x, z\\rangle +
            \\langle y, z\\rangle`,
          * :math:`\\langle x, x\\rangle = 0 \Leftrightarrow x = 0`.

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
        tensor_space((2, 3), 'int')
        """
        TensorSpace.__init__(self, shape, dtype, order)

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
        num_extra_args = sum(1 if a is not None else 0
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
            else:
                self.__weighting = _weighting(weighting, exponent)

            # Check (afterwards) that the weighting input was sane
            if isinstance(self.weighting, NumpyTensorSpaceArrayWeighting):
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
            self.__weighting = NumpyTensorSpaceCustomDist(dist)
        elif norm is not None:
            self.__weighting = NumpyTensorSpaceCustomNorm(norm)
        elif inner is not None:
            self.__weighting = NumpyTensorSpaceCustomInner(inner)
        else:
            self.__weighting = NumpyTensorSpaceNoWeighting(exponent)

    @property
    def impl(self):
        """Name of the implementation back-end: ``'numpy'``."""
        return 'numpy'

    @property
    def exponent(self):
        """Exponent of the norm and the distance."""
        return self.weighting.exponent

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.__weighting

    @property
    def is_weighted(self):
        """Return ``True`` if the space has a non-trivial weighting."""
        return not isinstance(self.weighting, NumpyTensorSpaceNoWeighting)

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : `array-like`, optional
            Input used to initialize the new element.

            If ``inp`` is `None`, an empty element is created with no
            guarantee of its state (memory allocation only).

            If ``inp`` is a `numpy.ndarray` of the same `shape` and
            `dtype` as this space, or if ``inp`` already lies in this
            space, it is wrapped, not copied.
            Other array-like objects are always copied.

        data_ptr : int, optional
            Pointer to the start memory address of a contiguous Numpy array
            or an equivalent raw container with the same total number of
            bytes. This option can only be used if `order` is ``'C'`` or
            ``'F'``, otherwise contiguousness cannot be guaranteed.
            The option is also mutually exclusive with ``inp``.

        Returns
        -------
        element : `NumpyTensor`
            The new element created (from ``inp``).

        Notes
        -----
        This method preserves "array views" of correct size and type,
        see the examples below.

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

        >>> int_space = odl.tensor_space((2, 3), dtype=int, order='F')
        >>> arr = np.array([[1, 2, 3],
        ...                 [4, 5, 6]], dtype=int, order='F')
        >>> ptr = arr.ctypes.data
        >>> y = int_space.element(data_ptr=ptr)
        >>> y
        tensor_space((2, 3), 'int', order='F').element(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        >>> y[0, 1] = -1
        >>> arr
        array([[ 1, -1,  3],
               [ 4,  5,  6]])
        """
        if inp is None and data_ptr is None:
            arr = np.empty(self.shape, dtype=self.dtype,
                           order=self.default_order)
            return self.element_type(self, arr)

        elif inp is None and data_ptr is not None:
            if self.order == 'A':
                raise ValueError("`data_ptr` cannot be used with 'A' "
                                 "ordering")
            ctype_array_def = ctypes.c_byte * self.nbytes
            as_ctype_array = ctype_array_def.from_address(data_ptr)
            as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
            arr = as_numpy_array.view(dtype=self.dtype)
            arr = arr.reshape(self.shape, order=self.order)
            return self.element_type(self, arr)

        elif inp is not None and data_ptr is None:
            if inp in self:
                # Short-circuit for space elements
                return inp

            # Use `order` to preserve views if possible
            arr = np.array(inp, copy=False, dtype=self.dtype, ndmin=self.ndim,
                           order=self.order)
            if arr.shape != self.shape:
                raise ValueError('shape of `inp` not equal to space `shape`: '
                                 '{} != {}'.format(arr.shape, self.shape))
            return self.element_type(self, arr)

        else:
            raise ValueError('cannot provide both `inp` and `data_ptr`')

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

        Notes
        -----
        This is all dtypes available in Numpy. See `numpy.sctypes`
        for more information.

        The available dtypes may depend on the specific system used.
        """
        all_dtypes = []
        for lst in np.sctypes.values():
            for dtype in lst:
                if dtype not in (np.object, np.void):
                    all_dtypes.append(np.dtype(dtype))
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
        _lincomb_impl(a, x1, b, x2, out)

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
            with the same `NumpyTensorSpace.shape`,
            `NumpyTensorSpace.dtype`, `NumpyTensorSpace.order`
            and `NumpyTensorSpace.weighting`, otherwise False.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> same_space = odl.rn(3, exponent=2)
        >>> same_space == space
        True

        Different `shape`, `exponent`, `dtype` or `order` all result in
        different spaces:

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
        >>> space_a = odl.rn((2, 3), order='A')  # default
        >>> space_c = odl.rn((2, 3), order='C')  # default
        >>> space_a == space_c
        False

        A `NumpyTensorSpace` with the same properties is considered
        equal:

        >>> same_space = odl.NumpyTensorSpace((2, 3), dtype='float64')
        >>> same_space == space_a
        True
        """
        if other is self:
            return True

        return (TensorSpace.__eq__(self, other) and
                self.weighting == other.weighting)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.shape, self.dtype, self.order,
                     self.weighting))

    @property
    def byaxis(self):
        """Return the subspace defined along one or several dimensions.

        Examples
        --------
        Indexing can be done with any index expression that is
        supported for indexing of ``numpy.arange(ndim)``:

        >>> space = odl.rn((2, 3, 4))
        >>> space.byaxis[0]
        rn(2)
        >>> space.byaxis[[2, 0, 1]]
        rn((4, 2, 3))
        >>> space.byaxis[1:]
        rn((3, 4))
        """
        space = self

        class byaxis(object):
            """Helper class for indexing by axis."""

            def __getitem__(self, indices):
                """Return space along axes in ``indices``."""
                if isinstance(indices, np.ndarray):
                    raise TypeError('indexing with numpy.ndarray not '
                                    'supported')

                indices = np.arange(space.ndim)[indices]
                if indices.ndim > 1:
                    indices = indices.squeeze()

                if indices.ndim > 1:
                    raise ValueError('index list must be one-dimensional')

                if np.isscalar(indices):
                    indices = [indices]

                newshape = tuple(space.shape[i] for i in indices)

                if isinstance(space.weighting, ArrayWeighting):
                    new_array = np.asarray(self.weighting.array[indices])
                    weighting = NumpyTensorSpaceArrayWeighting(
                        new_array, space.weighting.exponent)
                else:
                    weighting = space.weighting

                return type(space)(newshape, space.dtype, space.order,
                                   weighting=weighting)

            def __repr__(self):
                """Return ``repr(self)``."""
                return repr(space) + '.byaxis'

        return byaxis()

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Examples
        --------
        Integers and slices index the first axis:

        >>> space = odl.rn((2, 3, 4))
        >>> space[1]
        rn((3, 4))
        >>> space[1:]
        rn((1, 3, 4))

        Tuples, i.e. multi-indices, index per axis:

        >>> space[0, 1]
        rn(4)
        >>> space[0, 1:]
        rn((2, 4))
        """
        if isinstance(indices, Integral):
            newshape = self.shape[1:]

        elif isinstance(indices, slice):
            # Take slice along axis 0
            start, stop, step = indices.indices(self.shape[0])
            newshape = (int(np.ceil((stop - start) / step)),) + self.shape[1:]

        elif isinstance(indices, list):
            # Fancy indexing, only for compatibility since nothing really
            # interesting happens here
            idx_arr = np.array(indices, dtype=int, ndmin=2)
            for i, (row, length) in enumerate(zip(idx_arr, self.shape)):
                row[row < 0] += length
                if np.any((row < 0) | (row >= length)):
                    raise IndexError('list contains invalid indices in row {}'
                                     ''.format(i))
            newshape = (idx_arr.shape[1],) + self.shape[idx_arr.shape[0]:]

        elif isinstance(indices, tuple):
            newshape = []
            for ax, (idx, length) in enumerate(zip(indices, self.shape)):
                if isinstance(idx, Integral):
                    norm_idx = length - idx if idx < 0 else idx
                    if not 0 <= norm_idx < length:
                        raise IndexError(
                            'index {} out of range in axis {} with length {}'
                            ''.format(idx, ax, length))
                elif isinstance(idx, slice):
                    start, stop, step = idx.indices(length)
                    if start >= length:
                        newshape.append(0)
                    else:
                        stop = min(stop, length)
                        newshape.append(int(np.ceil((stop - start) / step)))
                else:
                    raise TypeError('index tuple may only contain'
                                    'integers or slices')
            newshape.extend(self.shape[len(indices):])
            newshape = tuple(newshape)

        else:
            raise TypeError('`indices` must be integer, slice, tuple or '
                            'or list, got {!r}'.format(indices))

        if isinstance(self.weighting, ArrayWeighting):
            new_array = np.asarray(self.weighting.array[indices])
            weighting = NumpyTensorSpaceArrayWeighting(
                new_array, self.weighting.exponent)
        else:
            weighting = self.weighting

        return type(self)(newshape, self.dtype, self.order,
                          weighting=weighting)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim <= 1:
            posargs = [self.size]
        else:
            posargs = [self.shape]

        if self.is_real_space:
            constructor_name = 'rn'
        elif self.is_complex_space:
            constructor_name = 'cn'
        else:
            constructor_name = 'tensor_space'

        if (constructor_name == 'tensor_space' or
                not is_numeric_dtype(self.dtype) or
                self.dtype != self.default_dtype(self.field)):
            posargs.append(dtype_str(self.dtype))

        optargs = [('order', self.order, 'A')]
        inner_str = signature_string(posargs, optargs)
        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str

        return '{}({})'.format(constructor_name, inner_str)

    @property
    def element_type(self):
        """Type of elements in this space: `NumpyTensor`."""
        return NumpyTensor


class NumpyTensor(Tensor):

    """Representation of a `NumpyTensorSpace` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        Tensor.__init__(self, space)
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

    @property
    def data_ptr(self):
        """A raw pointer to the data container of ``self``.

        Examples
        --------
        >>> import ctypes
        >>> space = odl.tensor_space(3, dtype='int32')
        >>> x = space.element([1, 2, 3])
        >>> arr_type = ctypes.c_int32 * 3  # C type "array of 6 int32"
        >>> buffer = arr_type.from_address(x.data_ptr)
        >>> arr = np.frombuffer(buffer, dtype='int32')
        >>> arr
        array([1, 2, 3], dtype=int32)

        In-place modification via pointer:

        >>> arr[0] = -1
        >>> x
        tensor_space(3, 'int32').element([-1, 2, 3])
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
        """Create an identical (deep) copy of this vector.

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

    __copy__ = copy

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

        Slices can be assigned to, except if lists are used for
        indexing:

        >>> y = x[:, ::2]  # view into x
        >>> y[:] = -9
        >>> x
        rn((2, 3)).element(
            [[-9.,  2., -9.],
             [-9.,  5., -9.]]
        )
        >>> y = x[[[0, 1], [1, 2]]]  # not a view, won't modify x
        >>> y
        rn(2).element([ 2., -9.])
        >>> y[:] = 0
        >>> x
        rn((2, 3)).element(
            [[-9.,  2., -9.],
             [-9.,  5., -9.]]
        )
        """
        arr = self.data[indices]
        if np.isscalar(arr):
            if self.space.field is not None:
                return self.space.field.element(arr)
            else:
                return arr
        else:
            space_constructor = type(self.space)
            if (self.order in ('C', 'F') and
                    arr.flags[self.order + '_CONTIGUOUS']):
                out_spc_order = self.order
            else:
                # To preserve the array view for non-contiguous slices,
                # we need to use 'A' for the space in that case.
                out_spc_order = 'A'
            space = space_constructor(
                arr.shape, dtype=self.dtype, order=out_spc_order,
                exponent=self.space.exponent, weighting=self.space.weighting)
            return space.element(arr)

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
        if isinstance(values, NumpyTensor):
            self.data[indices] = values.data
        else:
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
        >>> space = odl.cn(3)
        >>> x = space.element([1 + 1j, 2, 3 - 3j])
        >>> x.real
        rn(3).element([ 1.,  2.,  3.])
        """
        if self.space.is_real_space:
            return self
        elif self.space.is_complex_space:
            # Definitely non-contiguous
            real_space = self.space.astype(self.space.real_dtype, order='A')
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

        Examples
        --------
        >>> space = odl.cn(3)
        >>> x = space.element([1 + 1j, 2, 3 - 3j])
        >>> zero = odl.rn(3).zero()
        >>> x.real = zero
        >>> x
        cn(3).element([1j, 0j, -3j])

        Other array-like types and broadcasting:

        >>> x.real = 1.0
        >>> x
        cn(3).element([(1+1j), (1+0j), (1-3j)])
        >>> x.real = [2, 3, 4]
        >>> x
        cn(3).element([(2+1j), (3+0j), (4-3j)])
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
        >>> space = odl.cn(3)
        >>> x = space.element([1 + 1j, 2, 3 - 3j])
        >>> x.imag
        rn(3).element([ 1.,  0., -3.])
        """
        if self.space.is_real_space:
            return self.space.zero()
        elif self.space.is_complex_space:
            # Definitely non-contiguous
            real_space = self.space.astype(self.space.real_dtype, order='A')
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

        Examples
        --------
        >>> space = odl.cn(3)
        >>> x = space.element([1 + 1j, 2, 3 - 3j])
        >>> zero = odl.rn(3).zero()
        >>> x.imag = zero
        >>> x
        cn(3).element([(1+0j), (2+0j), (3+0j)])

        Other array-like types and broadcasting:

        >>> x.imag = 1.0
        >>> x
        cn(3).element([(1+1j), (2+1j), (3+1j)])
        >>> x.imag = [2, 3, 4]
        >>> x
        cn(3).element([(1+2j), (2+3j), (3+4j)])
        """
        if self.space.is_real_space:
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
        if self.space.is_real_space:
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

        This method is only available in Python 2.
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
        return complex(self.data[(0,) * self.ndim])

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
            type as ``self.space``, however **only** using the minimal
            set of parameters ``shape``, ``dtype`` and ``order``. If more
            properties (e.g. weighting) should be propagated, this method
            must be overridden.

        Parameters
        ----------
        ufunc : `numpy.ufunc`
            Ufunc that should be called on ``self``.
        method : str
            Method on ``ufunc`` that should be called on ``self``.
            Possible values:

            ``'__call__'``, ``'accumulate'``, ``'at'``, ``'outer'``,
            ``'reduce'``, ``'reduceat'``

        input1, ..., inputN:
            Positional arguments to ``ufunc.method``.
        kwargs:
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
        # Unwrap out if provided. The output parameters are all wrapped
        # in one tuple, even if there is only one.
        out_tuple = kwargs.pop('out', ())

        # We allow our own tensors and `numpy.ndarray` objects as `out`
        if not all(isinstance(out, (type(self), np.ndarray)) or out is None
                   for out in out_tuple):
            return NotImplemented

        # Convert inputs that are ODL tensors to arrays so that the
        # native Numpy ufunc is called later
        inputs = tuple(
            inp.asarray() if isinstance(inp, type(self)) else inp
            for inp in inputs)

        out = out1 = out2 = None
        if len(out_tuple) == 1:
            out = out_tuple[0]
        elif len(out_tuple) == 2:
            out1 = out_tuple[0]
            out2 = out_tuple[1]

        # Arguments for `writable_array` and/or space constructors
        order = str(kwargs.get('order', self.order)).upper()
        array_kwargs = {'order': order}
        order = 'A' if order == 'K' else order  # We don't use 'K'

        out_dtype = kwargs.get('dtype', None)
        if out_dtype is not None:
            array_kwargs['dtype'] = out_dtype

        exponent = self.space.exponent
        weighting = self.space.weighting

        # Trivial context to be used when an output is `None`
        class CtxNone(object):
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
                        spc_kwargs = {'weighting': weighting}
                    else:
                        spc_kwargs = {}
                    out_space = type(self.space)(self.shape, res.dtype, order,
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
                    out1_space = type(self.space)(self.shape, res1.dtype,
                                                  order)
                    out1 = out1_space.element(res1)
                if out2 is None:
                    out2_space = type(self.space)(self.shape, res2.dtype,
                                                  order)
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
                    if res.shape != self.shape:
                        # Don't propagate weighting if shape changes
                        weighting = NumpyTensorSpaceNoWeighting(exponent)
                    spc_kwargs = {'weighting': weighting}
                else:
                    spc_kwargs = {}

                out_space = type(self.space)(res.shape, res.dtype, order,
                                             **spc_kwargs)
                out = out_space.element(res)

            return out


def _blas_is_applicable(*args):
    """Whether BLAS routines can be applied or not.

    BLAS routines are available for single and double precision
    float or complex data only. If the arrays are non-contiguous,
    BLAS methods are usually slower, and array-writing routines do
    not work at all. Hence, only contiguous arrays are allowed.

    Parameters
    ----------
    x1,...,xN : `NumpyTensor`
        The tensors to be tested for BLAS conformity.

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
    else:
        return True


def _lincomb_impl(a, x1, b, x2, out):
    """Optimized implementation of ``out[:] = a * x1 + b * x2``."""
    size = native(x1.size)
    # TODO: this is commented out for testing
    # if size < THRESHOLD_SMALL:  # small array optimization
    #     out.data[:] = a * x1.data + b * x2.data
    #     return

    if (size > THRESHOLD_MEDIUM and
            _blas_is_applicable(x1.data, x2.data, out.data)):
        # Need flat data for BLAS, otherwise in-place does not work.
        # Raveling must happen in fixed order for non-contiguous out,
        # otherwise 'A' is applied to arrays, which makes the outcome
        # dependent on their respective contiguousness.
        if out.data.flags.f_contiguous:
            ravel_order = 'F'
        else:
            ravel_order = 'C'

        x1_arr = x1.data.ravel(order=ravel_order)
        x2_arr = x2.data.ravel(order=ravel_order)
        out_arr = out.data.ravel(order=ravel_order)
        axpy, scal, copy = linalg.blas.get_blas_funcs(
            ['axpy', 'scal', 'copy'], arrays=(x1_arr, x2_arr, out_arr))
    else:
        # TODO: test if these really work properly, e.g., with
        # non-contiguous data!
        def fallback_axpy(x1, x2, n, a):
            """Fallback axpy implementation avoiding copy."""
            if a != 0:
                # TODO: this could be unstable
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
        x1_arr = x1.data
        x2_arr = x2.data
        out_arr = out.data

    if x1 is x2 and b != 0:
        # x1 is aligned with x2 -> out = (a+b)*x1
        _lincomb_impl(a + b, x1, 0, x1, out)
    elif out is x1 and out is x2:
        # All the vectors are aligned -> out = (a+b)*out
        scal(a + b, out_arr, size)
    elif out is x1:
        # out is aligned with x1 -> out = a*out + b*x2
        if a != 1:
            scal(a, out_arr, size)
        if b != 0:
            axpy(x2_arr, out_arr, size, b)
    elif out is x2:
        # out is aligned with x2 -> out = a*x1 + b*out
        if b != 1:
            scal(b, out_arr, size)
        if a != 0:
            axpy(x1_arr, out_arr, size, a)
    else:
        # We have exhausted all alignment options, so x1 is not x2 is not out
        # We now optimize for various values of a and b
        if b == 0:
            if a == 0:  # Zero assignment -> out = 0
                out_arr[:] = 0
            else:  # Scaled copy -> out = a*x1
                copy(x1_arr, out_arr, size)
                if a != 1:
                    scal(a, out_arr, size)

        else:  # b != 0
            if a == 0:  # Scaled copy -> out = b*x2
                copy(x2_arr, out_arr, size)
                if b != 1:
                    scal(b, out_arr, size)

            elif a == 1:  # No scaling in x1 -> out = x1 + b*x2
                copy(x1_arr, out_arr, size)
                axpy(x2_arr, out_arr, size, b)
            else:  # Generic case -> out = a*x1 + b*x2
                copy(x2_arr, out_arr, size)
                if b != 1:
                    scal(b, out_arr, size)
                axpy(x1_arr, out_arr, size, a)


def _weighting(weights, exponent):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weights):
        weighting = NumpyTensorSpaceConstWeighting(weights, exponent)
    elif weights is None:
        weighting = NumpyTensorSpaceNoWeighting(exponent)
    else:  # last possibility: make an array
        arr = np.asarray(weights)
        weighting = NumpyTensorSpaceArrayWeighting(arr, exponent)
    return weighting


def npy_weighted_inner(weights):
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
    NumpyTensorSpaceConstWeighting
    NumpyTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=2.0).inner


def npy_weighted_norm(weights, exponent=2.0):
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
    NumpyTensorSpaceConstWeighting
    NumpyTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=exponent).norm


def npy_weighted_dist(weights, exponent=2.0):
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
    NumpyTensorSpaceConstWeighting
    NumpyTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=exponent).dist


def _norm_default(x):
    """Default Euclidean norm implementation."""
    if _blas_is_applicable(x.data):
        nrm2 = linalg.blas.get_blas_funcs('nrm2', dtype=x.dtype)
        norm = partial(nrm2, n=native(x.size))
    else:
        norm = np.linalg.norm
    return norm(x.data.ravel(order=x.order))


def _pnorm_default(x, p):
    """Default p-norm implementation."""
    return np.linalg.norm(x.data.ravel(order=x.order), ord=p)


def _pnorm_diagweight(x, p, w):
    """Diagonally weighted p-norm implementation."""
    # This is faster than first applying the weights and then summing with
    # BLAS dot or nrm2
    xp = np.abs(x.data.ravel(order=x.order))
    if p == float('inf'):
        xp *= w.ravel(order=x.order)  # w is a plain NumPy array
        return np.max(xp)
    else:
        xp = np.power(xp, p, out=xp)
        xp *= w.ravel(order=x.order)
        return np.sum(xp) ** (1 / p)


def _inner_default(x1, x2):
    """Default Euclidean inner product implementation."""
    if is_real_dtype(x1.dtype):
        if x1.size > THRESHOLD_MEDIUM:
            # This is as fast as BLAS dotc
            return np.tensordot(x1, x2, [range(x1.ndim)] * 2)
        else:
            # Several times faster for small arrays
            return np.dot(x1.data.ravel(order=x1.order),
                          x2.data.ravel(order=x1.order))
    else:
        # x2 as first argument because we want linearity in x1
        return np.vdot(x2.data.ravel(order=x1.order),
                       x1.data.ravel(order=x1.order))


# TODO: implement intermediate weighting schemes with arrays that are
# broadcast, i.e. between scalar and full-blown in dimensionality?


class NumpyTensorSpaceArrayWeighting(ArrayWeighting):

    """Weighting of a `NumpyTensorSpace` by an array.

    This class defines a weighting by an array that has the same shape
    as the tensor space. Since the space is not known to this class,
    no checks of shape or data type are performed.
    See ``Notes`` for mathematical details.
    """

    def __init__(self, array, exponent=2.0):
        """Initialize a new instance.

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
              \\langle A, B\\rangle_W :=
              \\langle W \odot A, B\\rangle =
              \\langle w \odot a, b\\rangle =
              b^{\mathrm{H}} (w \odot a),

          where :math:`a, b, w` are the "flattened" counterparts of
          tensors :math:`A, B, W`, respectively, :math:`b^{\mathrm{H}}`
          stands for transposed complex conjugate and :math:`w \odot a`
          for element-wise multiplication.

        - For other exponents, only norm and dist are defined. In the
          case of exponent :math:`\\infty`, the weighted norm is

          .. math::
              \| A\|_{W, \\infty} :=
              \| W \odot A\|_{\\infty} =
              \| w \odot a\|_{\\infty},

          otherwise it is (using point-wise exponentiation)

          .. math::
              \| A\|_{W, p} :=
              \| W^{1/p} \odot A\|_{p} =
              \| w^{1/p} \odot a\|_{\\infty}.

        - Note that this definition does **not** fulfill the limit
          property in :math:`p`, i.e.

          .. math::
              \| A\|_{W, p} \\not\\to
              \| A\|_{W, \\infty} \quad (p \\to \\infty)

          unless all weights are equal to 1.

        - The array :math:`W` may only have positive entries, otherwise
          it does not define an inner product or norm, respectively. This
          is not checked during initialization.
        """
        super(NumpyTensorSpaceArrayWeighting, self).__init__(
            array, impl='numpy', exponent=exponent)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.array.tobytes(), self.exponent))

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
            inner = _inner_default(x1 * self.array, x2)
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
            norm_squared = self.inner(x, x).real  # TODO: optimize?!
            if norm_squared < 0:
                norm_squared = 0.0  # Compensate for numerical error
            return float(np.sqrt(norm_squared))
        else:
            return float(_pnorm_diagweight(x, self.exponent, self.array))


class NumpyTensorSpaceConstWeighting(ConstWeighting):

    """Weighting of a `NumpyTensorSpace` by a constant.

    See ``Notes`` for mathematical details.
    """

    def __init__(self, const, exponent=2.0):
        """Initialize a new instance.

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
              \\langle a, b\\rangle_c :=
              c \, \\langle a, b\\rangle_c =
              c \, b^{\mathrm{H}} a,

          where :math:`b^{\mathrm{H}}` standing for transposed complex
          conjugate.

        - For other exponents, only norm and dist are defined. In the
          case of exponent :math:`\\infty`, the weighted norm is defined
          as

          .. math::
              \| a \|_{c, \\infty} :=
              c\, \| a \|_{\\infty},

          otherwise it is

          .. math::
              \| a \|_{c, p} :=
              c^{1/p}\, \| a \|_{p}.

        - Note that this definition does **not** fulfill the limit
          property in :math:`p`, i.e.

          .. math::
              \| a\|_{c, p} \\not\\to
              \| a \|_{c, \\infty} \quad (p \\to \\infty)

          unless :math:`c = 1`.

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
        else:
            inner = self.const * _inner_default(x1, x2)
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
        x1, x2 : `NumpyTensor`
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


class NumpyTensorSpaceNoWeighting(NoWeighting,
                                  NumpyTensorSpaceConstWeighting):

    """Weighting of a `NumpyTensorSpace` with constant 1."""

    # Implement singleton pattern for efficiency in the default case
    __instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern if ``exp==2.0``."""
        if len(args) == 0:
            exponent = kwargs.pop('exponent', 2.0)
        else:
            exponent = args[0]
            args = args[1:]

        if exponent == 2.0:
            if not cls.__instance:
                inst = super(NoWeighting, NumpyTensorSpaceNoWeighting).__new__(
                    cls, *args, **kwargs)
                cls.__instance = inst
            return cls.__instance
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        exponent : positive `float`, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        super(NumpyTensorSpaceNoWeighting, self).__init__(
            impl='numpy', exponent=exponent)


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
