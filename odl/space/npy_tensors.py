# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""NumPy implementation of tensor spaces."""

from __future__ import absolute_import, division, print_function
from future.utils import native

import ctypes
from builtins import object
from functools import partial

import numpy as np

from odl.set.sets import ComplexNumbers, RealNumbers
from odl.space.base_tensors import TensorSpace
from odl.util import (
    dtype_str, getargspec, is_numeric_dtype, is_real_dtype, signature_string)

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
            Default: 2.0

        Other Parameters
        ----------------
        weighting : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weighting``:

            - ``None``: no weighting, i.e. weighting with ``1.0`` (default).
            - ``float``: Weighting by a constant.
            - `array-like`: Pointwise weighting by an array.

        See Also
        --------
        odl.space.space_utils.rn : constructor for real tensor spaces
        odl.space.space_utils.cn : constructor for complex tensor spaces
        odl.space.space_utils.tensor_space :
            constructor for tensor spaces of arbitrary scalar data type

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
        if self.dtype.char not in self.available_dtypes():
            raise ValueError('`dtype` {!r} not supported'
                             ''.format(dtype_str(dtype)))

        weighting = kwargs.pop('weighting', None)
        if weighting is not None and not is_numeric_dtype(self.dtype):
            raise TypeError(
                'cannot use `weighting` with non-numeric `dtype` {}'
                ''.format(self.dtype)
            )
        exponent = kwargs.pop('exponent', 2.0)

        # Exponent and weighting
        self.__exponent = float(exponent)

        if weighting is None:
            weighting = 1.0

        if np.isscalar(weighting):
            if weighting <= 0:
                raise ValueError(
                    'scalar `weighting` must be positive, got {}'
                    ''.format(weighting)
                )
            self.__weighting = float(weighting)
            self.__weighting_type = 'const'
        else:
            weighting = np.atleast_1d(weighting)
            if weighting.shape != self.shape:
                raise ValueError(
                    '`weighting` array must have the same shape as this '
                    'space, but {} != {}'
                    ''.format(weighting.shape, self.shape)
                )
            if not is_real_dtype(weighting.dtype):
                raise ValueError(
                    '`weighting.dtype` must be real, got array with dtype {}'
                    ''.format(dtype_str(weighting.dtype))
                )
            self.__weighting = weighting
            self.__weighting_type = 'array'

        # Caching
        self.__ufuncs = None
        self.__reduce = None

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
        """This space's weighting factor(s)."""
        return self.__weighting

    @property
    def weighting_type(self):
        """This space's type of weighting."""
        return self.__weighting_type

    @property
    def is_weighted(self):
        """Return ``True`` if the space is not weighted by constant 1.0."""
        return not (self.weighting_type == 'const' and self.weighting == 1.0)

    @property
    def exponent(self):
        """Exponent of the norm and the distance."""
        return self.__exponent

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
        element : `numpy.ndarray`
            The new element, created from ``inp`` or from scratch.

        Examples
        --------
        Without arguments, an uninitialized element is created. With an
        array-like input, the element can be initialized:

        >>> space = odl.rn(3)
        >>> empty = space.element()
        >>> empty.shape
        (3,)
        >>> empty in space
        True
        >>> x = space.element([1, 2, 3])
        >>> x
        array([ 1.,  2.,  3.])

        If the input already is a `numpy.ndarray` of correct `shape` and
        `dtype`, a view will be created that shares memory with the original
        array. Mutations will affect both:

        >>> arr = np.array([1, 2, 3], dtype=float)
        >>> elem = odl.rn(3).element(arr)
        >>> elem[0] = 0
        >>> elem
        array([ 0.,  2.,  3.])
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
        array([[1, 2, 3],
               [4, 5, 6]])
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

            return arr

        elif inp is None and data_ptr is not None:
            if order is None:
                raise ValueError('`order` cannot be None for element '
                                 'creation from pointer')

            ctype_array_def = ctypes.c_byte * self.nbytes
            as_ctype_array = ctype_array_def.from_address(data_ptr)
            as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
            arr = as_numpy_array.view(dtype=self.dtype)
            arr = arr.reshape(self.shape, order=order)
            return arr

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
            return arr

        else:
            raise TypeError('cannot provide both `inp` and `data_ptr`')

    def zero(self):
        """Return a tensor of all zeros.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> x = space.zero()
        >>> x
        array([ 0.,  0.,  0.])
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
        array([ 1.,  1.,  1.])
        """
        return self.element(np.ones(self.shape, dtype=self.dtype,
                                    order=self.default_order))

    @staticmethod
    def available_dtypes():
        """Return the set of data types available in this implementation.

        Notes
        -----
        This is all dtypes available in Numpy. See ``numpy.sctypes``
        for more information.

        The available dtypes may depend on the specific system used.
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
        array([ 0.,  1.,  3.])
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
        return _weighted_dist(x1, x2, self.exponent, self.weighting)

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
        return _weighted_norm(x, self.exponent, self.weighting)

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
        return self.field.element(_weighted_inner(x1, x2, self.weighting))

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
        array([-1.,  0., -3.])
        >>> out = space.element()
        >>> result = space.multiply(x, y, out=out)
        >>> result
        array([-1.,  0., -3.])
        >>> result is out
        True
        """
        np.multiply(x1, x2, out=out)

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
        array([ 2.,  0.,  2.])
        >>> out = space.element()
        >>> result = space.divide(x, y, out=out)
        >>> result
        array([ 2.,  0.,  2.])
        >>> result is out
        True
        """
        np.divide(x1, x2, out=out)

    @property
    def ufuncs(self):
        """Access to NumPy ufuncs."""
        if self.__ufuncs is not None:
            return self.__ufuncs

        class NumpyTensorSpaceUfuncs(object):

            """Accessor class for Ufuncs on tensor spaces."""

            def __getattr__(self, name):
                """Return ``self.name``."""
                attr = getattr(np, name, None)
                if not isinstance(attr, np.ufunc):
                    raise ValueError('{!r} is not a ufunc'.format(name))
                return attr

        self.__ufuncs = NumpyTensorSpaceUfuncs()
        return self.__ufuncs

    @property
    def reduce(self):
        """Access to NumPy reductions."""
        if self.__reduce is not None:
            return self.__reduce

        class NumpyTensorSpaceReduce(object):

            """Accessor class for reductions on tensor spaces."""

            def __getattr__(self, name):
                """Return ``self.name``."""
                attr = getattr(np, name, None)
                try:
                    spec = getargspec(attr)
                except (ValueError, TypeError):
                    raise ValueError(
                        '{!r} is not a valid reduction'.format(name)
                    )
                if 'keepdims' not in spec.args:
                    raise ValueError(
                        '{!r} is not a valid reduction'.format(name)
                    )
                return attr

        self.__reduce = NumpyTensorSpaceReduce()
        return self.__reduce

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

        Of course, random garbage is not in the space:

        >>> spc = odl.tensor_space((2, 3), dtype='uint64')
        >>> None in spc
        False
        >>> object in spc
        False
        >>> False in spc
        False
        """
        # TODO: may need adaption
        return (
            isinstance(other, np.ndarray)
            and other.shape == self.shape
            and other.dtype == self.dtype
        )

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

        if self.weighting_type != getattr(other, 'weighting_type', None):
            return False

        weighting_equal = (
            (
                self.weighting_type == 'const'
                and self.weighting == other.weighting
            ) or (
                self.weighting_type == 'array'
                and self.weighting is other.weighting
            )
        )

        return (
            super(NumpyTensorSpace, self).__eq__(other)
            and self.exponent == other.exponent
            and weighting_equal
        )

    def __hash__(self):
        """Return ``hash(self)``."""
        if self.weighting_type == 'const':
            weighting_hash = hash(self.weighting)
        elif self.weighting_type == 'array':
            weighting_hash = hash(self.weighting.tobytes())
        else:
            raise RuntimeError

        return hash(
            (
                super(NumpyTensorSpace, self).__hash__(),
                self.exponent,
                weighting_hash
            )
        )

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

        class NpyTensorSpacebyaxis(object):

            """Helper class for indexing by axis."""

            def __getitem__(self, indices):
                """Return ``self[indices]``."""
                try:
                    iter(indices)
                except TypeError:
                    newshape = space.shape[indices]
                else:
                    newshape = tuple(space.shape[i] for i in indices)

                if space.weighting_type == 'const':
                    weighting = space.weighting
                else:
                    # Can't preserve pointwise weighting, no idea how to
                    # remove axes
                    weighting = 1.0

                return type(space)(newshape, space.dtype, weighting=weighting)

            def __repr__(self):
                """Return ``repr(self)``."""
                return repr(space) + '.byaxis'

        return NpyTensorSpacebyaxis()

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.weighting_type == 'const':
            if self.weighting == 1.0:
                weight_str = ''
            else:
                weight_str = 'weighting=' + str(self.weighting)
        else:
            weight_str = 'weighting=' + np.array2string(
                self.weighting, separator=', '
            )

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
        if weight_str:
            inner_str += ', ' + weight_str

        return '{}({})'.format(ctor_name, inner_str)


def _blas_is_applicable(*args):
    """Whether BLAS routines can be applied or not.

    BLAS routines are available for single and double precision
    float or complex data only. If the arrays are non-contiguous,
    BLAS methods are usually slower, and array-writing routines do
    not work at all. Hence, only contiguous arrays are allowed.

    Parameters
    ----------
    x1,...,xN : numpy.ndarray
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
        x1_arr = x1
        x2_arr = x2
        out_arr = out

    else:
        # Need flat data for BLAS, otherwise in-place does not work.
        # Raveling must happen in fixed order for non-contiguous out,
        # otherwise 'A' is applied to arrays, which makes the outcome
        # dependent on their respective contiguousness.
        if out.flags.f_contiguous:
            ravel_order = 'F'
        else:
            ravel_order = 'C'

        x1_arr = x1.ravel(order=ravel_order)
        x2_arr = x2.ravel(order=ravel_order)
        out_arr = out.ravel(order=ravel_order)
        axpy, scal, copy = scipy.linalg.blas.get_blas_funcs(
            ['axpy', 'scal', 'copy'], arrays=(x1_arr, x2_arr, out_arr))

    if x1 is x2 and b != 0:
        # x1 is aligned with x2 -> out = (a+b)*x1
        _lincomb_impl(a + b, x1, 0, x1, out)
    elif out is x1 and out is x2:
        # All the vectors are aligned -> out = (a+b)*out
        if (a + b) != 0:
            scal(a + b, out_arr, size)
        else:
            out_arr[:] = 0
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


def _norm_default(x):
    """Default Euclidean norm implementation."""
    # Lazy import to improve `import odl` time
    import scipy.linalg

    if _blas_is_applicable(x):
        nrm2 = scipy.linalg.blas.get_blas_funcs('nrm2', dtype=x.dtype)
        norm = partial(nrm2, n=native(x.size))
    else:
        norm = np.linalg.norm
    return norm(x.ravel())


def _pnorm_default(x, p):
    """Default p-norm implementation."""
    return np.linalg.norm(x.ravel(), ord=p)


def _pnorm_diagweight(x, p, w):
    """Diagonally weighted p-norm implementation."""
    # Ravel both in the same order (w is a numpy array)
    order = 'F' if all(a.flags.f_contiguous for a in (x, w)) else 'C'

    # This is faster than first applying the weights and then summing with
    # BLAS dot or nrm2
    xp = np.abs(x.ravel(order))
    if p == float('inf'):
        xp *= w.ravel(order)
        return np.max(xp)
    else:
        xp = np.power(xp, p, out=xp)
        xp *= w.ravel(order)
        return np.sum(xp) ** (1 / p)


def _inner_default(x1, x2):
    """Default Euclidean inner product implementation."""
    # Ravel both in the same order
    order = 'F' if all(a.flags.f_contiguous for a in (x1, x2)) else 'C'

    if is_real_dtype(x1.dtype):
        if x1.size > THRESHOLD_MEDIUM:
            # This is as fast as BLAS dotc
            return np.tensordot(x1, x2, [range(x1.ndim)] * 2)
        else:
            # Several times faster for small arrays
            return np.dot(x1.ravel(order),
                          x2.ravel(order))
    else:
        # x2 as first argument because we want linearity in x1
        return np.vdot(x2.ravel(order),
                       x1.ravel(order))


# TODO: implement intermediate weighting schemes with arrays that are
# broadcast, i.e. between scalar and full-blown in dimensionality?


def _weighted_inner(x1, x2, weights):
    """Weighted inner product on a `NumpyTensorSpace`."""
    if (
        np.isscalar(weights)
        or (isinstance(weights, np.ndarray) and weights.size == 1)
    ):
        return _const_weighted_inner(x1, x2, weights)
    elif isinstance(weights, np.ndarray) and weights.shape == x1.shape:
        return _array_weighted_inner(x1, x2, weights)
    else:
        raise ValueError(
            '`weights` is neither a constant nor an adequate array'
        )


def _array_weighted_inner(x1, x2, weights):
    """Inner product weighted by an array (i.e., pointwise)."""
    inner = _inner_default(x1 * weights, x2)
    return inner.item()


def _const_weighted_inner(x1, x2, weight):
    """Inner product weighted by a constant."""
    inner = weight * _inner_default(x1, x2)
    return inner.item()


def _weighted_norm(x, p, weights):
    """Weighted p-norm on a `NumpyTensorSpace`."""
    if (
        np.isscalar(weights)
        or (isinstance(weights, np.ndarray) and weights.size == 1)
    ):
        return _const_weighted_norm(x, p, weights)
    elif isinstance(weights, np.ndarray) and weights.shape == x.shape:
        return _array_weighted_norm(x, p, weights)
    else:
        raise ValueError(
            '`weights` is neither a constant nor an adequate array'
        )


def _array_weighted_norm(x, p, weights):
    """Norm with exponent p, weighted by an array (i.e., pointwise)."""
    if p == 2.0:
        # TODO(kohr-h): optimize?!
        norm_squared = _array_weighted_inner(x, x, weights).real
        if norm_squared < 0:
            norm_squared = 0.0  # Compensate for numerical error
        return np.sqrt(norm_squared).item()
    else:
        return _pnorm_diagweight(x, p, weights).item()


def _const_weighted_norm(x, p, weight):
    """Norm with exponent p, weighted by a constant."""
    if p == 2.0:
        return (np.sqrt(weight) * _norm_default(x)).item()
    elif p == float('inf'):
        return (weight * _pnorm_default(x, float('inf'))).item()
    else:
        return (weight ** (1 / p) * _pnorm_default(x, p)).item()


def _weighted_dist(x1, x2, p, weights):
    """Weighted p-distance on a `NumpyTensorSpace`."""
    if (
        np.isscalar(weights)
        or (isinstance(weights, np.ndarray) and weights.size == 1)
    ):
        return _const_weighted_dist(x1, x2, p, weights)
    elif isinstance(weights, np.ndarray) and weights.shape == x1.shape:
        return _array_weighted_dist(x1, x2, p, weights)
    else:
        raise ValueError(
            "`weights` is neither a constant nor an adequate array"
        )


def _array_weighted_dist(x1, x2, p, weights):
    """Dist with exponent p, weighted by an array (one entry per subspace)."""
    return _array_weighted_norm(x1 - x2, p, weights)


def _const_weighted_dist(x1, x2, p, weight):
    """Dist with exponent p, weighted by a constant."""
    return _const_weighted_norm(x1 - x2, p, weight)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
