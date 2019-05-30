# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Cartesian products of `LinearSpace` instances."""

from __future__ import print_function, division, absolute_import
from itertools import product
from numbers import Integral
import numpy as np

from odl.set import LinearSpace
from odl.set.space import LinearSpaceElement
from odl.space.weighting import (
    Weighting, ArrayWeighting, ConstWeighting,
    CustomInner, CustomNorm, CustomDist)
from odl.util import is_real_dtype, signature_string, indent
from odl.util.ufuncs import ProductSpaceUfuncs


__all__ = ('ProductSpace',)


class ProductSpace(LinearSpace):

    """Cartesian product of `LinearSpace`'s.

    A product space is the Cartesian product ``X_1 x ... x X_n`` of
    linear spaces ``X_i``. It is itself a linear space, where the linear
    combination is defined component-wise. Inner product, norm and
    distance can also be defined in natural ways from the corresponding
    functions in the individual components.
    """

    def __init__(self, *spaces, **kwargs):
        r"""Initialize a new instance.

        Parameters
        ----------
        space1,...,spaceN : `LinearSpace` or int
            The individual spaces ("factors / parts") in the product
            space. Can also be given as ``space, n`` with ``n`` integer,
            in which case the power space ``space ** n`` is created.
        exponent : non-zero float or ``float('inf')``, optional
            Order of the product distance/norm, i.e.

            ``dist(x, y) = np.linalg.norm(x-y, ord=exponent)``

            ``norm(x) = np.linalg.norm(x, ord=exponent)``

            Values ``0 <= exponent < 1`` are currently unsupported
            due to numerical instability. See ``Notes`` for further
            information about the interpretation of the values.

            Default: 2.0

        field : `Field`, optional
            Scalar field of the resulting space.
            Default: ``spaces[0].field``

        weighting : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weighting``:

            ``None`` : no weighting (default)

            `Weighting` : weighting class, used directly. Such a
            class instance can be retrieved from the space by the
            `ProductSpace.weighting` property.

            `array-like` : weigh each component with one entry from the
            array. The array must be one-dimensional and have the same
            length as the number of spaces.

            float : same weighting factor in each component

        Other Parameters
        ----------------
        dist : callable, optional
            The distance function defining a metric on the space.
            It must accept two `ProductSpaceElement` arguments and
            fulfill the following mathematical conditions for any
            three space elements ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``

            By default, ``dist(x, y)`` is calculated as ``norm(x - y)``.

            Cannot be combined with: ``weighting, norm, inner``

        norm : callable, optional
            The norm implementation. It must accept an
            `ProductSpaceElement` argument, return a float and satisfy the
            following conditions for all space elements ``x, y`` and scalars
            ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``

            By default, ``norm(x)`` is calculated as ``inner(x, x)``.

            Cannot be combined with: ``weighting, dist, inner``

        inner : callable, optional
            The inner product implementation. It must accept two
            `ProductSpaceElement` arguments, return a element from
            the field of the space (real or complex number) and
            satisfy the following conditions for all space elements
            ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``

            Cannot be combined with: ``weighting, dist, norm``

        Examples
        --------
        Product of two rn spaces

        >>> r2x3 = ProductSpace(odl.rn(2), odl.rn(3))

        Powerspace of rn space

        >>> r2x2x2 = ProductSpace(odl.rn(2), 3)

        Notes
        -----
        Inner product, norm and distance are evaluated by collecting
        the result of the corresponding operation in the individual
        components and reducing the resulting vector to a single number.
        The ``exponent`` parameter influences only this last part,
        not the computations in the individual components. We give the
        exact definitions in the following:

        Let :math:`\mathcal{X} = \mathcal{X}_1 \times \dots \times
        \mathcal{X}_d` be a product space, and
        :math:`\langle \cdot, \cdot\rangle_i`,
        :math:`\lVert \cdot \rVert_i`, :math:`d_i(\cdot, \cdot)` be
        inner products, norms and distances in the respective
        component spaces.

        **Inner product:**

        .. math::
            \langle x, y \rangle = \sum_{i=1}^d \langle x_i, y_i \rangle_i

        **Norm:**

        - :math:`p < \infty`:

        .. math::
            \lVert x\rVert =
            \left( \sum_{i=1}^d \lVert x_i \rVert_i^p \right)^{1/p}

        - :math:`p = \infty`:

        .. math::
            \lVert x\rVert = \max_i \lVert x_i \rVert_i

        **Distance:**

        - :math:`p < \infty`:

        .. math::
            d(x, y) = \left( \sum_{i=1}^d d_i(x_i, y_i)^p \right)^{1/p}

        - :math:`p = \infty`:

        .. math::
            d(x, y) = \max_i d_i(x_i, y_i)

        To implement own versions of these functions, you can use
        the following snippet to gather the vector of norms (analogously
        for inner products and distances)::

            norms = np.fromiter(
                (xi.norm() for xi in x),
                dtype=np.float64, count=len(x))

        See Also
        --------
        ProductSpaceArrayWeighting
        ProductSpaceConstWeighting
        """
        field = kwargs.pop('field', None)
        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weighting = kwargs.pop('weighting', None)
        exponent = float(kwargs.pop('exponent', 2.0))
        if kwargs:
            raise TypeError('got unexpected keyword arguments: {}'
                            ''.format(kwargs))

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if sum(x is None for x in (dist, norm, inner, weighting)) < 3:
            raise ValueError('invalid combination of options weighting, '
                             'dist, norm and inner')

        if any(x is not None for x in (dist, norm, inner)) and exponent != 2.0:
            raise ValueError('`exponent` cannot be used together with '
                             'inner, norm or dist')

        # Make a power space if the second argument is an integer.
        # For the case that the integer is 0, we already set the field here.
        if len(spaces) == 2 and isinstance(spaces[1], Integral):
            field = spaces[0].field
            spaces = [spaces[0]] * spaces[1]

        # Validate the space arguments
        if not all(isinstance(spc, LinearSpace) for spc in spaces):
            raise TypeError(
                'all arguments must be `LinearSpace` instances, or the '
                'first argument must be `LinearSpace` and the second '
                'integer; got {!r}'.format(spaces))
        if not all(spc.field == spaces[0].field for spc in spaces):
            raise ValueError('all spaces must have the same field')

        # Assign spaces and field
        self.__spaces = tuple(spaces)

        # Cache for efficiency
        self.__is_power_space = all(spc == self.spaces[0]
                                    for spc in self.spaces[1:])

        # Assing or infer field
        if field is None:
            if len(self) == 0:
                raise ValueError('no spaces provided, cannot deduce field')
            else:
                field = self.spaces[0].field

        super(ProductSpace, self).__init__(field)

        # Assign weighting
        if weighting is not None:
            if isinstance(weighting, Weighting):
                self.__weighting = weighting
            elif np.isscalar(weighting):
                self.__weighting = ProductSpaceConstWeighting(
                    weighting, exponent)
            elif weighting is None:
                # Need to wait until dist, norm and inner are handled
                pass
            else:  # last possibility: make a product space element
                arr = np.asarray(weighting)
                if arr.dtype == object:
                    raise ValueError('invalid weighting argument {}'
                                     ''.format(weighting))
                if arr.ndim == 1:
                    self.__weighting = ProductSpaceArrayWeighting(
                        arr, exponent)
                else:
                    raise ValueError('weighting array has {} dimensions, '
                                     'expected 1'.format(arr.ndim))

        elif dist is not None:
            self.__weighting = ProductSpaceCustomDist(dist)
        elif norm is not None:
            self.__weighting = ProductSpaceCustomNorm(norm)
        elif inner is not None:
            self.__weighting = ProductSpaceCustomInner(inner)
        else:  # all None -> no weighing
            self.__weighting = ProductSpaceConstWeighting(1.0, exponent)

    def __len__(self):
        """Return ``len(self)``.

        This length is the number of spaces at the top level only,
        and is equal to ``self.shape[0]``.
        """
        return len(self.spaces)

    @property
    def nbytes(self):
        """Total number of bytes in memory used by an element of this space."""
        return sum(spc.nbytes for spc in self.spaces)

    @property
    def shape(self):
        """Total spaces per axis, computed recursively.

        The recursion ends at the fist level that does not have a shape.

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> pspace = odl.ProductSpace(r2, r3)
        >>> pspace.shape
        (2,)
        >>> pspace2 = odl.ProductSpace(pspace, 3)
        >>> pspace2.shape
        (3, 2)

        If the space is a "pure" product space, shape recurses all the way
        into the components:

        >>> r2_2 = odl.ProductSpace(r2, 3)
        >>> r2_2.shape
        (3, 2)
        """
        if len(self) == 0:
            return ()
        elif self.is_power_space:
            try:
                sub_shape = self[0].shape
            except AttributeError:
                sub_shape = ()
        else:
            sub_shape = ()

        return (len(self),) + sub_shape

    @property
    def size(self):
        """Total number of involved spaces, computed recursively.

        The recursion ends at the fist level that does not comprise
        a *power* space, i.e., which is not made of equal spaces.

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> pspace = odl.ProductSpace(r2, r3)
        >>> pspace.size
        2
        >>> pspace2 = odl.ProductSpace(pspace, 3)
        >>> pspace2.size
        6
        """
        return (0 if self.shape == () else
                int(np.prod(self.shape, dtype='int64')))

    @property
    def spaces(self):
        """A tuple containing all spaces."""
        return self.__spaces

    @property
    def is_power_space(self):
        """``True`` if all member spaces are equal."""
        return self.__is_power_space

    @property
    def exponent(self):
        """Exponent of the product space norm/dist, ``None`` for custom."""
        return self.weighting.exponent

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.__weighting

    @property
    def is_weighted(self):
        """Return ``True`` if the space is not weighted by constant 1.0."""
        return not (
            isinstance(self.weighting, ProductSpaceConstWeighting) and
            self.weighting.const == 1.0)

    @property
    def dtype(self):
        """The data type of this space.

        This is only well defined if all subspaces have the same dtype.

        Raises
        ------
        AttributeError
            If any of the subspaces does not implement `dtype` or if the dtype
            of the subspaces does not match.
        """
        dtypes = [space.dtype for space in self.spaces]

        if all(dtype == dtypes[0] for dtype in dtypes):
            return dtypes[0]
        else:
            raise AttributeError("`dtype`'s of subspaces not equal")

    @property
    def is_real(self):
        """True if this is a space of real valued vectors."""
        return all(spc.is_real for spc in self.spaces)

    @property
    def is_complex(self):
        """True if this is a space of complex valued vectors."""
        return all(spc.is_complex for spc in self.spaces)

    @property
    def real_space(self):
        """Variant of this space with real dtype."""
        return ProductSpace(*[space.real_space for space in self.spaces])

    @property
    def complex_space(self):
        """Variant of this space with complex dtype."""
        return ProductSpace(*[space.complex_space for space in self.spaces])

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
        newspace : `ProductSpace`
            Version of this space with given data type.
        """
        if dtype is None:
            # Need to filter this out since Numpy iterprets it as 'float'
            raise ValueError('`None` is not a valid data type')

        dtype = np.dtype(dtype)
        current_dtype = getattr(self, 'dtype', object)

        if dtype == current_dtype:
            return self
        else:
            return ProductSpace(*[space.astype(dtype)
                                  for space in self.spaces])

    def element(self, inp=None, cast=True):
        """Create an element in the product space.

        Parameters
        ----------
        inp : optional
            If ``inp`` is ``None``, a new element is created from
            scratch by allocation in the spaces. If ``inp`` is
            already an element of this space, it is re-wrapped.
            Otherwise, a new element is created from the
            components by calling the ``element()`` methods
            in the component spaces.
        cast : bool, optional
            If ``True``, casting is allowed. Otherwise, a ``TypeError``
            is raised for input that is not a sequence of elements of
            the spaces that make up this product space.

        Returns
        -------
        element : `ProductSpaceElement`
            The new element

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> vec_2, vec_3 = r2.element(), r3.element()
        >>> r2x3 = ProductSpace(r2, r3)
        >>> vec_2x3 = r2x3.element()
        >>> vec_2.space == vec_2x3[0].space
        True
        >>> vec_3.space == vec_2x3[1].space
        True

        Create an element of the product space

        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> prod = ProductSpace(r2, r3)
        >>> x2 = r2.element([1, 2])
        >>> x3 = r3.element([1, 2, 3])
        >>> x = prod.element([x2, x3])
        >>> x
        ProductSpace(rn(2), rn(3)).element([
            [ 1.,  2.],
            [ 1.,  2.,  3.]
        ])
        """
        # If data is given as keyword arg, prefer it over arg list
        if inp is None:
            inp = [space.element() for space in self.spaces]

        if inp in self:
            return inp

        if len(inp) != len(self):
            raise ValueError('length of `inp` {} does not match length of '
                             'space {}'.format(len(inp), len(self)))

        if (all(isinstance(v, LinearSpaceElement) and v.space == space
                for v, space in zip(inp, self.spaces))):
            parts = list(inp)
        elif cast:
            # Delegate constructors
            parts = [space.element(arg)
                     for arg, space in zip(inp, self.spaces)]
        else:
            raise TypeError('input {!r} not a sequence of elements of the '
                            'component spaces'.format(inp))

        return self.element_type(self, parts)

    @property
    def examples(self):
        """Return examples from all sub-spaces."""
        for examples in product(*[spc.examples for spc in self.spaces]):
            name = ', '.join(name for name, _ in examples)
            element = self.element([elem for _, elem in examples])
            yield (name, element)

    def zero(self):
        """Create the zero element of the product space.

        The i-th component of the product space zero element is the
        zero element of the i-th space in the product.

        Parameters
        ----------
        None

        Returns
        -------
        zero : ProductSpaceElement
            The zero element in the product space.

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> zero_2, zero_3 = r2.zero(), r3.zero()
        >>> r2x3 = ProductSpace(r2, r3)
        >>> zero_2x3 = r2x3.zero()
        >>> zero_2 == zero_2x3[0]
        True
        >>> zero_3 == zero_2x3[1]
        True
        """
        return self.element([space.zero() for space in self.spaces])

    def one(self):
        """Create the one element of the product space.

        The i-th component of the product space one element is the
        one element of the i-th space in the product.

        Parameters
        ----------
        None

        Returns
        -------
        one : ProductSpaceElement
            The one element in the product space.

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> one_2, one_3 = r2.one(), r3.one()
        >>> r2x3 = ProductSpace(r2, r3)
        >>> one_2x3 = r2x3.one()
        >>> one_2 == one_2x3[0]
        True
        >>> one_3 == one_2x3[1]
        True
        """
        return self.element([space.one() for space in self.spaces])

    def _lincomb(self, a, x, b, y, out):
        """Linear combination ``out = a*x + b*y``."""
        for space, xp, yp, outp in zip(self.spaces, x.parts, y.parts,
                                       out.parts):
            space._lincomb(a, xp, b, yp, outp)

    def _dist(self, x1, x2):
        """Distance between two elements."""
        return self.weighting.dist(x1, x2)

    def _norm(self, x):
        """Norm of an element."""
        return self.weighting.norm(x)

    def _inner(self, x1, x2):
        """Inner product of two elements."""
        return self.weighting.inner(x1, x2)

    def _multiply(self, x1, x2, out):
        """Product ``out = x1 * x2``."""
        for spc, xp, yp, outp in zip(self.spaces, x1.parts, x2.parts,
                                     out.parts):
            spc._multiply(xp, yp, outp)

    def _divide(self, x1, x2, out):
        """Quotient ``out = x1 / x2``."""
        for spc, xp, yp, outp in zip(self.spaces, x1.parts, x2.parts,
                                     out.parts):
            spc._divide(xp, yp, outp)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `ProductSpace` instance, has
            the same length and the same factors. ``False`` otherwise.

        Examples
        --------
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> rn, rm = odl.rn(2), odl.rn(3)
        >>> r2x3, rnxm = ProductSpace(r2, r3), ProductSpace(rn, rm)
        >>> r2x3 == rnxm
        True
        >>> r3x2 = ProductSpace(r3, r2)
        >>> r2x3 == r3x2
        False
        >>> r5 = ProductSpace(*[odl.rn(1)]*5)
        >>> r2x3 == r5
        False
        >>> r5 = odl.rn(5)
        >>> r2x3 == r5
        False
        """
        if other is self:
            return True
        else:
            return (isinstance(other, ProductSpace) and
                    len(self) == len(other) and
                    self.weighting == other.weighting and
                    all(x == y for x, y in zip(self.spaces,
                                               other.spaces)))

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.spaces, self.weighting))

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Examples
        --------
        Integers are used to pick components, slices to pick ranges:

        >>> r2, r3, r4 = odl.rn(2), odl.rn(3), odl.rn(4)
        >>> pspace = odl.ProductSpace(r2, r3, r4)
        >>> pspace[1]
        rn(3)
        >>> pspace[1:]
        ProductSpace(rn(3), rn(4))

        With lists, arbitrary components can be stacked together:

        >>> pspace[[0, 2, 1, 2]]
        ProductSpace(rn(2), rn(4), rn(3), rn(4))

        Tuples, i.e. multi-indices, will recursively index higher-order
        product spaces. However, remaining indices cannot be passed
        down to component spaces that are not product spaces:

        >>> pspace2 = odl.ProductSpace(pspace, 3)  # 2nd order product space
        >>> pspace2
        ProductSpace(ProductSpace(rn(2), rn(3), rn(4)), 3)
        >>> pspace2[0]
        ProductSpace(rn(2), rn(3), rn(4))
        >>> pspace2[1, 0]
        rn(2)
        >>> pspace2[:-1, 0]
        ProductSpace(rn(2), 2)
        """
        if isinstance(indices, Integral):
            return self.spaces[indices]

        elif isinstance(indices, slice):
            return ProductSpace(*self.spaces[indices], field=self.field)

        elif isinstance(indices, tuple):
            # Use tuple indexing for recursive product spaces, i.e.,
            # pspace[0, 0] == pspace[0][0]
            if not indices:
                return self
            idx = indices[0]
            if isinstance(idx, Integral):
                # Single integer in tuple, picking that space and passing
                # through the rest of the tuple. If the picked space
                # is not a product space and there are still indices left,
                # raise an error.
                space = self.spaces[idx]
                rest_indcs = indices[1:]
                if not rest_indcs:
                    return space
                elif isinstance(space, ProductSpace):
                    return space[rest_indcs]
                else:
                    raise IndexError('too many indices for recursive '
                                     'product space: remaining indices '
                                     '{}'.format(rest_indcs))
            elif isinstance(idx, slice):
                # Doing the same as with single integer with all spaces
                # in the slice, but wrapping the result into a ProductSpace.
                spaces = self.spaces[idx]
                rest_indcs = indices[1:]
                if len(spaces) == 0 and rest_indcs:
                    # Need to catch this situation since the code further
                    # down doesn't trigger an error
                    raise IndexError('too many indices for recursive '
                                     'product space: remaining indices '
                                     '{}'.format(rest_indcs))
                if not rest_indcs:
                    return ProductSpace(*spaces)
                elif all(isinstance(space, ProductSpace) for space in spaces):
                    return ProductSpace(
                        *(space[rest_indcs] for space in spaces),
                        field=self.field)
                else:
                    raise IndexError('too many indices for recursive '
                                     'product space: remaining indices '
                                     '{}'.format(rest_indcs))
            else:
                raise TypeError('index tuple can only contain'
                                'integers or slices')

        elif isinstance(indices, list):
            return ProductSpace(*[self.spaces[i] for i in indices],
                                field=self.field)

        else:
            raise TypeError('`indices` must be integer, slice, tuple or '
                            'list, got {!r}'.format(indices))

    def __str__(self):
        """Return ``str(self)``."""
        if len(self) == 0:
            return '{}'
        elif self.is_power_space:
            return '({}) ** {}'.format(self.spaces[0], len(self))
        else:
            return ' x '.join(str(space) for space in self.spaces)

    def __repr__(self):
        """Return ``repr(self)``."""
        weight_str = self.weighting.repr_part
        edgeitems = np.get_printoptions()['edgeitems']
        if len(self) == 0:
            posargs = []
            posmod = ''
            optargs = [('field', self.field, None)]
            oneline = True
        elif self.is_power_space:
            posargs = [self.spaces[0], len(self)]
            posmod = '!r'
            optargs = []
            oneline = True
        elif self.size <= 2 * edgeitems:
            posargs = self.spaces
            posmod = '!r'
            optargs = []
            argstr = ', '.join(repr(s) for s in self.spaces)
            oneline = (len(argstr + weight_str) <= 40 and
                       '\n' not in argstr + weight_str)
        else:
            posargs = (self.spaces[:edgeitems] +
                       ('...',) +
                       self.spaces[-edgeitems:])
            posmod = ['!r'] * edgeitems + ['!s'] + ['!r'] * edgeitems
            optargs = []
            oneline = False

        if oneline:
            inner_str = signature_string(posargs, optargs, sep=', ',
                                         mod=[posmod, '!r'])
            if weight_str:
                inner_str = ', '.join([inner_str, weight_str])
            return '{}({})'.format(self.__class__.__name__, inner_str)
        else:
            inner_str = signature_string(posargs, optargs, sep=',\n',
                                         mod=[posmod, '!r'])
            if weight_str:
                inner_str = ',\n'.join([inner_str, weight_str])
            return '{}(\n{}\n)'.format(self.__class__.__name__,
                                       indent(inner_str))

    @property
    def element_type(self):
        """`ProductSpaceElement`"""
        return ProductSpaceElement


class ProductSpaceElement(LinearSpaceElement):

    """Elements of a `ProductSpace`."""

    def __init__(self, space, parts):
        """Initialize a new instance."""
        super(ProductSpaceElement, self).__init__(space)
        self.__parts = tuple(parts)

    @property
    def parts(self):
        """Parts of this product space element."""
        return self.__parts

    @property
    def shape(self):
        """Number of values per axis in ``self``, computed recursively.

        The recursion ends at the fist level that does not have a shape.

        Raises
        ------
        ValueError
            If a `ProductSpace` is encountered that is not a power space.

        See Also
        --------
        ProductSpace.shape

        Examples
        --------
        >>> r4_3 = odl.ProductSpace(odl.rn(4), 3)
        >>> x = r4_3.element()
        >>> x.shape
        (3, 4)
        >>> r4_2_3 = odl.ProductSpace(r4_3, 2)
        >>> y = r4_2_3.element()
        >>> y.shape
        (2, 3, 4)
        """
        return self.space.shape

    @property
    def ndim(self):
        """Number axes in ``self``, computed recursively.

        Raises
        ------
        ValueError
            If a `ProductSpace` is encountered that is not a power space.

        See Also
        --------
        shape

        Examples
        --------
        >>> r4_3 = odl.ProductSpace(odl.rn(4), 3)
        >>> x = r4_3.element()
        >>> x.ndim
        2
        >>> r4_2_3 = odl.ProductSpace(r4_3, 2)
        >>> y = r4_2_3.element()
        >>> y.ndim
        3
        """
        return len(self.shape)

    @property
    def size(self):
        """Total number of involved spaces, computed recursively.

        See Also
        --------
        ProductSpace.size
        """
        return int(np.prod(self.shape))

    @property
    def dtype(self):
        """The data type of the space of this element."""
        return self.space.dtype

    def __len__(self):
        """Return ``len(self)``."""
        return len(self.space)

    @property
    def nbytes(self):
        """Total number of bytes in memory used by this element."""
        return self.space.nbytes

    def __eq__(self, other):
        """Return ``self == other``.

        Overrides the default `LinearSpace` method since it is implemented with
        the distance function, which is prone to numerical errors. This
        function checks equality per component.
        """
        if other is self:
            return True
        elif other not in self.space:
            return False
        else:
            return all(sp == op for sp, op in zip(self.parts, other.parts))

    def __getitem__(self, indices):
        """Return ``self[indices]``."""
        if isinstance(indices, Integral):
            return self.parts[indices]
        elif isinstance(indices, slice):
            return self.space[indices].element(self.parts[indices])
        elif isinstance(indices, list):
            out_parts = [self.parts[i] for i in indices]
            return self.space[indices].element(out_parts)
        elif isinstance(indices, tuple):
            if len(indices) == 0:
                return ProductSpace().element()
            elif len(indices) == 1:
                # Tuple with a single entry - we just unpack and delegate
                return self[indices[0]]
            else:
                # Tuple with multiple entries
                if isinstance(indices[0], Integral):
                    # In case the first entry is an integer, we drop the
                    # axis and return directly from `parts`
                    return self.parts[indices[0]][indices[1:]]
                else:
                    # indices[0] is a slice or list. We first retrieve the
                    # parts indexed in this axis.
                    # In any case we know that we want to keep this axis.
                    if isinstance(indices[0], list):
                        part = [self.parts[i] for i in indices[0]]
                    else:
                        part = self.parts[indices[0]]

                    if (len(indices[1:]) == 1 and
                            not all(isinstance(p, ProductSpaceElement)
                                    for p in part)):
                        # This case means we have "hit the bottom", i.e.,
                        # there are non-ProductSpaces involved. In order
                        # not to retrieve scalar values from these
                        # elements, we use a slice of size 1.
                        idx = indices[1]
                        indexed = [p[idx:idx + 1] for p in part]
                    else:
                        # Here we're still in the "product space chain",
                        # so we can use recursion to go on.
                        indexed = [p[indices[1:]] for p in part]

                    # Finally make a wrapping space for the indexed elements
                    new_space = ProductSpace(*(p.space for p in indexed))
                    return new_space.element(indexed)
        else:
            raise TypeError('bad index type {}'.format(type(indices)))

    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``."""
        # Get the parts to which we assign values
        if isinstance(indices, Integral):
            indexed_parts = (self.parts[indices],)
            values = (values,)
        elif isinstance(indices, slice):
            indexed_parts = self.parts[indices]
        elif isinstance(indices, list):
            indexed_parts = tuple(self.parts[i] for i in indices)
        elif isinstance(indices, tuple):
            if len(indices) == 0:
                return
            else:
                # We need to explicitly use __setitem__ here, otherwise
                # __getitem__ is used and assigned to, which fails if
                # a space like rn(3) is indexed at the very end.
                part = self.parts[indices[0]]
                if isinstance(part, LinearSpaceElement):
                    part.__setitem__(indices[1:], values)
                else:
                    # part is a tuple
                    for p in part:
                        p.__setitem__(indices[1:], values)
                return
        else:
            raise TypeError('bad index type {}'.format(type(indices)))

        # Do the assignment, with broadcasting if desired
        try:
            iter(values)
        except TypeError:
            # `values` is not iterable, assume it can be assigned to
            # all indexed parts
            for p in indexed_parts:
                p[:] = values
        else:
            # `values` is iterable; it could still represent a single
            # element of a power space.
            if self.space.is_power_space and values in self.space[0]:
                # Broadcast a single element across a power space
                for p in indexed_parts:
                    p[:] = values
            else:
                # Now we really have one assigned value per part
                if len(values) != len(indexed_parts):
                    raise ValueError(
                        'length of iterable `values` not equal to number of '
                        'indexed parts ({} != {})'
                        ''.format(len(values), len(indexed_parts)))
                for p, v in zip(indexed_parts, values):
                    p[:] = v

    def asarray(self, out=None):
        """Extract the data of this vector as a numpy array.

        Only available if `is_power_space` is True.

        The ordering is such that it commutes with indexing::

            self[ind].asarray() == self.asarray()[ind]

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype and
            shape.

        Raises
        ------
        ValueError
            If `is_power_space` is false.

        Examples
        --------
        >>> spc = odl.ProductSpace(odl.rn(3), 2)
        >>> x = spc.element([[ 1.,  2.,  3.],
        ...                  [ 4.,  5.,  6.]])
        >>> x.asarray()
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]])
        """
        if not self.space.is_power_space:
            raise ValueError('cannot use `asarray` if `space.is_power_space` '
                             'is `False`')
        else:
            if out is None:
                out = np.empty(self.shape, self.dtype)

            for i in range(len(self)):
                out[i] = np.asarray(self[i])
            return out

    def __array__(self):
        """An array representation of ``self``.

        Only available if `is_power_space` is True.

        The ordering is such that it commutes with indexing::

            np.array(self[ind]) == np.array(self)[ind]

        Raises
        ------
        ValueError
            If `is_power_space` is false.

        Examples
        --------
        >>> spc = odl.ProductSpace(odl.rn(3), 2)
        >>> x = spc.element([[ 1.,  2.,  3.],
        ...                  [ 4.,  5.,  6.]])
        >>> np.asarray(x)
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]])
        """
        return self.asarray()

    def __array_wrap__(self, array):
        """Return a new product space element wrapping the ``array``.

        Only available if `is_power_space` is ``True``.

        Parameters
        ----------
        array : `numpy.ndarray`
            Array to be wrapped.

        Returns
        -------
        wrapper : `ProductSpaceElement`
            Product space element wrapping ``array``.
        """
        return self.space.element(array)

    @property
    def ufuncs(self):
        """`ProductSpaceUfuncs`, access to Numpy style ufuncs.

        These are always available if the underlying spaces are
        `TensorSpace`.

        Examples
        --------
        >>> r22 = odl.ProductSpace(odl.rn(2), 2)
        >>> x = r22.element([[1, -2], [-3, 4]])
        >>> x.ufuncs.absolute()
        ProductSpace(rn(2), 2).element([
            [ 1.,  2.],
            [ 3.,  4.]
        ])

        These functions can also be used with non-vector arguments and
        support broadcasting, per component and even recursively:

        >>> x.ufuncs.add([1, 2])
        ProductSpace(rn(2), 2).element([
            [ 2.,  0.],
            [-2.,  6.]
        ])
        >>> x.ufuncs.subtract(1)
        ProductSpace(rn(2), 2).element([
            [ 0., -3.],
            [-4.,  3.]
        ])

        There is also support for various reductions (sum, prod, min, max):

        >>> x.ufuncs.sum()
        0.0

        Writing to ``out`` is also supported:

        >>> y = r22.element()
        >>> result = x.ufuncs.absolute(out=y)
        >>> result
        ProductSpace(rn(2), 2).element([
            [ 1.,  2.],
            [ 3.,  4.]
        ])
        >>> result is y
        True

        See Also
        --------
        odl.util.ufuncs.TensorSpaceUfuncs
            Base class for ufuncs in `TensorSpace` spaces, subspaces may
            override this for greater efficiency.
        odl.util.ufuncs.ProductSpaceUfuncs
            For a list of available ufuncs.
        """
        return ProductSpaceUfuncs(self)

    @property
    def real(self):
        """Real part of the element.

        The real part can also be set using ``x.real = other``, where ``other``
        is array-like or scalar.

        Examples
        --------
        >>> space = odl.ProductSpace(odl.cn(3), odl.cn(2))
        >>> x = space.element([[1 + 1j, 2, 3 - 3j],
        ...                    [-1 + 2j, -2 - 3j]])
        >>> x.real
        ProductSpace(rn(3), rn(2)).element([
            [ 1.,  2.,  3.],
            [-1., -2.]
        ])

        The real part can also be set using different array-like types:

        >>> x.real = space.real_space.zero()
        >>> x
        ProductSpace(cn(3), cn(2)).element([
            [ 0.+1.j,  0.+0.j,  0.-3.j],
            [ 0.+2.j,  0.-3.j]
        ])

        >>> x.real = 1.0
        >>> x
        ProductSpace(cn(3), cn(2)).element([
            [ 1.+1.j,  1.+0.j,  1.-3.j],
            [ 1.+2.j,  1.-3.j]
        ])

        >>> x.real = [[2, 3, 4], [5, 6]]
        >>> x
        ProductSpace(cn(3), cn(2)).element([
            [ 2.+1.j,  3.+0.j,  4.-3.j],
            [ 5.+2.j,  6.-3.j]
        ])
        """
        real_part = [part.real for part in self.parts]
        return self.space.real_space.element(real_part)

    @real.setter
    def real(self, newreal):
        """Setter for the real part.

        This method is invoked by ``x.real = other``.

        Parameters
        ----------
        newreal : array-like or scalar
            Values to be assigned to the real part of this element.
        """
        try:
            iter(newreal)
        except TypeError:
            # `newreal` is not iterable, assume it can be assigned to
            # all indexed parts
            for part in self.parts:
                part.real = newreal
            return

        if self.space.is_power_space:
            try:
                # Set same value in all parts
                for part in self.parts:
                    part.real = newreal
            except (ValueError, TypeError):
                # Iterate over all parts and set them separately
                for part, new_re in zip(self.parts, newreal):
                    part.real = new_re
                pass
        elif len(newreal) == len(self):
            for part, new_re in zip(self.parts, newreal):
                part.real = new_re
        else:
            raise ValueError(
                'dimensions of the new real part does not match the space, '
                'got element {} to set real part of {}'.format(newreal, self))

    @property
    def imag(self):
        """Imaginary part of the element.

        The imaginary part can also be set using ``x.imag = other``, where
        ``other`` is array-like or scalar.


        Examples
        --------
        >>> space = odl.ProductSpace(odl.cn(3), odl.cn(2))
        >>> x = space.element([[1 + 1j, 2, 3 - 3j],
        ...                    [-1 + 2j, -2 - 3j]])
        >>> x.imag
        ProductSpace(rn(3), rn(2)).element([
            [ 1.,  0., -3.],
            [ 2., -3.]
        ])

        The imaginary part can also be set using different array-like types:

        >>> x.imag = space.real_space.zero()
        >>> x
        ProductSpace(cn(3), cn(2)).element([
            [ 1.+0.j,  2.+0.j,  3.+0.j],
            [-1.+0.j, -2.+0.j]
        ])

        >>> x.imag = 1.0
        >>> x
        ProductSpace(cn(3), cn(2)).element([
            [ 1.+1.j,  2.+1.j,  3.+1.j],
            [-1.+1.j, -2.+1.j]
        ])

        >>> x.imag = [[2, 3, 4], [5, 6]]
        >>> x
        ProductSpace(cn(3), cn(2)).element([
            [ 1.+2.j,  2.+3.j,  3.+4.j],
            [-1.+5.j, -2.+6.j]
        ])
        """
        imag_part = [part.imag for part in self.parts]
        return self.space.real_space.element(imag_part)

    @imag.setter
    def imag(self, newimag):
        """Setter for the imaginary part.

        This method is invoked by ``x.imag = other``.

        Parameters
        ----------
        newimag : array-like or scalar
            Values to be assigned to the imaginary part of this element.
        """
        try:
            iter(newimag)
        except TypeError:
            # `newimag` is not iterable, assume it can be assigned to
            # all indexed parts
            for part in self.parts:
                part.imag = newimag
            return

        if self.space.is_power_space:
            try:
                # Set same value in all parts
                for part in self.parts:
                    part.imag = newimag
            except (ValueError, TypeError):
                # Iterate over all parts and set them separately
                for part, new_im in zip(self.parts, newimag):
                    part.imag = new_im
                pass
        elif len(newimag) == len(self):
            for part, new_im in zip(self.parts, newimag):
                part.imag = new_im
        else:
            raise ValueError(
                'dimensions of the new imaginary part does not match the '
                'space, got element {} to set real part of {}}'
                ''.format(newimag, self))

    def conj(self):
        """Complex conjugate of the element."""
        complex_conj = [part.conj() for part in self.parts]
        return self.space.element(complex_conj)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> from odl import rn  # need to import rn into namespace
        >>> r2, r3 = odl.rn(2), odl.rn(3)
        >>> r2x3 = ProductSpace(r2, r3)
        >>> x = r2x3.element([[1, 2], [3, 4, 5]])
        >>> eval(repr(x)) == x
        True

        The result is readable:

        >>> x
        ProductSpace(rn(2), rn(3)).element([
            [ 1.,  2.],
            [ 3.,  4.,  5.]
        ])

        Nestled spaces work as well:

        >>> X = ProductSpace(r2x3, r2x3)
        >>> x = X.element([[[1, 2], [3, 4, 5]],[[1, 2], [3, 4, 5]]])
        >>> eval(repr(x)) == x
        True
        >>> x
        ProductSpace(ProductSpace(rn(2), rn(3)), 2).element([
            [
                [ 1.,  2.],
                [ 3.,  4.,  5.]
            ],
            [
                [ 1.,  2.],
                [ 3.,  4.,  5.]
            ]
        ])
        """
        inner_str = '[\n'
        if len(self) < 5:
            inner_str += ',\n'.join('{}'.format(
                _indent(_strip_space(part))) for part in self.parts)
        else:
            inner_str += ',\n'.join('{}'.format(
                _indent(_strip_space(part))) for part in self.parts[:3])
            inner_str += ',\n    ...\n'
            inner_str += ',\n'.join('{}'.format(
                _indent(_strip_space(part))) for part in self.parts[-1:])

        inner_str += '\n]'

        return '{!r}.element({})'.format(self.space, inner_str)

    def show(self, title=None, indices=None, **kwargs):
        """Display the parts of this product space element graphically.

        Parameters
        ----------
        title : string, optional
            Title of the figures

        indices : int, slice, tuple or list, optional
            Display parts of ``self`` in the way described in the following.

            A single list of integers selects the corresponding parts
            of this vector.

            For other tuples or lists, the first entry indexes the parts of
            this vector, and the remaining entries (if any) are used to
            slice into the parts. Handling those remaining indices is
            up to the ``show`` methods of the parts to be displayed.

            The types of the first entry trigger the following behaviors:

                - ``int``: take the part corresponding to this index
                - ``slice``: take a subset of the parts
                - ``None``: equivalent to ``slice(None)``, i.e., everything

            Typical use cases are displaying of selected parts, which can
            be achieved with a list, e.g., ``indices=[0, 2]`` for parts
            0 and 2, and plotting of all parts sliced in a certain way,
            e.g., ``indices=[None, 20, None]`` for showing all parts
            sliced with indices ``[20, None]``.

            A single ``int``, ``slice``, ``list`` or ``None`` object
            indexes the parts only, i.e., is treated roughly as
            ``(indices, Ellipsis)``. In particular, for ``None``, all
            parts are shown with default slicing.

        in_figs : sequence of `matplotlib.figure.Figure`, optional
            Update these figures instead of creating new ones. Typically
            the return value of an earlier call to ``show`` is used
            for this parameter.

        kwargs
            Additional arguments passed on to the ``show`` methods of
            the parts.

        Returns
        -------
        figs : tuple of `matplotlib.figure.Figure`
            The resulting figures. In an interactive shell, they are
            automatically displayed.

        See Also
        --------
        odl.discr.lp_discr.DiscreteLpElement.show :
            Display of a discretized function
        odl.space.base_tensors.Tensor.show :
            Display of sequence type data
        odl.util.graphics.show_discrete_data :
            Underlying implementation
        """
        if title is None:
            title = 'ProductSpaceElement'

        if indices is None:
            if len(self) < 5:
                indices = list(range(len(self)))
            else:
                indices = list(np.linspace(0, len(self) - 1, 4, dtype=int))
        else:
            if (isinstance(indices, tuple) or
                    (isinstance(indices, list) and
                     not all(isinstance(idx, Integral) for idx in indices))):
                # Tuples or lists containing non-integers index by axis.
                # We use the first index for the current pspace and pass
                # on the rest.
                indices, kwargs['indices'] = indices[0], indices[1:]

            # Support `indices=[None, 0, None]` like syntax (`indices` is
            # the first entry as of now in that case)
            if indices is None:
                indices = slice(None)

            if isinstance(indices, slice):
                indices = list(range(*indices.indices(len(self))))
            elif isinstance(indices, Integral):
                indices = [indices]
            else:
                # Use `indices` as-is
                pass

        in_figs = kwargs.pop('fig', None)
        in_figs = [None] * len(indices) if in_figs is None else in_figs

        figs = []
        parts = self[indices]
        if len(parts) == 0:
            return ()
        elif len(parts) == 1:
            # Don't extend the title if there is only one plot
            fig = parts[0].show(title=title, fig=in_figs[0], **kwargs)
            figs.append(fig)
        else:
            # Extend titles by indexed part to make them distinguishable
            for i, part, fig in zip(indices, parts, in_figs):
                fig = part.show(title='{}. Part {}'.format(title, i), fig=fig,
                                **kwargs)
                figs.append(fig)

        return tuple(figs)


# --- Add arithmetic operators that broadcast --- #


def _broadcast_arithmetic(op):
    """Return ``op(self, other)`` with broadcasting.

    Parameters
    ----------
    op : string
        Name of the operator, e.g. ``'__add__'``.

    Returns
    -------
    broadcast_arithmetic_op : function
        Function intended to be used as a method for `ProductSpaceVector`
        which performs broadcasting if possible.

    Notes
    -----
    Broadcasting is the operation of "applying an operator multiple times" in
    some sense. For example:

    .. math::
        (1, 2) + 1 = (2, 3)

    is a form of broadcasting. In this implementation, we only allow "single
    layer" broadcasting, i.e., we do not support broadcasting over several
    product spaces at once.
    """
    def _broadcast_arithmetic_impl(self, other):
        if (self.space.is_power_space and other in self.space[0]):
            results = []
            for xi in self:
                res = getattr(xi, op)(other)
                if res is NotImplemented:
                    return NotImplemented
                else:
                    results.append(res)

            return self.space.element(results)
        else:
            return getattr(LinearSpaceElement, op)(self, other)

    # Set docstring
    docstring = """Broadcasted {op}.""".format(op=op)
    _broadcast_arithmetic_impl.__doc__ = docstring

    return _broadcast_arithmetic_impl


for op in ['add', 'sub', 'mul', 'div', 'truediv']:
    for modifier in ['', 'r', 'i']:
        name = '__{}{}__'.format(modifier, op)
        setattr(ProductSpaceElement, name, _broadcast_arithmetic(name))


class ProductSpaceArrayWeighting(ArrayWeighting):

    """Array weighting for `ProductSpace`.

    This class defines a weighting that has a different value for
    each index defined in a given space.
    See ``Notes`` for mathematical details.
    """

    def __init__(self, array, exponent=2.0):
        r"""Initialize a new instance.

        Parameters
        ----------
        array : 1-dim. `array-like`
            Weighting array of the inner product.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no inner
            product is defined.

        Notes
        -----
        - For exponent 2.0, a new weighted inner product with array
          :math:`w` is defined as

          .. math::
              \langle x, y \rangle_w = \langle w \odot x, y \rangle

          with component-wise multiplication :math:`w \odot x`. For other
          exponents, only ``norm`` and ``dist`` are defined. In the case
          of exponent ``inf``, the weighted norm is

          .. math::
              \|x\|_{w,\infty} = \|w \odot x\|_\infty,

          otherwise it is

          .. math::
              \|x\|_{w,p} = \|w^{1/p} \odot x\|_p.

        - Note that this definition does **not** fulfill the limit property
          in :math:`p`, i.e.,

          .. math::
              \|x\|_{w,p} \not\to \|x\|_{w,\infty}
              \quad\text{for } p \to \infty

          unless :math:`w = (1,...,1)`. The reason for this choice
          is that the alternative with the limit property consists in
          ignoring the weights altogether.

        - The array may only have positive entries, otherwise it does not
          define an inner product or norm, respectively. This is not checked
          during initialization.
        """
        super(ProductSpaceArrayWeighting, self).__init__(
            array, impl='numpy', exponent=exponent)

    def inner(self, x1, x2):
        """Calculate the array-weighted inner product of two elements.

        Parameters
        ----------
        x1, x2 : `ProductSpaceElement`
            Elements whose inner product is calculated.

        Returns
        -------
        inner : float or complex
            The inner product of the two provided elements.
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))

        inners = np.fromiter(
            (x1i.inner(x2i) for x1i, x2i in zip(x1, x2)),
            dtype=x1[0].space.dtype, count=len(x1))

        inner = np.dot(inners, self.array)
        if is_real_dtype(x1[0].dtype):
            return float(inner)
        else:
            return complex(inner)

    def norm(self, x):
        """Calculate the array-weighted norm of an element.

        Parameters
        ----------
        x : `ProductSpaceElement`
            Element whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the provided element.
        """
        if self.exponent == 2.0:
            norm_squared = self.inner(x, x).real  # TODO: optimize?!
            return np.sqrt(norm_squared)
        else:
            norms = np.fromiter(
                (xi.norm() for xi in x), dtype=np.float64, count=len(x))
            if self.exponent in (1.0, float('inf')):
                norms *= self.array
            else:
                norms *= self.array ** (1.0 / self.exponent)

            return float(np.linalg.norm(norms, ord=self.exponent))


class ProductSpaceConstWeighting(ConstWeighting):

    """Constant weighting for `ProductSpace`.

    """

    def __init__(self, constant, exponent=2.0):
        r"""Initialize a new instance.

        Parameters
        ----------
        constant : positive float
            Weighting constant of the inner product
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no inner
            product is defined.

        Notes
        -----
        - For exponent 2.0, a new weighted inner product with constant
          :math:`c` is defined as

          .. math::
            \langle x, y \rangle_c = c\, \langle x, y \rangle.

          For other exponents, only ``norm`` and ```dist`` are defined.
          In the case of exponent ``inf``, the weighted norm is

          .. math::
              \|x\|_{c,\infty} = c\, \|x\|_\infty,

          otherwise it is

          .. math::
              \|x\|_{c,p} = c^{1/p} \, \|x\|_p.

        - Note that this definition does **not** fulfill the limit property
          in :math:`p`, i.e.,

          .. math::
              \|x\|_{c,p} \not\to \|x\|_{c,\infty}
              \quad \text{for } p \to \infty

          unless :math:`c = 1`. The reason for this choice
          is that the alternative with the limit property consists in
          ignoring the weight altogether.

        - The constant must be positive, otherwise it does not define an
          inner product or norm, respectively.
        """
        super(ProductSpaceConstWeighting, self).__init__(
            constant, impl='numpy', exponent=exponent)

    def inner(self, x1, x2):
        """Calculate the constant-weighted inner product of two elements.

        Parameters
        ----------
        x1, x2 : `ProductSpaceElement`
            Elements whose inner product is calculated.

        Returns
        -------
        inner : float or complex
            The inner product of the two provided elements.
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))

        inners = np.fromiter(
            (x1i.inner(x2i) for x1i, x2i in zip(x1, x2)),
            dtype=x1[0].space.dtype, count=len(x1))

        inner = self.const * np.sum(inners)
        return x1.space.field.element(inner)

    def norm(self, x):
        """Calculate the constant-weighted norm of an element.

        Parameters
        ----------
        x1 : `ProductSpaceElement`
            Element whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the element.
        """
        if self.exponent == 2.0:
            norm_squared = self.inner(x, x).real  # TODO: optimize?!
            return np.sqrt(norm_squared)
        else:
            norms = np.fromiter(
                (xi.norm() for xi in x), dtype=np.float64, count=len(x))

            if self.exponent in (1.0, float('inf')):
                return (self.const *
                        float(np.linalg.norm(norms, ord=self.exponent)))
            else:
                return (self.const ** (1 / self.exponent) *
                        float(np.linalg.norm(norms, ord=self.exponent)))

    def dist(self, x1, x2):
        """Calculate the constant-weighted distance between two elements.

        Parameters
        ----------
        x1, x2 : `ProductSpaceElement`
            Elements whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the elements.
        """
        dnorms = np.fromiter(
            ((x1i - x2i).norm() for x1i, x2i in zip(x1, x2)),
            dtype=np.float64, count=len(x1))

        if self.exponent == float('inf'):
            return self.const * np.linalg.norm(dnorms, ord=self.exponent)
        else:
            return (self.const ** (1 / self.exponent) *
                    np.linalg.norm(dnorms, ord=self.exponent))


class ProductSpaceCustomInner(CustomInner):

    """Class for handling a user-specified inner products."""

    def __init__(self, inner):
        """Initialize a new instance.

        Parameters
        ----------
        inner : callable
            The inner product implementation. It must accept two
            `ProductSpaceElement` arguments, return a element from
            the field of the space (real or complex number) and
            satisfy the following conditions for all space elements
            ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``
        """
        super(ProductSpaceCustomInner, self).__init__(
            impl='numpy', inner=inner)


class ProductSpaceCustomNorm(CustomNorm):

    """Class for handling a user-specified norm on `ProductSpace`.

    Note that this removes ``inner``.
    """

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : callable
            The norm implementation. It must accept a
            `ProductSpaceElement` argument, return a float and satisfy
            the following conditions for all space elements
            ``x, y`` and scalars ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``
        """
        super(ProductSpaceCustomNorm, self).__init__(norm, impl='numpy')


class ProductSpaceCustomDist(CustomDist):

    """Class for handling a user-specified distance on `ProductSpace`.

    Note that this removes ``inner`` and ``norm``.
    """

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : callable
            The distance function defining a metric on
            `ProductSpace`. It must accept two `ProductSpaceElement`
            arguments and fulfill the following mathematical conditions
            for any three space elements ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``
        """
        super(ProductSpaceCustomDist, self).__init__(dist, impl='numpy')


def _strip_space(x):
    """Strip the SPACE.element( ... ) part from a repr."""
    r = repr(x)
    space_repr = '{!r}.element('.format(x.space)
    if r.startswith(space_repr) and r.endswith(')'):
        r = r[len(space_repr):-1]
    return r


def _indent(x):
    """Indent a string by 4 characters."""
    lines = x.splitlines()
    for i, line in enumerate(lines):
        lines[i] = '    ' + line
    return '\n'.join(lines)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
