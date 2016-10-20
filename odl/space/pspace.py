# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Cartesian products of `LinearSpace` instances."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from builtins import range, str, super, zip
from future import standard_library
standard_library.install_aliases()

from numbers import Integral
from itertools import product
import numpy as np

from odl.set import LinearSpace, LinearSpaceElement
from odl.space.weighting import (
    WeightingBase, VectorWeightingBase, ConstWeightingBase, NoWeightingBase,
    CustomInnerProductBase, CustomNormBase, CustomDistBase)
from odl.util.ufuncs import ProductSpaceUFuncs
from odl.util.utility import is_real_dtype


__all__ = ('ProductSpace', 'ProductSpaceElement')


class ProductSpace(LinearSpace):

    """Cartesian product of `LinearSpace`'s.

    A product space is the Cartesian product ``X_1 x ... x X_n`` of
    linear spaces ``X_i``. It is itself a linear space, where the linear
    combination is defined component-wise. Inner product, norm and
    distance can also be defined in natural ways from the corresponding
    functions in the individual components.
    """

    def __init__(self, *spaces, **kwargs):
        """Initialize a new instance.

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

        weight : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weight``:

            ``None`` : no weighting (default)

            `WeightingBase` : weighting class, used directly. Such a
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
            This creates an intermediate array ``x - y``, which can be
            avoided by choosing ``dist_using_inner=True``.

            Cannot be combined with: ``weight, norm, inner``

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

            Cannot be combined with: ``weight, dist, inner``

        inner : callable, optional
            The inner product implementation. It must accept two
            `ProductSpaceElement` arguments, return a element from
            the field of the space (real or complex number) and
            satisfy the following conditions for all space elements
            ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``

            Cannot be combined with: ``weight, dist, norm``

        dist_using_inner : bool, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.

            Default: ``False``.

            Cannot be combined with: ``dist``

        Returns
        -------
        prodspace : `ProductSpace`

        See Also
        --------
        ProductSpaceElementWeighting
        ProductSpaceConstWeighting

        Examples
        --------
        >>> r2x3 = ProductSpace(odl.rn(2), odl.rn(3))

        Notes
        -----
        Inner product, norm and distance are evaluated by collecting
        the result of the corresponding operation in the individual
        components and reducing the resulting vector to a single number.
        The ``exponent`` parameter influences only this last part,
        not the computations in the individual components. We give the
        exact definitions in the following:

        Let :math:`\mathcal{X} = \mathcal{X}_1 \\times \dots \\times
        \mathcal{X}_d` be a product space, and
        :math:`\langle \cdot, \cdot\\rangle_i`,
        :math:`\lVert \cdot \\rVert_i`, :math:`d_i(\cdot, \cdot)` be
        inner products, norms and distances in the respective
        component spaces.

        **Inner product:**

            :math:`\langle x, y \\rangle =
            \\sum_{i=1}^d \langle x_i, y_i \\rangle_i`

        **Norm:**

        - :math:`p < \infty`:

            :math:`\lVert x\\rVert =
            \left( \sum_{i=1}^d \lVert x_i \\rVert_i^p \\right)^{1/p}`

        - :math:`p = \infty`:

            :math:`\lVert x\\rVert =
            \max_i \lVert x_i \\rVert_i`

        **Distance:**

        - :math:`p < \infty`:

            :math:`d(x, y) =
            \left( \sum_{i=1}^d d_i(x_i, y_i)^p \\right)^{1/p}`

        - :math:`p = \infty`:

            :math:`d(x, y) =
            \max_i d_i(x_i, y_i)`

        To implement own versions of these functions, you can use
        the following snippet to gather the vector of norms (analogously
        for inner products and distances)::

            norms = np.fromiter(
                (xi.norm() for xi in x),
                dtype=np.float64, count=len(x))

        """
        field = kwargs.pop('field', None)
        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weight = kwargs.pop('weight', None)
        exponent = kwargs.pop('exponent', 2.0)
        dist_using_inner = bool(kwargs.pop('dist_using_inner', False))
        if kwargs:
            raise TypeError('got unexpected keyword arguments: {}'
                            ''.format(kwargs))

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if sum(x is None for x in (dist, norm, inner, weight)) < 3:
            raise ValueError('invalid combination of options weight, '
                             'dist, norm and inner')

        if any(x is not None for x in (dist, norm, inner)) and exponent != 2.0:
            raise ValueError('`exponent` cannot be used together with '
                             'inner, norm or dist')

        # Make a power space if the second argument is an integer
        if (len(spaces) == 2 and
                isinstance(spaces[0], LinearSpace) and
                isinstance(spaces[1], Integral)):
            spaces = [spaces[0]] * spaces[1]

        # Validate the space arguments
        wrong_spaces = [spc for spc in spaces
                        if not isinstance(spc, LinearSpace)]
        if wrong_spaces:
            raise TypeError('{!r} not LinearSpace instance(s)'
                            ''.format(wrong_spaces))

        if not all(spc.field == spaces[0].field for spc in spaces):
            raise ValueError('all spaces must have the same field')

        # Assign spaces and field
        self.__spaces = tuple(spaces)
        self.__size = len(spaces)
        if field is None:
            if self.size == 0:
                raise ValueError('no spaces provided, cannot deduce field')
            field = self.spaces[0].field

        # Cache for efficiency
        self.__is_power_space = all(spc == self.spaces[0]
                                    for spc in self.spaces[1:])

        super().__init__(field)

        # Assign weighting
        if weight is not None:
            if isinstance(weight, WeightingBase):
                self.__weighting = weight
            elif np.isscalar(weight):
                self.__weighting = ProductSpaceConstWeighting(
                    weight, exponent, dist_using_inner=dist_using_inner)
            elif weight is None:
                # Need to wait until dist, norm and inner are handled
                pass
            else:  # last possibility: make a product space element
                arr = np.asarray(weight)
                if arr.dtype == object:
                    raise ValueError('invalid weight argument {}'
                                     ''.format(weight))
                if arr.ndim == 1:
                    self.__weighting = ProductSpaceElementWeighting(
                        arr, exponent, dist_using_inner=dist_using_inner)
                else:
                    raise ValueError('weighting array has {} dimensions, '
                                     'expected 1'.format(arr.ndim))

        elif dist is not None:
            self.__weighting = ProductSpaceCustomDist(dist)
        elif norm is not None:
            self.__weighting = ProductSpaceCustomNorm(norm)
        elif inner is not None:
            self.__weighting = ProductSpaceCustomInnerProduct(inner)
        else:  # all None -> no weighing
            self.__weighting = ProductSpaceNoWeighting(
                exponent, dist_using_inner=dist_using_inner)

    @property
    def size(self):
        """Number of factors."""
        return self.__size

    def __len__(self):
        """Return ``len(self)``."""
        return self.size

    @property
    def shape(self):
        """Number of spaces per axis."""
        # Currently supporting only 1d product spaces
        return (self.size,)

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
        """Return ``True`` if weighting is not `ProductSpaceNoWeighting`."""
        return not isinstance(self.weighting, ProductSpaceNoWeighting)

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
        cast : bool
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
        >>> print(x)
        {[1.0, 2.0], [1.0, 2.0, 3.0]}
        """

        # If data is given as keyword arg, prefer it over arg list
        if inp is None:
            inp = [space.element() for space in self.spaces]

        if inp in self:
            return inp

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
                    self.shape == other.shape and
                    self.weighting == other.weighting and
                    all(x == y for x, y in zip(self.spaces,
                                               other.spaces)))

    def __getitem__(self, indices):
        """Return ``self[indices]``."""
        if isinstance(indices, Integral):
            return self.spaces[indices]
        elif isinstance(indices, slice):
            return ProductSpace(*self.spaces[indices],
                                field=self.field)
        else:
            return ProductSpace(*[self.spaces[i] for i in indices],
                                field=self.field)

    def __str__(self):
        """Return ``str(self)``."""
        if self.size == 0:
            return '{}'
        elif all(self.spaces[0] == space for space in self.spaces):
            return '{' + str(self.spaces[0]) + '}^' + str(self.size)
        else:
            return ' x '.join(str(space) for space in self.spaces)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.size == 0:
            return '{}(field={})'.format(self.__class__.__name__, self.field)
        elif self.is_power_space:
            return '{}({!r}, {})'.format(self.__class__.__name__,
                                         self.spaces[0], self.size)
        else:
            inner_str = ', '.join(repr(space) for space in self.spaces)
            return '{}({})'.format(self.__class__.__name__, inner_str)

    @property
    def element_type(self):
        """`ProductSpaceElement`"""
        return ProductSpaceElement


class ProductSpaceElement(LinearSpaceElement):

    """Elements of a `ProductSpace`."""

    def __init__(self, space, parts):
        """Initialize a new instance."""
        super().__init__(space)
        self._parts = list(parts)

    @property
    def parts(self):
        """Parts of this product space element."""
        return self._parts

    @property
    def size(self):
        """Number of factors of this element's space."""
        return self.space.size

    def __len__(self):
        """Return ``len(self)``."""
        return len(self.space)

    def __eq__(self, other):
        """Return ``self == other``.

        Overrides the default `LinearSpace` method since it is
        implemented with the distance function, which is prone to
        numerical errors. This function checks equality per
        component.
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
        else:
            out_parts = [self.parts[i] for i in indices]
            return self.space[indices].element(out_parts)

    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``."""
        try:
            self.parts[indices] = values
        except TypeError:
            for i, index in enumerate(indices):
                self.parts[index] = values[i]

    @property
    def ufunc(self):
        """`ProductSpaceUFuncs`, access to Numpy style ufuncs.

        These are always available if the underlying spaces are
        `NtuplesBase`.

        Examples
        --------
        >>> r22 = odl.ProductSpace(odl.rn(2), 2)
        >>> x = r22.element([[1, -2], [-3, 4]])
        >>> x.ufunc.absolute()
        ProductSpace(rn(2), 2).element([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        These functions can also be used with non-vector arguments and
        support broadcasting, per component and even recursively:

        >>> x.ufunc.add([1, 2])
        ProductSpace(rn(2), 2).element([
            [2.0, 0.0],
            [-2.0, 6.0]
        ])
        >>> x.ufunc.subtract(1)
        ProductSpace(rn(2), 2).element([
            [0.0, -3.0],
            [-4.0, 3.0]
        ])

        There is also support for various reductions (sum, prod, min, max):

        >>> x.ufunc.sum()
        0.0

        Writing to ``out`` is also supported:

        >>> y = r22.element()
        >>> result = x.ufunc.absolute(out=y)
        >>> result
        ProductSpace(rn(2), 2).element([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        >>> result is y
        True

        See Also
        --------
        odl.util.ufuncs.NtuplesBaseUFuncs
            Base class for ufuncs in `NtuplesBase` spaces, subspaces may
            override this for greater efficiency.
        odl.util.ufuncs.ProductSpaceUFuncs
            For a list of available ufuncs.
        """
        return ProductSpaceUFuncs(self)

    def __str__(self):
        """Return ``str(self)``."""
        inner_str = ', '.join(str(part) for part in self.parts)
        return '{{{}}}'.format(inner_str)

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
            [1.0, 2.0],
            [3.0, 4.0, 5.0]
        ])

        Nestled spaces work as well

        >>> X = ProductSpace(r2x3, r2x3)
        >>> x = X.element([[[1, 2], [3, 4, 5]],[[1, 2], [3, 4, 5]]])
        >>> eval(repr(x)) == x
        True
        >>> x
        ProductSpace(ProductSpace(rn(2), rn(3)), 2).element([
            [
                [1.0, 2.0],
                [3.0, 4.0, 5.0]
            ],
            [
                [1.0, 2.0],
                [3.0, 4.0, 5.0]
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
        title : string
            Title of the figures

        indices : index expression, optional
            Indices can refer to parts of a `ProductSpaceElement` and slices
            in the parts in the following way:

            Single index (``indices=0``)
            => display that part

            Single slice (``indices=slice(None)``), or
            index list (``indices=[0, 1, 3]``)
            => display those parts

            Any tuple, for example:
            Created by `numpy.s_` ``indices=np.s_[0, :, :]`` or
            Using a raw tuple ``indices=([0, 3], slice(None))``
            => take the first elements to select the parts and
            pass the rest on to the underlying show methods.

        kwargs
            Additional arguments passed on to the ``show`` methods of
            the parts.

        Returns
        -------
        fig : list of `matplotlib.figure.Figure`
            The resulting figures. It is also shown to the user.

        See Also
        --------
        odl.discr.lp_discr.DiscreteLpElement.show :
            Display of a discretized function
        odl.space.base_ntuples.NtuplesBaseVector.show :
            Display of sequence type data
        odl.util.graphics.show_discrete_data :
            Underlying implementation
        """
        if title is None:
            title = 'ProductSpaceElement'

        if indices is None:
            if len(self) < 5:
                indices = np.arange(self.size)
            else:
                indices = np.linspace(0, self.size - 1, 4, dtype=int)
        else:
            if isinstance(indices, tuple):
                indices, kwargs['indices'] = indices[0], indices[1:]

            if isinstance(indices, slice):
                indices = range(*indices.indices(self.size))
            elif isinstance(indices, Integral):
                indices = [indices]

            # else try with indices as is

        in_figs = kwargs.pop('fig', None)
        in_figs = [None] * len(indices) if in_figs is None else in_figs

        figs = []
        for i, part, fig in zip(indices, self[indices], in_figs):
            fig = part.show(title='{}. Part {}'.format(title, i), fig=fig,
                            **kwargs)
            figs += [fig]

        return figs


# --- Add arithmetic operators that broadcast ---
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


class ProductSpaceElementWeighting(VectorWeightingBase):

    """Vector weighting for `ProductSpace`.

    For exponent 2.0, a new weighted inner product with vector ``w``
    is defined as::

        <x, y>_w = <w * x, y>

    with component-wise multiplication ``w * x``. For other exponents,
    only ``norm`` and ```dist`` are defined. In the case of exponent
    ``inf``, the weighted norm is::

        ||x||_{w,inf} = ||w * x||_inf

    otherwise it is::

        ||x||_{w,p} = ||w^(1/p) * x||_p

    Not that this definition does **not** fulfill the limit property
    in ``p``, i.e.::

        ||x||_{w,p} --/-> ||x||_{w,inf}  for p --> inf

    unless ``w = (1,...,1)``.

    The vector may only have positive entries, otherwise it does not
    define an inner product or norm, respectively. This is not checked
    during initialization.
    """

    def __init__(self, vector, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        vector : 1-dim. `array-like`
            Weighting vector of the inner product.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no inner
            product is defined.
        dist_using_inner : bool, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            Can only be used if ``exponent`` is 2.0.
        """
        super().__init__(vector, impl='numpy', exponent=exponent,
                         dist_using_inner=dist_using_inner)

    def inner(self, x1, x2):
        """Calculate the vector weighted inner product of two elements.

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

        inner = np.dot(inners, self.vector)
        if is_real_dtype(x1[0].dtype):
            return float(inner)
        else:
            return complex(inner)

    def norm(self, x):
        """Calculate the vector-weighted norm of an element.

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
                norms *= self.vector
            else:
                norms *= self.vector ** (1.0 / self.exponent)

            return float(np.linalg.norm(norms, ord=self.exponent))


class ProductSpaceConstWeighting(ConstWeightingBase):

    """Constant weighting for `ProductSpace`.

    For exponent 2.0, a new weighted inner product with constant
    ``c`` is defined as::

        <x, y>_c = c * <x, y>

    For other exponents, only ``norm`` and ```dist`` are defined.
    In the case of exponent ``inf``, the weighted norm is::

        ||x||_{c,inf} = c * ||x||_inf

    otherwise it is::

        ||x||_{c,p} = c^(1/p) * ||x||_p

    Note that this definition does **not** fulfill the limit property
    in ``p``, i.e.::

        ||x||_{c,p} --/-> ||x||_{c,inf}  for p --> inf

    unless ``c = 1``.

    The constant must be positive, otherwise it does not define an
    inner product or norm, respectively.
    """

    def __init__(self, constant, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        constant : positive float
            Weighting constant of the inner product
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no inner
            product is defined.
        dist_using_inner : bool, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            Can only be used if ``exponent`` is 2.0.
        """
        super().__init__(constant, impl='numpy', exponent=exponent,
                         dist_using_inner=dist_using_inner)

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
        if is_real_dtype(x1[0].dtype):
            return float(inner)
        else:
            return complex(inner)

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
        if self.dist_using_inner:
            norms1 = np.fromiter(
                (x1i.norm() for x1i in x1),
                dtype=np.float64, count=len(x1))
            norm1 = np.linalg.norm(norms1)

            norms2 = np.fromiter(
                (x2i.norm() for x2i in x2),
                dtype=np.float64, count=len(x2))
            norm2 = np.linalg.norm(norms2)

            inners = np.fromiter(
                (x1i.inner(x2i) for x1i, x2i in zip(x1, x2)),
                dtype=x1[0].space.dtype, count=len(x1))
            inner_re = np.sum(inners.real)

            dist_squared = norm1 ** 2 + norm2 ** 2 - 2 * inner_re
            if dist_squared < 0.0:  # Compensate for numerical error
                dist_squared = 0.0
            return np.sqrt(self.const) * float(np.sqrt(dist_squared))
        else:
            dnorms = np.fromiter(
                ((x1i - x2i).norm() for x1i, x2i in zip(x1, x2)),
                dtype=np.float64, count=len(x1))

            if self.exponent == float('inf'):
                return self.const * np.linalg.norm(dnorms, ord=self.exponent)
            else:
                return (self.const ** (1 / self.exponent) *
                        np.linalg.norm(dnorms, ord=self.exponent))


class ProductSpaceNoWeighting(NoWeightingBase, ProductSpaceConstWeighting):

    """Weighting of `ProductSpace` with constant 1."""

    # Implement singleton pattern for efficiency in the default case
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern if ``exp==2.0``."""
        if len(args) == 0:
            exponent = kwargs.pop('exponent', 2.0)
            dist_using_inner = kwargs.pop('dist_using_inner', False)
        elif len(args) == 1:
            exponent = args[0]
            args = args[1:]
            dist_using_inner = kwargs.pop('dist_using_inner', False)
        else:
            exponent = args[0]
            dist_using_inner = args[1]
            args = args[2:]

        if exponent == 2.0 and not dist_using_inner:
            if not cls._instance:
                cls._instance = super().__new__(cls, *args, **kwargs)
            return cls._instance
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        dist_using_inner : bool, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            Can only be used if ``exponent`` is 2.0.
        """
        super().__init__(impl='numpy', exponent=exponent,
                         dist_using_inner=dist_using_inner)


class ProductSpaceCustomInnerProduct(CustomInnerProductBase):

    """Class for handling a user-specified inner product on `ProductSpace`."""

    def __init__(self, inner, dist_using_inner=False):
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

        dist_using_inner : bool, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            Can only be used if ``exponent`` is 2.0.
        """
        super().__init__(impl='numpy', inner=inner,
                         dist_using_inner=dist_using_inner)


class ProductSpaceCustomNorm(CustomNormBase):

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
        super().__init__(norm, impl='numpy')


class ProductSpaceCustomDist(CustomDistBase):

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
        super().__init__(dist, impl='numpy')


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
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
