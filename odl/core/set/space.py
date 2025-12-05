# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Abstract linear vector spaces."""

from builtins import object
from enum import Enum
from dataclasses import dataclass
import numpy as np
from numbers import Number
from typing import Union

from odl.core.set.sets import Field, Set, UniversalSet


__all__ = ('LinearSpace', 'UniversalSpace')


class NumOperationParadigmSupport(Enum):
    NOT_SUPPORTED = 0
    SUPPORTED = 1
    PREFERRED = 2

    @property
    def is_supported(self):
        return self.value > 0

    @property
    def is_preferred(self):
        return self.value > 1

    def __bool__(self):
        return self.is_supported


@dataclass
class SupportedNumOperationParadigms:
    in_place: NumOperationParadigmSupport
    out_of_place: NumOperationParadigmSupport


class LinearSpace(Set):
    """Abstract linear vector space.

    Its elements are represented as instances of the
    `LinearSpaceElement` class.
    """

    def __init__(self, field):
        """Initialize a new instance.

        This method should be called by all inheriting methods so that
        the `field` property of the space is properly set.

        Parameters
        ----------
        field : `Field` or None
            Scalar field of numbers for this space.
        """
        if field is None or isinstance(field, Field):
            self.__field = field
        else:
            raise TypeError(
                f"`field` must be a `Field` instance or `None`, got {field}"
            )

    @property
    def field(self):
        """Scalar field of numbers for this vector space."""
        return self.__field

    def element(self, inp=None, **kwargs):
        """Create a `LinearSpaceElement` from ``inp`` or from scratch.

        If called without ``inp`` argument, an arbitrary element of the
        space is generated without guarantee of its state.

        If ``inp in self``, this has return ``inp`` or a view of ``inp``,
        otherwise, a copy may or may not occur.

        Parameters
        ----------
        inp : optional
            Input data from which to create the element.
        kwargs :
            Optional further arguments.

        Returns
        -------
        element : `LinearSpaceElement`
            A new element of this space.
        """
        raise NotImplementedError('abstract method')

    @property
    def examples(self):
        """Example elements `zero` and `one` (if available)."""
        # All spaces should yield the zero element
        yield ('Zero', self.zero())

        try:
            yield ('One', self.one())
        except NotImplementedError:
            pass

    @property
    def supported_num_operation_paradigms(self) -> NumOperationParadigmSupport:
        """Specify whether the low-level numerical operations in this space
        support in-place style, whether they support out-of-place style, and
        if one of them is preferred.
        Generally speaking, for fixed-dimensional spaces whose implementation
        is a monolithic array, in-place style is preferrable because it avoids
        allocating new memory.
        By contrast, in spaces that support e.g. adaptive mesh resolution, the
        in-place style may have little advantage because allocation can only
        be decided based on the inputs, and for automatic differentiation it
        may even be necessary to use purely-functional out-of-place style."""
        raise NotImplementedError('abstract method')

    def _lincomb(self, a, x1, b, x2, out):
        """Implement ``out[:] = a * x1 + b * x2``.

        This method is intended to be private. Public callers should
        resort to `lincomb` which is type-checked.
        """
        raise NotImplementedError('abstract method')

    def _dist(self, x1, x2):
        """Return the distance between ``x1`` and ``x2``.

        This method is intended to be private. Public callers should
        resort to `dist` which is type-checked.
        """
        return self.norm(x1 - x2)

    def _norm(self, x):
        """Return the norm of ``x``.

        This method is intended to be private. Public callers should
        resort to `norm` which is type-checked.
        """
        return float(np.sqrt(self.inner(x, x).real))

    def _inner(self, x1, x2):
        """Return the inner product of ``x1`` and ``x2``.

        This method is intended to be private. Public callers should
        resort to `inner` which is type-checked.
        """
        raise LinearSpaceNotImplementedError(
            f"inner product not implemented in space {self}"
        )

    def _multiply(self, x1, x2, out):
        """Implement the pointwise multiplication ``out[:] = x1 * x2``.

        This method is intended to be private. Public callers should
        resort to `multiply` which is type-checked.
        """
        raise LinearSpaceNotImplementedError(
            f"multiplication not implemented in space {self}"
        )

    def one(self):
        """Return the one (multiplicative unit) element of this space."""
        raise LinearSpaceNotImplementedError(
            f"`one` element not implemented in space {self}"
        )

    # Default methods
    def zero(self):
        """Return the zero (additive unit) element of this space."""
        tmp = self.element()
        self.lincomb(0, tmp, 0, tmp, tmp)
        return tmp

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contains : bool
            ``True`` if ``other`` is a `LinearSpaceElement` instance and
            ``other.space`` is equal to this space, ``False`` otherwise.

        Notes
        -----
        This is the strict default where spaces must be equal.
        Subclasses may choose to implement a less strict check.
        """
        return getattr(other, 'space', None) == self

    # Error checking variant of methods
    def lincomb(self, a, x1, b=None, x2=None, out=None):
        """Implement ``out[:] = a * x1 + b * x2``.

        This function implements

            ``out[:] = a * x1``

        or, if ``b`` and ``x2`` are given,

            ``out = a * x1 + b * x2``.

        Parameters
        ----------
        a : `field` element
            Scalar to multiply ``x1`` with.
        x1 : `LinearSpaceElement`
            First space element in the linear combination.
        b : `field` element, optional
            Scalar to multiply ``x2`` with. Required if ``x2`` is
            provided.
        x2 : `LinearSpaceElement`, optional
            Second space element in the linear combination.
        out : `LinearSpaceElement`, optional
            Element to which the result is written.

        Returns
        -------
        out : `LinearSpaceElement`
            Result of the linear combination. If ``out`` was provided,
            the returned object is a reference to it.

        Notes
        -----
        The elements ``out``, ``x1`` and ``x2`` may be aligned, thus a call

            ``space.lincomb(2, x, 3.14, x, out=x)``

        is (mathematically) equivalent to

            ``x = x * (2 + 3.14)``.
        """
        paradigms = self.supported_num_operation_paradigms

        def assert_x2_has_b():
            if b is None and x2 is not None:
                raise ValueError("`x2` provided but not `b`")

        def assert_x1_in_self():
            if x1 not in self:
                raise LinearSpaceTypeError(f"`x1` {x1} is not an element of {self}")

        def assert_x2_in_self():
            if x2 not in self:
                raise LinearSpaceTypeError(f"`x2` {x2} is not an element of {self}")

        def assert_a_in_field():
            if self.field is not None and a not in self.field:
                raise LinearSpaceTypeError(
                    f"`a` {a} not an element of the field {self.field} of {self}"
                )

        def assert_b_in_field():
            if self.field is not None and b not in self.field:
                raise LinearSpaceTypeError(
                    f"`b` {b} not an element of the field {self.field} of {self}"
                )

        if out is None:
            if (paradigms.in_place.is_preferred
                 or not paradigms.out_of_place.is_supported):
                out = self.element()
            else:
                assert_a_in_field()
                assert_x1_in_self()
                if b is None:  # Single element
                    assert_x2_has_b()
                    result = self._lincomb(a, x1, 0, x1, out=None)
                    assert(result is not None)
                    return result
                else:  # Two elements
                    assert_b_in_field()
                    assert_x2_in_self()
                    result = self._lincomb(a, x1, b, x2, out=None)
                    assert(result is not None)
                    return result
        elif out not in self:
            raise LinearSpaceTypeError(f"`out` {out} is not an element of {self}")

        if (not paradigms.in_place.is_supported):
            out.assign(self.lincomb(a, x1, b, x2, out=None), avoid_deep_copy=True)
            return out

        assert_a_in_field()
        assert_x1_in_self()

        if b is None:  # Single element
            assert_x2_has_b()
            self._lincomb(a, x1, 0, x1, out)

        else:  # Two elements
            assert_b_in_field()
            assert_x2_in_self()

            self._lincomb(a, x1, b, x2, out)

        return out

    def dist(self, x1, x2):
        """Return the distance between ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `LinearSpaceElement`
            Elements whose distance to compute.

        Returns
        -------
        dist : float
            Distance between ``x1`` and ``x2``.
        """
        if x1 not in self:
            raise LinearSpaceTypeError(f"`x1` {x1} is not an element of {self}")
        if x2 not in self:
            raise LinearSpaceTypeError(f"`x2` {x2} is not an element of {self}")
        return float(self._dist(x1, x2))

    def norm(self, x):
        """Return the norm of ``x``.

        Parameters
        ----------
        x : `LinearSpaceElement`
            Element whose norm to compute.

        Returns
        -------
        norm : float
            Norm of ``x``.
        """
        if x not in self:
            raise LinearSpaceTypeError(f"`x` {x} is not an element of {self}")
        return float(self._norm(x))

    def inner(self, x1, x2):
        """Return the inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `LinearSpaceElement`
            Elements whose inner product to compute.

        Returns
        -------
        inner : `LinearSpace.field` element
            Inner product of ``x1`` and ``x2``.
        """
        if x1 not in self:
            raise LinearSpaceTypeError(f"`x1` {x1} is not an element of {self}")
        if x2 not in self:
            raise LinearSpaceTypeError(f"`x2` {x2} is not an element of {self}")
        inner = self._inner(x1, x2)
        if self.field is None:
            return inner
        else:
            return self.field.element(self._inner(x1, x2))

    def _elementwise_num_operation(self, operation:str
                                       , x1: Union["LinearSpaceElement", Number]
                                       , x2: Union[None, "LinearSpaceElement", Number] = None
                                       , out=None
                                       , namespace=None
                                       , **kwargs ):
        """TODO(Justus) rewrite docstring
        Apply the numerical operation implemented by `low_level_method` to
        `x1` and `x2`.
        This is done either in in-place fashion or out-of-place, depending on
        which style is preferred for this space."""

        raise NotImplementedError("abstract method")

    def _element_reduction(self, operation:str
                               , x: "LinearSpaceElement"
                               , **kwargs
                               ):
        raise NotImplementedError("abstract method")

    @property
    def element_type(self):
        """Type of elements of this space (`LinearSpaceElement`)."""
        return LinearSpaceElement

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)
    
    def __pow__(self, shape):
        """Return ``self ** shape``.

        Notes
        -----
        This can be overridden by subclasses in order to give better memory
        coherence or otherwise a better interface.

        Examples
        --------
        Create simple power space:

        >>> r2 = odl.rn(2)
        >>> r2 ** 4
        ProductSpace(rn(2), 4)

        Multiple powers work as expected:

        >>> r2 ** (4, 2)
        ProductSpace(ProductSpace(rn(2), 4), 2)
        """
        from odl.core.space import ProductSpace

        try:
            shape = (int(shape),)
        except TypeError:
            shape = tuple(shape)

        pspace = self
        for n in shape:
            pspace = ProductSpace(pspace, n)

        return pspace

    def __mul__(self, other):
        """Return ``self * other``.

        Notes
        -----
        This can be overridden by subclasses in order to give better memory
        coherence or otherwise a better interface.

        Examples
        --------
        Create simple product space:

        >>> r2 = odl.rn(2)
        >>> r3 = odl.rn(3)
        >>> r2 * r3
        ProductSpace(rn(2), rn(3))
        """
        from odl.core.space import ProductSpace

        if not isinstance(other, LinearSpace):
            raise TypeError(f"Can only multiply with `LinearSpace`, got {other}")

        return ProductSpace(self, other)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class LinearSpaceElement(object):

    """Abstract class for `LinearSpace` elements.

    Do not use this class directly -- to create an element of a vector
    space, call the space's `LinearSpace.element` method instead.
    """

    def __init__(self, space):
        """Initialize a new instance.

        All deriving classes must call this method to set the `space`
        property.
        """
        self.__space = space

    @property
    def space(self):
        """Space to which this element belongs."""
        return self.__space

    # Convenience functions
    def assign(self, other, avoid_deep_copy: bool = False):
        """Assign the values of ``other`` to ``self``, like `self[:] = other`.

        If ``avoid_deep_copy`` is True, an attempt is made to reuse the
        storage of ``other`` for ``self``.
        This is in general unsafe (later modifications to ``self`` would
        impact also ``other``, vice versa), but faster and useful particularly
        when ``other`` is ephemeral (rvalue, in C++ terminology).

        Spaces with immutable elements (i.e., `in_place=NOT_SUPPORTED` in
        `supported_num_operation_paradigms`) may opt for performing only a
        shallow copy even if ``avoid_deep_copy`` is False.
        On the other hand, if some type conversion is necessary on ``other``
        then this will usually prompt a copy even if ``avoid_deep_copy``
        is True."""
        if other in self.space:
            self._assign(other, avoid_deep_copy=avoid_deep_copy)
        else:
            self._assign(self.space.element(other),
                         avoid_deep_copy=avoid_deep_copy)

    def _assign(self, other, avoid_deep_copy):
        """Low-level implementation of `self = other`. Assign the values of
        ``other``, which is assumed to be in the same space, to ``self``."""
        raise NotImplementedError(f"Abstract method, not implemented for {type(self)}")

    def copy(self):
        """Create an identical (deep) copy of self."""
        result = self.space.element()
        result.assign(self)
        return result

    def lincomb(self, a, x1, b=None, x2=None):
        """Implement ``self[:] = a * x1 + b * x2``.

        Parameters
        ----------
        a : element of ``space.field``
            Scalar to multiply ``x1`` with.
        x1 : `LinearSpaceElement`
            First space element in the linear combination.
        b : element of ``space.field``, optional
            Scalar to multiply ``x2`` with. Required if ``x2`` is
            provided.
        x2 : `LinearSpaceElement`, optional
            Second space element in the linear combination.

        See Also
        --------
        LinearSpace.lincomb
        """
        return self.space.lincomb(a, x1, b, x2, out=self)

    def set_zero(self):
        """Set this element to zero.

        See Also
        --------
        LinearSpace.zero
        """
        self.assign(self.space.zero())
        return self

    # Convenience methods
    # def __iadd__(self, other):
    #     """Implement ``self += other``."""
    #     if self.space.field is None:
    #         return NotImplemented
    #     elif other in self.space:
    #         return self.space.lincomb(1, self, 1, other, out=self)
    #     elif isinstance(other, LinearSpaceElement):
    #         # We do not `return NotImplemented` here since we don't want a
    #         # fallback for in-place. Otherwise python attempts
    #         # `self = self + other` which does not modify self.
    #         raise TypeError('cannot add {!r} and {!r} in-place'
    #                         ''.format(self, other))
    #     elif other in self.space.field:
    #         one = getattr(self.space, 'one', None)
    #         if one is None:
    #             raise TypeError('cannot add {!r} and {!r} in-place'
    #                             ''.format(self, other))
    #         else:
    #             # other --> other * space.one()
    #             return self.space.lincomb(1, self, other, one(), out=self)
    #     else:
    #         try:
    #             other = self.space.element(other)
    #         except (TypeError, ValueError):
    #             raise TypeError('cannot add {!r} and {!r} in-place'
    #                             ''.format(self, other))
    #         else:
    #             return self.__iadd__(other)

    def __add__(self, other):
        """Return ``self + other``."""
        return self.space._elementwise_num_operation(
            'add', self, other
        )
    
    def __sub__(self, other):
        """Return ``self - other``."""
        return self.space._elementwise_num_operation(
            'subtract', self, other
        )
    
    def __mul__(self, other):
        """Return ``self * other``."""
        return self.space._elementwise_num_operation(
            'multiply', self, other
        )
    
    def __truediv__(self, other):
        """Implement ``self / other``."""
        if isinstance(other, Number) and other == 0:
            raise ZeroDivisionError
        return self.space._elementwise_num_operation(
                'divide', self, other
            )
    
    def __floordiv__(self, other):        
        """Implement ``self // other``."""
        return self.space._elementwise_num_operation(
            'floor_divide', self, other
        )

    def __mod__(self, other):        
        """Implement ``self % other``."""
        return self.space._elementwise_num_operation(
            'remainder', self, other
        )
    
    def __pow__(self, other):
        """Implement ``self ** other``, element wise"""
        return self.space._elementwise_num_operation(
            'pow', self, other
        )

    def __radd__(self, other):
        """Return ``other + self``."""
        return self.space._elementwise_num_operation(
            'add', other, self
        )

    def __rsub__(self, other):
        """Return ``other - self``."""
        return self.space._elementwise_num_operation(
            'subtract', other, self
        )
 
    def __rmul__(self, other):
        """Return ``other * self``."""
        return self.space._elementwise_num_operation(
            'multiply', other, self
        )
    
    def __rtruediv__(self, other):
        """Implement ``other / self``."""
        return self.space._elementwise_num_operation(
             'divide', other, self
        )
    
    def __rfloordiv__(self, other):
        """Implement ``other // self``."""
        return self.space._elementwise_num_operation(
            'floor_divide', other, self
        )
    
    def __rmod__(self, other):        
        """Implement ``other % self``."""
        return self.space._elementwise_num_operation(
            'remainder', other, self
        )
    
    def __rpow__(self, other):
        """Implement ``other ** self``, element wise"""
        return self.space._elementwise_num_operation(
            'pow', other, self
        )
    
    def __iadd__(self, other):
        """Implement ``self += other``."""
        self.space._elementwise_num_operation(
            'add', self, other, self
        )
        return self
    
    def __isub__(self, other):
        """Implement ``self -= other``."""
        self.space._elementwise_num_operation(
            'subtract', self, other, self
        )
        return self
    
    def __imul__(self, other):
        """Return ``self *= other``."""
        self.space._elementwise_num_operation(
            'multiply', self, other, self
        )
        return self
    
    def __itruediv__(self, other):
        """Implement ``self /= other``."""
        if isinstance(other, Number) and other == 0:
            raise ZeroDivisionError
        self.space._elementwise_num_operation(
                'divide', self, other, self
            )
        return self
    
    def __ifloordiv__(self, other):
        """Implement ``self //= other``."""
        self.space._elementwise_num_operation(
            'floor_divide', self, other, self
        )
        return self
    
    def __imod__(self, other):
        """Implement ``self %= other``."""
        self.space._elementwise_num_operation(
            'remainder', self, other, self
        )
        return self
    
    def __ipow__(self, other):
        """Implement ``self *= other``, element wise"""
        self.space._elementwise_num_operation(
            'pow', self, other, self
        )
        return self
    
    def __neg__(self):
        """Return ``-self``."""
        if self.space.field is None:
            return NotImplemented
        return (-1) * self

    def __pos__(self):
        """Return ``+self``."""
        return self.copy()

    def __lt__(self, other):
        """Implement ``self < other``."""
        return self.space._elementwise_num_operation('less', self, other)
    
    def __le__(self, other):
        """Implement ``self <= other``."""
        return self.space._elementwise_num_operation('less_equal', self, other)
    
    def __gt__(self, other):
        """Implement ``self > other``."""
        return self.space._elementwise_num_operation('greater', self, other)
    
    def __ge__(self, other):
        """Implement ``self >= other``."""
        return self.space._elementwise_num_operation('greater_equal', self, other)
    
    def __eq__(self, other):
        """Return ``self == other``.

        Two elements are equal if their distance is zero.

        Parameters
        ----------
        other : `LinearSpaceElement`
            Element of this space.

        Returns
        -------
        equals : bool
            ``True`` if the elements are equal ``False`` otherwise.

        See Also
        --------
        LinearSpace.dist

        Notes
        -----
        Equality is very sensitive to numerical errors, thus any
        arithmetic operations should be expected to break equality.

        Examples
        --------
        >>> rn = odl.rn(1, norm=np.linalg.norm)
        >>> x = rn.element([0.1])
        >>> x == x
        True
        >>> y = rn.element([0.1])
        >>> x == y
        True
        >>> z = rn.element([0.3])
        >>> x + x + x == z
        False
        """
        if other is self:
            # Optimization for a common case
            return True
        elif (not isinstance(other, LinearSpaceElement) or
              other.space != self.space):
            # Cannot use (if other not in self.space) since this is not
            # reflexive.
            return False
        else:
            return self.space.dist(self, other) == 0

    def __ne__(self, other):
        """Return ``self != other``."""
        return not self.__eq__(other)

    # Disable hash since vectors are mutable
    __hash__ = None

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

    def __copy__(self):
        """Return a copy of this element.

        See Also
        --------
        LinearSpace.copy
        """
        return self.copy()

    def __deepcopy__(self, memo):
        """Return a deep copy of this element.

        See Also
        --------
        LinearSpace.copy
        """
        return self.copy()

    def norm(self):
        """Return the norm of this element.

        See Also
        --------
        LinearSpace.norm
        """
        return self.space.norm(self)

    def dist(self, other):
        """Return the distance of ``self`` to ``other``.

        See Also
        --------
        LinearSpace.dist
        """
        return self.space.dist(self, other)

    def inner(self, other):
        """Return the inner product of ``self`` and ``other``.

        See Also
        --------
        LinearSpace.inner
        """
        return self.space.inner(self, other)

    def multiply(self, other, out=None):
        """Return ``out = self * other``.

        If ``out`` is provided, the result is written to it.

        See Also
        --------
        LinearSpace.multiply
        """
        return self.space.multiply(self, other, out=out)

    def divide(self, other, out=None):
        """Return ``out = self / other``.

        If ``out`` is provided, the result is written to it.

        See Also
        --------
        LinearSpace.divide
        """
        return self.space.divide(self, other, out=out)

    @property
    def T(self):
        """This element's transpose, i.e. the functional ``<. , self>``.

        Returns
        -------
        transpose : `InnerProductOperator`

        Notes
        -----
        This function is only defined in inner product spaces.

        In a complex space, the conjugate transpose of is taken instead
        of the transpose only.

        Examples
        --------
        >>> rn = odl.rn(3)
        >>> x = rn.element([1, 2, 3])
        >>> y = rn.element([2, 1, 3])
        >>> x.T(y)
        13.0
        """
        from odl.core.operator import InnerProductOperator
        return InnerProductOperator(self.copy())

    def __array__(self):
        raise RuntimeError("""
           You are trying to convert an ODL object to a plain array, possibly via a NumPy operation. This is not supported in ODL-1.0 anymore because it interferes with the more general Array API and easily leads to confusing results.

           Instead, you should either:

           - Use the ODL operation (e.g. `odl.sin(x)`)
           - Unwrap the raw array contained in the ODL object, as `x.data`
           - Explicitly convert to NumPy (or another raw array type) via DLPack
           """)

    # Give an `Element` a higher priority than any NumPy array type. This
    # forces the usage of `__op__` of `Element` if the other operand
    # is a NumPy object (applies also to scalars!).
    __array_priority__ = 1000000.0


class UniversalSpace(LinearSpace):

    """A dummy linear space class.

    Mostly raising `LinearSpaceNotImplementedError`.
    """

    def __init__(self):
        """Initialize a new instance."""
        super(UniversalSpace, self).__init__(field=UniversalSet())

    def element(self, inp=None):
        """Dummy element creation method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _lincomb(self, a, x1, b, x2, out):
        """Dummy linear combination.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _dist(self, x1, x2):
        """Dummy distance method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _norm(self, x):
        """Dummy norm method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _inner(self, x1, x2):
        """Dummy inner product method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _multiply(self, x1, x2, out):
        """Dummy multiplication method.

        raises `LinearSpaceNotImplementedError`."""
        raise LinearSpaceNotImplementedError

    def _divide(self, x1, x2, out):
        """Dummy division method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def __eq__(self, other):
        """Return ``self == other``.

        Dummy check, ``True`` for any `LinearSpace`.
        """
        return isinstance(other, LinearSpace)

    def __contains__(self, other):
        """Return ``other in self``.

        Dummy membership check, ``True`` for any `LinearSpaceElement`.
        """
        return isinstance(other, LinearSpaceElement)


class LinearSpaceTypeError(TypeError):
    """Exception for type errors in `LinearSpace`'s.

    This exception is raised when the wrong type of element is fed to
    `LinearSpace.lincomb` and related functions.
    """


class LinearSpaceNotImplementedError(NotImplementedError):
    """Exception for unimplemented functionality in `LinearSpace`'s.

    This exception is raised when a method is called in `LinearSpace`
    that has not been defined in a specific space.
    """


if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests
    run_doctests()
