# Copyright 2014, 2015 The ODL development group
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

"""Abstract vector spaces.

The classes in this module represent abstract mathematical concepts
of vector spaces. They cannot be used directly but are rather intended
to be subclassed by concrete space implementations. The spaces
provide default implementations of the most important vector space
operations. See the documentation of the respective classes for more
details.

Class descriptions
------------------

+--------------+-------------+----------------------------------------+
|Class name    |Direct       |Description                             |
|              |ancestors    |                                        |
+==============+=============+========================================+
|`LinearSpace` |`Set`        |**Abstract class.** A vector space over |
|              |             |a field (real or complex numbers)       |
|              |             |defining a vector-vector addition and a |
|              |             |scalar-vector multiplication with       |
|              |             |certain properties. See the article     |
|              |             |`Vector space`_ on Wikipedia for further|
|              |             |information.                            |
+--------------+-------------+----------------------------------------+
|`MetricSpace` |`LinearSpace`|**Abstract class.** A vector space with |
|              |             |a metric, i.e. a `dist` function        |
|              |             |measuring the distance between two      |
|              |             |vectors.                                |
+--------------+-------------+----------------------------------------+
|`NormedSpace` |`MetricSpace`|**Abstract class.** A metric space with |
|              |             |a `norm` function measuring the length  |
|              |             |a vector. The `dist` function is induced|
|              |             |by the norm as                          |
|              |             |`dist(x, y) = norm(x - y)`.             |
+--------------+-------------+----------------------------------------+
|`HilbertSpace`|`NormedSpace`|**Abstract class.** A normed space with |
|              |             |an inner product measuring angles       |
|              |             |between vectors with unit length. The   |
|              |             |`norm` function is induced by the inner |
|              |             |product by the according to             |
|              |             |`norm(x) = inner(x, x)`.                |
+--------------+-------------+----------------------------------------+
|`Algebra`     |`LinearSpace`|**Abstract class.** A linear space with |
|              |             |a vector-vector multiplication under    |
|              |             |which the space is closed. See the      |
|              |             |Algebra_ article on Wikipedia for       |
|              |             |further information. (Note that we      |
|              |             |assume commutativity and unitality.)    |
+--------------+-------------+----------------------------------------+

See also
--------
The `Set` class is defined in `odl.space.set`.

.. _Vector space: https://en.wikipedia.org/wiki/Vector_space#Definition
.. _Algebra: https://en.wikipedia.org/wiki/Associative_algebra
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import object, str, super
from future import standard_library
from future.utils import with_metaclass

# External module imports
from abc import ABCMeta, abstractmethod, abstractproperty
from math import sqrt

# ODL imports
from odl.space.set import Set
from odl.utility.utility import errfmt

standard_library.install_aliases()

__all__ = ('LinearSpace', 'MetricSpace', 'NormedSpace', 'HilbertSpace',
           'Algebra')


class LinearSpace(Set):

    """Abstract linear vector space.

    Its elements are represented as instances of the inner
    `LinearSpace.Vector` class.

    The concept of linear vector spaces in ODL is largely inspired by
    the `Rice Vector Library`_ (RVL).

    The abstract `LinearSpace` class is intended for quick prototyping.
    It has a number of abstract methods which must be overridden by a
    subclass. On the other hand, it provides automatic error checking
    and numerous attributes and methods for convenience.

    In the following, the abstract methods are explained in detail.

    Abstract methods
    ----------------

    `element(inp=None)`
    ~~~~~~~~~~~~~~~~~~~~
    This public method is the factory for the inner
    `LinearSpace.Vector` class. It creates a new element of the space,
    either from scratch or from an existing data container. In the
    simplest possible case, it just delegates the construction to the
    `Vector` class.

    If no data is provided, the new element is **merely allocated, not
    initialized**, thus it can contain *any* value.

    **Parameters:**
        `inp` : `object`, optional
            A container for values for the element initialization

    **Returns:**
        `element` : `LinearSpace.Vector`
            The new vector.

    `_lincomb(z, a, x, b, y)`
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    This private method is the raw implementation (i.e. no error
    checking) of the linear combination `z <-- a * x + b * y`.
    `_lincomb` and its public counterpart `lincomb` are used to cover
    a range of convenience functions, see below.

    **Parameters:**
        `z` : `LinearSpace.Vector`
            Element to which the result of the computation is written
        `a` : `LinearSpace.field` element
            Multiplicative scalar factor for input vector `x`
        `x` : `LinearSpace.Vector`
            First input vector
        `b` : `LinearSpace.field` element
            Multiplicative scalar factor for input vector `y`
        `y` : `LinearSpace.Vector`
            Second input vector

    **Returns:** `None`

    **Requirements:**
     * Aliasing of `x`, `y` and `z` **must** be allowed.
     * The input vectors `x` and `y` **must not** be modified.
     * The initial state of the output vector `z` **must not**
       influence the result.

    `field`
    ~~~~~~~
    The public attribute determining the type of scalars which
    underlie the space. Can be either `RealNumbers` or
    `ComplexNumbers` (see `odl.space.set`).

    Must be implemented as a `@property` to make it immutable.

    `equals(other)`
    ~~~~~~~~~~~~~~~
    `LinearSpace` inherits this abstract method from `Set`. Its
    purpose is to check two `LinearSpace` instances for equality.

    **Parameters:**
        `other` : `object`
            The object to compare to.

    **Returns:**
        `equals` : `boolean`
            `True` if `other` is the same `LinearSpace`, `False`
            otherwise.

    Default convenience methods
    ---------------------------
    `LinearSpace` provides several default methods for convenience
    which use the abstract methods above. A subclass may override
    them with own implementations.

    +--------------+--------------------+-----------------------------+
    |Signature     |Return type         |Description                  |
    +==============+====================+=============================+
    |`lincomb(z, a,|`None`              |Linear combination           |
    |x, b, y)`     |                    |`z <-- a * x + b * y`. Like  |
    |              |                    |`_lincomb()`, but with type  |
    |              |                    |checks.                      |
    +--------------+--------------------+-----------------------------+
    |`zero()`      |`LinearSpace.Vector`|Create a zero vector by first|
    |              |                    |issuing `x = element()` and  |
    |              |                    |then                         |
    |              |                    |`_lincomb(x, 0, x, 0, x)`    |
    +--------------+--------------------+-----------------------------+

    Magic methods
    -------------

    +----------------------+----------------+--------------------+
    |Signature             |Provides syntax |Implementation      |
    +======================+================+====================+
    |`__eq__(other)`       |`self == other` |`equals(other)`     |
    +----------------------+----------------+--------------------+
    |`__ne__(other)`       |`self != other` |`not equals(other)` |
    +----------------------+----------------+--------------------+
    |`__contains__(other)` |`other in self` |`contains(other)`   |
    +----------------------+----------------+--------------------+

    See also
    --------
    See Wikipedia's `Vector space`_ article for a mathematical
    overview.

    .. _`Rice Vector Library`:
       http://www.trip.caam.rice.edu/software/rvl/rvl/doc/html/
    .. _`Vector space`:
       https://en.wikipedia.org/wiki/Vector_space
    """

    @abstractmethod
    def element(self, inp=None):
        """Create an element from `inp` or from scratch.

        If called without `inp` argument, an arbitrary element in the
        space is generated without guarantee of its state.

        Parameters
        ----------
        inp : `object`, optional
            The input data from which to create the element

        Returns
        -------
        element : `LinearSpace.Vector`
            A vector in this space
        """

    @abstractmethod
    def _lincomb(self, z, a, x, b, y):
        """Calculate z = a*x + b*y.

        This method is intended to be private, public callers should
        resort to lincomb which is type-checked.
        """

    @abstractproperty
    def field(self):
        """The field of the vector space."""

    # Default methods
    def zero(self):
        """A zero vector in this space.

        The zero vector is defined as the additive unit of a space.

        Parameters
        ----------
        None

        Returns
        -------
        v : Vector
            The zero vector of this space
        """
        # Default implementation using lincomb
        tmp = self.element()
        self._lincomb(tmp, 0, tmp, 0, tmp)
        return tmp

    def contains(self, other):
        """Test an object for membership in space.

        Parameters
        ----------
        other : `object`
            The object to test for membership

        Returns
        -------
        contains : `bool`
            True if `other` is a `LinearSpace.Vector` instance and
            `other.space` is equal to this space.

        Notes
        -----
        This is the strict default where spaces must be equal.
        Subclasses may choose to implement a less strict check.
        """
        return isinstance(other, LinearSpace.Vector) and other.space == self

    # Error checking variant of methods
    def lincomb(self, z, a, x, b=None, y=None):
        """Linear combination of vectors.

        Calculates

        z = a*x
        or if b and y are given
        z = a*x + b*y

        with error checking of types.

        Parameters
        ----------
        z : Vector
            The Vector that the result should be written to.
        a : Scalar in the field of this space
            Scalar to multiply `x` with.
        x : Vector
            The first of the summands
        b : Scalar, optional
            Scalar to multiply `y` with.
        y : Vector, optional
            The second of the summands

        Returns
        -------
        None

        Notes
        -----
        Some notes and examples

        Alignment
        ~~~~~~~~~
        The vectors `z`, `x` and `y` may be aligned, thus a call

        space.lincomb(x, 2, x, 3.14, x)

        is (mathematically) equivalent to

        x = x * (1 + 2 + 3.14)
        """
        if z not in self:
            raise TypeError(errfmt('''
            `z` {!r} not in space {!r}.'''.format(z, self)))

        if a not in self.field:
            raise TypeError(errfmt('''
            `a` {!r} not in `field` {!r} of space {!r}.
            '''.format(a, self.field, self)))

        if x not in self:
            raise TypeError(errfmt('''
            `x` {!r} not in space {!r}.'''.format(x, self)))

        if b is None:  # Single argument
            if y is not None:
                raise ValueError('`y` provided but not `b`.')

            # Call method
            return self._lincomb(z, a, x, 0, x)
        else:  # Two arguments
            if b not in self.field:
                raise TypeError(errfmt('''
                `b` {!r} not in `field` {!r} of space {!r}.
                '''.format(b, self.field, self)))

            if y not in self:
                raise TypeError(errfmt('''
                `y` {!r} not in space {!r}.'''.format(y, self)))

            # Call method
            return self._lincomb(z, a, x, b, y)

    class Vector(with_metaclass(ABCMeta, object)):

        """Abstract `LinearSpace` element.

        Not intended for creation of vectors, use the space's
        `element()` method instead.

        Attributes
        ----------

        +-------+-------------+---------------------------------------+
        |Name   |Type         |Description                            |
        +=======+=============+=======================================+
        |`space`|`LinearSpace`|The space to which this vector belongs |
        +-------+-------------+---------------------------------------+

        Methods
        -------

        +----------------+--------------------+-----------------------+
        |Signature       |Return type         |Description            |
        +================+====================+=======================+
        |`assign(other)` |`None`              |Copy the values of     |
        |                |                    |`other` to this vector.|
        +----------------+--------------------+-----------------------+
        |`copy()`        |`LinearSpace.Vector`|Create a (deep) copy of|
        |                |                    |this vector.           |
        +----------------+--------------------+-----------------------+
        |`lincomb(a, x,  |`None`              |Linear combination     |
        |b=None, y=None)`|                    |`a * x + b * y`, stored|
        |                |                    |in this vector.        |
        +----------------+--------------------+-----------------------+
        |`set_zero()`    |`None`              |Multiply this vector   |
        |                |                    |by zero.               |
        +----------------+--------------------+-----------------------+

        Magic methods
        -------------

        +------------------+----------------+-------------------------+
        |Signature         |Provides syntax |Implementation           |
        +==================+================+=========================+
        |`__iadd__(other)` |`self += other` |`lincomb(self, 1, self,  |
        |                  |                |1, other)`               |
        +------------------+----------------+-------------------------+
        |`__isub__(other)` |`self -= other` |`lincomb(self, 1, self,  |
        |                  |                |-1, other)`              |
        +------------------+----------------+-------------------------+
        |`__imul__(scalar)`|`self *= scalar`|`lincomb(self, scalar,   |
        |                  |                |self)`                   |
        +------------------+----------------+-------------------------+
        |`__itruediv__     |`self /= scalar`|`__imul__(1.0 / scalar)` |
        |(scalar)`         |                |                         |
        +------------------+----------------+-------------------------+
        |`__idiv__(scalar)`|`self /= scalar`|same as `__itruediv__`   |
        +------------------+----------------+-------------------------+
        |`__add__(other)`  |`self + other`  |`x = element()`;         |
        |                  |                |`lincomb(x, 1, self, 1,  |
        |                  |                |other)`                  |
        +------------------+----------------+-------------------------+
        |`__sub__(other)`  |`self - other`  |`x = element()`;         |
        |                  |                |`lincomb(x, 1, self, -1, |
        |                  |                |other)`                  |
        +------------------+----------------+-------------------------+
        |`__mul__(scalar)` |`self * scalar` |`x = element()`;         |
        |                  |                |`lincomb(x, scalar,      |
        |                  |                |self)`                   |
        +------------------+----------------+-------------------------+
        |`__rmul__(scalar)`|`scalar * self` |`__mul__(scalar)`        |
        +------------------+----------------+-------------------------+
        |`__truediv__      |`self /= scalar`|`__mul__(1.0 / scalar)`  |
        |(scalar)`         |                |                         |
        +------------------+----------------+-------------------------+
        |`__div__(scalar)` |`self /= scalar`|same as `__truediv__`    |
        +------------------+----------------+-------------------------+
        |`__pos__()`       |`+self`         |`copy()`                 |
        +------------------+----------------+-------------------------+
        |`__neg__()`       |`-self`         |`x = element()`;         |
        |                  |                |`lincomb(x, -1, self)`   |
        +------------------+----------------+-------------------------+

        Note that `lincomb` and `element` refer to `LinearSpace`
        methods.
        """

        def __init__(self, space):
            """Default initializer of vectors.

            All deriving classes must call this method to set space.
            """
            if not isinstance(space, LinearSpace):
                raise TypeError(errfmt('''
                'space' ({}) is not a LinearSpace instance'''.format(space)))
            self._space = space

        @property
        def space(self):
            """The space this vector belongs to."""
            return self._space

        # Convenience functions
        def assign(self, other):
            """Assign the values of other to this vector."""
            self.space.lincomb(self, 1, other)

        def copy(self):
            """Create an identical (deep) copy of this vector."""
            result = self.space.element()
            result.assign(self)
            return result

        def lincomb(self, a, x, b=None, y=None):
            """Assign a linear combination to this vector.

            Implemented as `space.lincomb(self, a, x, b, y)`.
            """
            self.space.lincomb(self, a, x, b, y)

        def set_zero(self):
            """Set this vector to the zero vector."""
            self.space.lincomb(self, 0, self, 0, self)

        # Convenience operators
        def __iadd__(self, other):
            """Implementation of 'self += other'."""
            self.space.lincomb(self, 1, self, 1, other)
            return self

        def __isub__(self, other):
            """Implementation of 'self -= other'."""
            self.space.lincomb(self, 1, self, -1, other)
            return self

        def __imul__(self, other):
            """Implementation of 'self *= other'."""
            self.space.lincomb(self, other, self)
            return self

        def __itruediv__(self, other):
            """Implementation of 'self /= other' (true division)."""
            return self.__imul__(1.0 / other)

        def __idiv__(self, other):
            """Implementation of 'self /= other'."""
            return self.__itruediv__(other)

        def __add__(self, other):
            """Implementation of 'self + other'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, 1, self, 1, other)
            return tmp

        def __sub__(self, other):
            """Implementation of 'self - other'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, 1, self, -1, other)
            return tmp

        def __mul__(self, other):
            """Implementation of 'self * other'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, other, self)
            return tmp

        def __rmul__(self, other):
            """Implementation of 'other * self'."""
            return self.__mul__(other)

        def __truediv__(self, other):
            """Implementation of 'self / other' (true division)."""
            return self.__mul__(1.0 / other)

        def __div__(self, other):
            """Implementation of 'self / scalar'."""
            return self.__truediv__(other)

        def __neg__(self):
            """Implementation of '-self'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, -1.0, self)
            return tmp

        def __pos__(self):
            """Implementation of '+self'."""
            return self.copy()

        def __str__(self):
            """Implementation of str()."""
            return str(self.space) + ".Vector"


class MetricSpace(LinearSpace):

    """Abstract metric space.

    Its elements are represented as instances of the inner
    `MetricSpace.Vector` class.

    In addition to `LinearSpace`, `MetricSpace` has the following
    abstract method:

    `_dist(x, y)`
    ~~~~~~~~~~~~~
    A raw (not type-checking) private method measuring the distance
    between two vectors `x` and `y`.

    **Parameters:**
        `x` : `object`
            The first vector
        `y` : `object`
            The second vector

    **Returns:**
        `distance` : `RealNumber`
            The distance between `x` and `y`, measured in the space's
            metric

    **Requirements:**
     * `_dist(x, y) == _dist(y, x)`
     * `_dist(x, y) <= _dist(x, z) + _dist(z, y)`
     * `_dist(x, y) >= 0`
     * `_dist(x, y) == 0` (approx.) if and only if `x == y` (approx.)

    Differences to `LinearSpace`
    ----------------------------

    +------------+------------+----------------------------------------+
    |Signature   |Return type |Description                             |
    +============+============+========================================+
    |`dist(x, y)`|`float`     |Distance between two space elements.    |
    |            |            |Like `_dist()`, but with type checks.   |
    +------------+------------+----------------------------------------+

    See also
    --------
    See `LinearSpace` for a list of additional attributes and methods
    as well as further help.

    See Wikipedia's `Metric space`_ article for a mathematical
    overview.

    .. _`Metric space`:
       https://en.wikipedia.org/wiki/Metric_space
    """

    @abstractmethod
    def _dist(self, x, y):
        """Calculate the distance between two vectors."""

    # Default implemented methods
    def dist(self, x, y):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x : MetricSpace.Vector
            The first element

        y : MetricSpace.Vector
            The second element

        Returns
        -------
        dist : RealNumber
               Distance between vectors
        """
        if x not in self:
            raise TypeError('`x` {} not in space {}'.format(x, self))

        if y not in self:
            raise TypeError('`y` {} not in space {}'.format(y, self))

        return float(self._dist(x, y))

    class Vector(LinearSpace.Vector):

        """Abstract `MetricSpace` element.

        Not intended for creation of vectors, use the space's
        `element()` method instead.

        Differences to `LinearSpace.Vector`
        -----------------------------------

        Methods
        -------

        +---------------+-----------+---------------------------------+
        |Signature      |Return type|Description                      |
        +===============+===========+=================================+
        |`dist(other)`  |`float`    |Distance of this vector to       |
        |               |           |`other`. Implemented as          |
        |               |           |`space.dist(self, other)`.       |
        +---------------+-----------+---------------------------------+
        |`equals(other)`|`boolean`  |Test if `other` is equal to this |
        |               |           |vector. Implemented as           |
        |               |           |`dist(other) == 0`.              |
        +---------------+-----------+---------------------------------+

        Magic methods
        -------------

        +------------------+----------------+-------------------------+
        |Signature         |Provides syntax |Implementation           |
        +==================+================+=========================+
        |`__eq__(ohter)`   |`self == other` |`equals(other)`          |
        +------------------+----------------+-------------------------+
        |`__ne__(other)`   |`self != other` |`not equals(other)`      |
        +------------------+----------------+-------------------------+

        See also
        --------
        See `LinearSpace.Vector` for a list of additional attributes
        and methods as well as further help.
        """

        def __init__(self, space):
            if not isinstance(space, MetricSpace):
                raise TypeError(errfmt('''
                `space` {} not a MetricSpace.'''.format(space)))
            super().__init__(space)

        def equals(self, other):
            """Test two vectors for equality.

            Two vectors are equal if their distance is 0

            Parameters
            ----------
            other : MetricSpace.Vector
                    Vector in this space.

            Returns
            -------
            equals : boolean
                     True if the vectors are equal, else false.

            Note
            ----
            Equality is very sensitive to numerical errors, thus any
            operations on a vector should be expected to break equality
            testing.

            Example
            -------

            >>> from odl.space.cartesian import NormedRn
            >>> import numpy as np
            >>> X = NormedRn(1, norm=np.linalg.norm)
            >>> x = X.element([0.1])
            >>> x == x
            True
            >>> y = X.element([0.1])
            >>> x == y
            True
            >>> z = X.element([0.3])
            >>> x+x+x == z
            False
            """
            if (not isinstance(other, LinearSpace.Vector) or
                    other.space != self.space):
                # Cannot use (if other not in self.space) since this is not
                # reflexive.
                return False
            elif other is self:
                # Optimization for the most common case
                return True
            else:
                return self.dist(other) == 0

        def __eq__(self, other):
            return self.equals(other)

        def __ne__(self, other):
            return not self.equals(other)


class NormedSpace(MetricSpace):

    """Abstract normed space.

    Its elements are represented as instances of the inner
    `NormedSpace.Vector` class.

    In addition to `LinearSpace`, `NormedSpace` has the following
    abstract method:

    `_norm(x)`
    ~~~~~~~~~~
    A raw (not type-checking) private method measuring the length of a
    space element `x`.

    **Parameters:**
        `x` : `object`
            The vector to measure

    **Returns:**
        `norm` : `RealNumber`
            The length of `x` as measured in the space's metric

    **Requirements:**
     * `_norm(s * x) = |s| * _norm(x)` for any scalar `s`
     * `_norm(x + y) <= _norm(x) + _norm(y)`
     * `_norm(x) >= 0`
     * `_norm(x) == 0` (approx.) if and only if `x == 0` (approx.)

    Note
    ----
    A `NormedSpace` is a `MetricSpace` with the distance function
    `_dist(x, y) = _norm(x - y)`.

    Differences to `LinearSpace`
    ----------------------------

    +-------------+------------+--------------------------------------+
    |Signature    |Return type |Description                           |
    +=============+============+======================================+
    |`norm(x)`    |`float`     |Length of a space element. Like       |
    |             |            |`_norm()`, but with type checks.      |
    +-------------+------------+--------------------------------------+
    |`dist(x, y)` |`float`     |Distance between two space elements.  |
    |             |            |Implemented as `_norm(x - y)` with    |
    |             |            |type checks.                          |
    +-------------+------------+--------------------------------------+

    See also
    --------
    See `LinearSpace` for a list of additional attributes and methods
    as well as further help.

    See Wikipedia's `Normed vector space`_ article for a mathematical
    overview.

    .. _`Normed vector space`:
       https://en.wikipedia.org/wiki/Normed_vector_space
    """

    @abstractmethod
    def _norm(self, vector):
        """Implementation of norm."""

    # Default implemented methods
    def norm(self, vector):
        """Calculate the norm of a vector."""
        if not self.contains(vector):
            raise TypeError('x ({}) is not in space ({})'.format(vector, self))

        return float(self._norm(vector))

    # Default implmentation
    def _dist(self, x, y):
        """ The distance in Normed spaces is implicitly defined by the norm
        """
        return self._norm(x - y)

    class Vector(MetricSpace.Vector):

        """Abstract `NormedSpace` element.

        Not intended for creation of vectors, use the space's
        `element()` method instead.

        Differences to `LinearSpace.Vector`
        -----------------------------------

        Methods
        -------

        +---------------+-----------+---------------------------------+
        |Signature      |Return type|Description                      |
        +===============+===========+=================================+
        |`norm()`       |`float`    |Length of this vector.           |
        |               |           |Implemented as                   |
        |               |           |`space.norm(self)`.              |
        +---------------+-----------+---------------------------------+
        |`dist(other)`  |`float`    |Distance of this vector to       |
        |               |           |`other`. Implemented as          |
        |               |           |`space.norm(self - other)`.      |
        +---------------+-----------+---------------------------------+
        |`equals(other)`|`boolean`  |Test if `other` is equal to this |
        |               |           |vector. Implemented as           |
        |               |           |`dist(other) == 0`.              |
        +---------------+-----------+---------------------------------+

        See also
        --------
        See `LinearSpace.Vector` for a list of additional attributes
        and methods as well as further help.
        """

        def __init__(self, space):
            if not isinstance(space, NormedSpace):
                raise TypeError(errfmt('''
                'space' ({}) is not a NormedSpace instance'''.format(space)))
            super().__init__(space)


class HilbertSpace(NormedSpace):

    """Abstract (pre)-Hilbert space or inner product space.

    Its elements are represented as instances of the inner
    `HilbertSpace.Vector` class.

    In addition to `LinearSpace`, `HilbertSpace` has the following
    abstract method:

    `_inner(x, y)`
    ~~~~~~~~~~~~~~
    A raw (not type-checking) private method calculating the inner
    product of two space elements `x` and `y`.

    **Parameters:**
        `x` : `object`
            The first vector
        `y` : `object`
            The second vector

    **Returns:**
        `inner` : `space.field` element
            The inner product of `x` and `y`

    **Requirements:**
     * `_inner(x, y) == _inner(y, x)^*` with '*' = complex conjugation
     * `_inner(s * x, y) == s * _inner(x, y)` for `s` scalar
     * `_inner(x + z, y) == _inner(x, y) + _inner(z, y)`
     * `_inner(x, x) == 0` (approx.) if and only if `x == 0` (approx.)

    Note
    ----
    A `HilbertSpace` is a `NormedSpace` with the norm function
    `_norm(x) = sqrt(_inner(x, x))`, and in consequence also a
    `MetricSpace` with the distance function
    `_dist(x, y) = _norm(x - y)`.

    Differences to `LinearSpace`
    ----------------------------

    +-------------+------------+--------------------------------------+
    |Signature    |Return type |Description                           |
    +=============+============+======================================+
    |`inner(x, y)`|`field`     |Inner product of two space elements.  |
    |             |element     |Like `_inner()`, but with type        |
    |             |            |checks.                               |
    +-------------+------------+--------------------------------------+
    |`norm(x)`    |`float`     |Length of a space element. Implemented|
    |             |            |as `sqrt(_inner(x, x))` with type     |
    |             |            |checks.                               |
    +-------------+------------+--------------------------------------+
    |`dist(x, y)` |`float`     |Distance between two space elements.  |
    |             |            |Implemented as `_norm(x - y)` with    |
    |             |            |type checks.                          |
    +-------------+------------+--------------------------------------+

    See also
    --------
    See `LinearSpace` for a list of additional attributes and methods
    as well as further help.

    See Wikipedia's `Inner product space`_ article for a mathematical
    overview.

    .. _`Inner product space`:
       https://en.wikipedia.org/wiki/Inner_product_space
    """

    @abstractmethod
    def _inner(self, x, y):
        """ Implementation of inner
        """

    # Default implemented methods
    def inner(self, x, y):
        """ Calculates the inner product of the vectors x and y
        """

        # Check spaces
        if not self.contains(x):
            raise TypeError('x ({}) is not in space ({})'.format(x, self))

        if not self.contains(y):
            raise TypeError('y ({}) is not in space ({})'.format(y, self))

        return self.field.element(self._inner(x, y))

    # Default implmentation
    def _norm(self, x):
        """ The norm in Hilbert spaces is implicitly defined by the inner
        product
        """

        return sqrt(self._inner(x, x))

    class Vector(NormedSpace.Vector):

        """Abstract `HilbertSpace` element.

        Not intended for creation of vectors, use the space's
        `element()` method instead.

        Differences to `LinearSpace.Vector`
        -----------------------------------

        Methods
        -------

        +---------------+-------------+-------------------------------+
        |Signature      |Return type  |Description                    |
        +===============+=============+===============================+
        |`inner(other)` |`space.field`|Inner product of this vector   |
        |               |element      |with `other`. Implemented as   |
        |               |             |`space.inner(self, other)`.    |
        +---------------+-------------+-------------------------------+
        |`norm()`       |`float`      |Length of this vector.         |
        |               |             |Implemented as                 |
        |               |             |`inner(self, self)`.           |
        +---------------+-------------+-------------------------------+
        |`dist(other)`  |`float`      |Distance of this vector to     |
        |               |             |`other`. Implemented as        |
        |               |             |`space.norm(self - other)`.    |
        +---------------+-------------+-------------------------------+
        |`equals(other)`|`boolean`    |Test if `other` is equal to    |
        |               |             |this vector. Implemented as    |
        |               |             |`dist(other) == 0`.            |
        +---------------+-------------+-------------------------------+

        See also
        --------
        See `LinearSpace.Vector` for a list of additional attributes
        and methods as well as further help.
        """

        def __init__(self, space):
            if not isinstance(space, HilbertSpace):
                raise TypeError(errfmt('''
                'space' ({}) is not a HilbertSpace instance'''.format(space)))
            super().__init__(space)


class Algebra(LinearSpace):

    """Abstract (Banach) algebra with multiplication.

    Its elements are represented as instances of the inner
    `Algebra.Vector` class.

    In addition to `LinearSpace`, `Algebra` has the following
    abstract method:

    `_multiply(x, y)`
    ~~~~~~~~~~~~~~~~~
    A raw (not type-checking) private method multiplying two vectors
    `x` and `y`.

    **Parameters:**
        `x` : `object`
            First vector
        `y` : `object`
            Second vector, stores the final result

    **Returns:** `None`

    **Requirements:**
     * `y` after `_multiply(x, y)` equals `x` after `_multiply(y, x)`
     * `_multiply(s * x, y) <==> y *= s; _multiply(x, y)  <==>
        _multiply(x, y); y *= s` for `s` scalar
     * There is a space element `one` with
       `x` after `_multiply(one, x)` equals `x` equals `one`
       after `_multiply(x, one)`.

    Note
    ----
    The above conditions on the multiplication make `Algebra` a
    *unital commutative algebra* in the mathematical sense.

    Differences to `LinearSpace`
    ----------------------------

    +----------------+-----------+------------------------------------+
    |Signature       |Return type|Description                         |
    +================+===========+====================================+
    |`multiply(x, y)`|`None`     |Multiplication of two space         |
    |                |           |elements. Like `_multiply()`, but   |
    |                |           |with type checks.                   |
    +----------------+-----------+------------------------------------+

    See also
    --------
    See `LinearSpace` for a list of additional attributes and methods
    as well as further help.

    See Wikipedia's `Associative algebra`_ article for a mathematical
    overview.

    .. _`Associative algebra`:
       https://en.wikipedia.org/wiki/Associative_algebra
    """

    @abstractmethod
    def _multiply(self, x, y):
        """ Implementation of multiply
        """

    def multiply(self, x, y):
        """ Calculates the pointwise product of x and y and assigns it to y
        y = x * y
        """
        # Check spaces
        if not self.contains(x):
            raise TypeError('x ({}) is in wrong space'.format(x))

        if not self.contains(y):
            raise TypeError('y ({}) is in wrong space'.format(y))

        self._multiply(x, y)

    class Vector(LinearSpace.Vector):

        def __init__(self, space):
            if not isinstance(space, Algebra):
                raise TypeError(errfmt('''
                'space' ({}) is not an Algebra instance'''.format(space)))
            super().__init__(space)

        def __imul__(self, other):
            """ Overloads the *= operator to mean pointwise multiplication if
            the other object is a vector
            """
            if other in self.space:
                self.space.multiply(other, self)
                return self
            else:
                return super().__imul__(other)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
