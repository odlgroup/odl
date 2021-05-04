.. _linearspace_in_depth:

#############
Linear spaces
#############

The `LinearSpace` class represent abstract mathematical concepts
of vector spaces. It cannot be used directly but are rather intended
to be subclassed by concrete space implementations. The space
provides default implementations of the most important vector space
operations. See the documentation of the respective classes for more
details.

The concept of linear vector spaces in ODL is largely inspired by
the `Rice Vector Library
<http://www.trip.caam.rice.edu/software/rvl/rvl/doc/html/>`_ (RVL).

The abstract `LinearSpace` class is intended for quick prototyping.
It has a number of abstract methods which must be overridden by a
subclass. On the other hand, it provides automatic error checking
and numerous attributes and methods for convenience.

Abstract methods
----------------
In the following, the abstract methods are explained in detail.

Element creation
~~~~~~~~~~~~~~~~

``element(inp=None)``

This public method is the factory for the inner
`LinearSpaceElement` class. It creates a new element of the space,
either from scratch or from an existing data container. In the
simplest possible case, it just delegates the construction to the
`LinearSpaceElement` class.

If no data is provided, the new element is **merely allocated, not
initialized**, thus it can contain *any* value.

**Parameters:**
    inp : `object`, optional
        A container for values for the element initialization.

**Returns:**
    element : `LinearSpaceElement`
        The new element.

Linear combination
~~~~~~~~~~~~~~~~~~

``_lincomb(a, x1, b, x2, out)``

This private method is the raw implementation (i.e. without error
checking) of the linear combination ``out = a * x1 + b * x2``.
`LinearSpace._lincomb` and its public counterpart
`LinearSpace.lincomb` are used to cover a range of convenience
functions, see below.

**Parameters:**
    a, b : scalars, must be members of the space's ``field``
        Multiplicative scalar factors for input element ``x1`` or ``x2``,
        respectively.
    x1, x2 : `LinearSpaceElement`
        Input elements.
    out : `LinearSpaceElement`
        Element to which the result of the computation is written.

**Returns:** `None`

**Requirements:**
 * Aliasing of ``x1``, ``x2`` and ``out`` **must** be allowed.
 * The input elements ``x1`` and ``x2`` **must not** be modified.
 * The initial state of the output element ``out`` **must not**
   influence the result.

Underlying scalar field
~~~~~~~~~~~~~~~~~~~~~~~

``field``

The public attribute determining the type of scalars which
underlie the space. Can be instances of either `RealNumbers` or
`ComplexNumbers` (see `Field`).

Should be implemented as a ``@property`` to make it immutable.

Equality check
~~~~~~~~~~~~~~

``__eq__(other)``

`LinearSpace` inherits this abstract method from `Set`. Its
purpose is to check two `LinearSpace` instances for equality.

**Parameters:**
    other : `object`
        The object to compare to.

**Returns:**
    equals : `bool`
        `True` if ``other`` is the same `LinearSpace`, `False`
        otherwise.


Distance (optional)
~~~~~~~~~~~~~~~~~~~

``_dist(x1, x2)``

A raw (not type-checking) private method measuring the distance
between two elements ``x1`` and ``x2``.

A space with a distance is called a **metric space**.

**Parameters:**
    x1,x2 : `LinearSpaceElement`
        Elements whose mutual distance to calculate.

**Returns:**
    distance : `float`
        The distance between ``x1`` and ``x2``, measured in the space's
        metric

**Requirements:**
    * ``_dist(x, y) == _dist(y, x)``
    * ``_dist(x, y) <= _dist(x, z) + _dist(z, y)``
    * ``_dist(x, y) >= 0``
    * ``_dist(x, y) == 0`` (approx.) if and only if ``x == y`` (approx.)

Norm (optional)
~~~~~~~~~~~~~~~

``_norm(x)``

A raw (not type-checking) private method measuring the length of a
space element ``x``.

A space with a norm is called a **normed space**.

**Parameters:**
    x : `LinearSpaceElement`
        The element to measure.

**Returns:**
    norm : `float`
        The length of ``x`` as measured in the space's norm.

**Requirements:**
 * ``_norm(s * x) = |s| * _norm(x)`` for any scalar ``s``
 * ``_norm(x + y) <= _norm(x) + _norm(y)``
 * ``_norm(x) >= 0``
 * ``_norm(x) == 0`` (approx.) if and only if ``x == 0`` (approx.)

Inner product (optional)
~~~~~~~~~~~~~~~~~~~~~~~~

``_inner(x, y)``

A raw (not type-checking) private method calculating the inner
product of two space elements ``x`` and ``y``.

**Parameters:**
    x,y : `LinearSpaceElement`
        Elements whose inner product to calculate.

**Returns:**
    inner : `float` or `complex`
        The inner product of ``x`` and ``y``. If
        `LinearSpace.field` is the set of real
        numbers, ``inner`` is a `float`, otherwise `complex`.

**Requirements:**
 * ``_inner(x, y) == _inner(y, x)^*`` with '*' = complex conjugation
 * ``_inner(s * x, y) == s * _inner(x, y)`` for ``s`` scalar
 * ``_inner(x + z, y) == _inner(x, y) + _inner(z, y)``
 * ``_inner(x, x) == 0`` (approx.) if and only if ``x == 0`` (approx.)

Pointwise multiplication (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_multiply(x1, x2, out)``

A raw (not type-checking) private method multiplying two elements
``x1`` and ``x2`` point-wise and storing the result in ``out``.

**Parameters:**
    x1, x2 : `LinearSpaceElement`
        Elements whose point-wise product to calculate.
    out : `LinearSpaceElement`
        Element to store the result.

**Returns:** `None`

**Requirements:**
 * ``_multiply(x, y, out) <==> _multiply(y, x, out)``
 * ``_multiply(s * x, y, out) <==> _multiply(x, y, out); out *= s  <==>``
    ``_multiply(x, s * y, out)`` for any scalar ``s``
 * There is a space element ``one`` with
   ``out`` after ``_multiply(one, x, out)`` or ``_multiply(x, one, out)``
   equals ``x``.

Notes
-----
- A normed space is automatically a metric space with the distance
  function ``_dist(x, y) = _norm(x - y)``.
- A Hilbert space (inner product space) is automatically a normed space
  with the norm function ``_norm(x) = sqrt(_inner(x, x))``.
- The conditions on the pointwise multiplication constitute a
  *unital commutative algebra* in the mathematical sense.

References
----------
See Wikipedia's mathematical overview articles
`Vector space
<https://en.wikipedia.org/wiki/Vector_space>`_, `Algebra
<https://en.wikipedia.org/wiki/Associative_algebra>`_.
