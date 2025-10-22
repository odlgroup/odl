.. _functional_guide:

#####################
Functional
#####################

A *functional* is an operator :math:`S: X \to \mathbb{F}` that maps that maps from some vector space :math:`X` to the field of scalars :math:`F`.

In the ODL `solvers` package, functionals are implemented in the `Functional` class, a subclass to `Operator`.

From a mathematical perspective, the above is a valid definition of a functional.
However, since these functionals are primarily to be used for solving optimization problems, the following assumptions are made:

 * the vector space :math:`X` is a Hilbert space.
 * the field of scalars :math:`F` are the real numbers.

The first assumption is made in order to simplify the concept of *convex conjugate functional*, see ``convex_conj`` under :ref:`functional_guide__implementation` for more details, or the Wikipedia articles on `convex conjugate`_ and `Legendre transformation`_.

The second assumption is made in order to guarantee that we use a well-ordered set (in contrast to e.g. the complex numbers) over which optimization problems can be meaningfully defined, and that optimal solutions are in fact obtained.
See, for example, the Wikipedia articles on `field`_, `ordered field`_ and `least-upper-bound property`_.

Note that these conditions are not explicitly checked.
However, using the class in violation to the above assumptions might lead to unknown behavior since some of the mathematical results might not hold.
Also note that most of the theory, and most solvers, requires the functional to be convex.
However this property is not stored or checked anywhere in the class.
It is therefore the users responsibility to ensure that a functional has the properties required for a given optimization method.

The intended use of the `Functional` class is, as mentioned above, to be used when formulating and solving optimization problems.
One main difference with the `Operator` class is thus that it contains notions specially intended for optimization, such as *convex conjugate functional* and *proximal operator*.
For more information on these concepts, see ``convex_conj`` and ``proximal`` under :ref:`functional_guide__implementation`.
There is also a certain type of arithmetics associated with functionals, for more on this see :ref:`functional_guide__arithmetic`.


.. _convex conjugate: https://en.wikipedia.org/wiki/Convex_conjugate
.. _Legendre transformation: https://en.wikipedia.org/wiki/Legendre_transformation

.. _field: https://en.wikipedia.org/wiki/Field_(mathematics)
.. _ordered field: https://en.wikipedia.org/wiki/Ordered_field
.. _least-upper-bound property: https://en.wikipedia.org/wiki/Least-upper-bound_property


.. _functional_guide__implementation:

Implementation of functionals
=============================

To define your own functional, start by writing::

    class MyFunctional(odl.solvers.Functional):

        """Docstring goes here."""

        def __init__(self, space):
            # Sets `Operator.domain` to `space` and `Operator.range` to `space.field`
            super(MyFunctional, self).__init__(space)

        ...

`Functional` needs to be provided with a space, i.e., the domain on which it is defined, from which it infers the range.

 * ``space``: `LinearSpace`
            The domain of this functional, i.e., the set of elements to
            which this functional can be applied.

Moreover, there are two optional parameters that can be provided in the initializer.
These are ``linear``, which indicates whether the functional is linear or not, and ``grad_lipschitz``, which is the Lipschitz constant of the gradient.

 * ``linear``: bool, optional
            If `True`, the functional is considered as linear.

 * ``grad_lipschitz``: float, optional
            The Lipschitz constant of the gradient.

A functional also has three optional properties and one optional method associated with it.
The properties are:

 * ``functional.gradient``. This returns the gradient operator of the functional, i.e., the operator that corresponds to the mapping

   .. math::

      x \to \nabla S(x),

   where :math:`\nabla S(x)` is the the space element representing the Frechet derivative (directional derivative) at the point :math:`x`

   .. math::

      S'(x)(g) = \langle \nabla S(x), g \rangle \quad \text{for all } g \in X.

   See also `Functional.derivative`.


 * ``functional.convex_conj``. This is the convex conjugate of the functional, itself again a functional, which is also known as the `Legendre transform`_ or `Fenchel conjugate`_.
   It is defined as

   .. math::

      S^*(x^*) = \sup_{x \in X} \{ \langle x^*,x \rangle - S(x)  \},

   where :math:`x^*` is an element in :math:`X` and :math:`\langle x^*,x \rangle` is the inner product.
   (Note that :math:`x^*` should live in the space :math:`X^*`, the (continuous/normed) `dual space`_ of :math:`X`, however since we assume that :math:`X` is a Hilbert space we have :math:`X^* = X`).

 * ``proximal``. This returns a `proximal factory` for the proximal operator of the functional.
   The proximal operator is defined as

   .. math::

      \text{prox}_{\sigma S}(x) = \text{arg min}_{y \in X} \{ S(y) + \frac{1}{2\sigma} \|y - x\|_2^2 \}.

The default behavior of these is to raise a ``NotImplemetedError``.

The `Functional` class also contains default implementations of two helper functions:

 * ``derivative(point)``. Given an implementation of the gradient, this method returns the (directional) derivative operator in ``point``.
   This is the linear operator

   .. math::

      x \to \langle x, \nabla S(p) \rangle,

   where :math:`\nabla S(p)` is the gradient of the functional in the point :math:`p`.

 * ``translated(shift)``. Given a functional :math:`S` and a shift :math:`y`, this method creates the functional :math:`S(\cdot - y)`.


.. _dual space: https://en.wikipedia.org/wiki/Dual_space
.. _Legendre transform: https://en.wikipedia.org/wiki/Legendre_transformation
.. _Fenchel conjugate: https://en.wikipedia.org/wiki/Convex_conjugate


.. _functional_guide__arithmetic:

Functional arithmetic
=====================
It is common in applications to perform arithmetic operations with functionals, for example adding two functionals :math:`S` and :math:`T`:

.. math::
   [S+T](x) = S(x) + T(x),

or multiplication of a functional by a scalar:

.. math::
   [\alpha S](x) = \alpha S (x).

Another example is translating a functional with a vector :math:`y`:

.. math::
   translate(S(x), y) = S(x - y),

or given an `Operator` :math:`A` whose range is the same as the domain of the functional we also have composition:

.. math::
    [S * A](x) = S(A(x)).

In some of these cases, properties and methods such as ``gradient``, ``convex_conjugate`` and ``proximal`` can be calculated automatically given a default implementation of the corresponding property in :math:`S` and :math:`T`.

All available functional arithmetic, including which properties and methods that automatically can be calculated, is shown below.
``S``, ``T`` represent `Functional`'s with common domain and range, and ``A`` an `Operator` whose range is the same as the domain of the functional.
``a`` is a scalar in the field of the domain of ``S`` and ``T``, and ``y`` is a vector in the domain of ``S`` and ``T``.

+---------------------+-----------------+--------------------------------------------------------------------------------+
| Code                | Meaning         | Class                                                                          |
+=====================+=================+================================================================================+
| ``(S + T)(x)``      | ``S(x) + T(x)`` | `FunctionalSum`                                                                |
|                     |                 | - Retains `Functional.gradient`.                                               |
+---------------------+-----------------+--------------------------------------------------------------------------------+
| ``(S + a)(x)``      | ``S(x) + a``    | `FunctionalScalarSum`                                                          |
|                     |                 | - Retains all properties.                                                      |
|                     |                 | Note that this never means scaling of the argument.                            |
+---------------------+-----------------+--------------------------------------------------------------------------------+
| ``(S * A)(x)``      | ``S(A(x))``     | `FunctionalComp`                                                               |
|                     |                 | - Retains `Functional.gradient`.                                               |
+---------------------+-----------------+--------------------------------------------------------------------------------+
| ``(a * S)(x)``      | ``a * S(x)``    | `FunctionalLeftScalarMult`                                                     |
|                     |                 | - Retains all properties if ``a`` is positive.                                 |
|                     |                 | Otherwise only `Functional.gradient` and `Functional.derivative` are retained. |
+---------------------+-----------------+--------------------------------------------------------------------------------+
| ``(S * a)(x)``      | ``S(a * x)``    | `FunctionalRightScalarMult`                                                    |
|                     |                 | - Retains all properties.                                                      |
+---------------------+-----------------+--------------------------------------------------------------------------------+
| ``(v * S)(x)``      | ``v * S(x)``    | `FunctionalLeftVectorMult`                                                     |
|                     |                 | - Results in an operator rather than a functional.                             |
+---------------------+-----------------+--------------------------------------------------------------------------------+
| ``(S * v)(x)``      | ``S(v * x)``    | `FunctionalRightVectorMult`                                                    |
|                     |                 | - Retains gradient and convex conjugate.                                       |
+---------------------+-----------------+--------------------------------------------------------------------------------+
| ``f.translated(y)`` | ``f(. - y)``    | `FunctionalTranslation`                                                        |
|                     |                 | - Retains all properties.                                                      |
+---------------------+-----------------+--------------------------------------------------------------------------------+


Code example
============
This section contains an example of an implementation of a functional, namely the functional :math:`\|x\|_2^2 + \langle x, y \rangle`.
Another example can be found in `functional_basic_example.py <https://github.com/odlgroup/odl/blob/master/examples/solvers/functional_basic_example.py>`_, and more implementations of other functionals can be found in `default_functionals.py <https://github.com/odlgroup/odl/blob/master/odl/solvers/functional/default_functionals.py>`_.

.. literalinclude:: code/functional_indepth_example.py
   :language: python

