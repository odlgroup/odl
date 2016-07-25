.. _functional_in_depth:

#####################
Functional
#####################

A *functional* is an operator ``f`` that maps from some vector space ``X`` to the field of scalars ``F`` associated with the vector space:

.. math::
   
   f : X \to F.

In the ODL solver package, functionals are implemented in the `Functional` class, a subclass to `Operator`.

From a mathematical presepective, the above is a valide definition of a functional. However, since the purpose of these functionals are primarely to be used for solving optimization problems, the following assumptions are made:

 * the vector space ``X`` is a Hilbert space.
 * the field of scalars ``F`` are the real numbers.

It is possible to use the class without fulfilling these assumtptions, however in this case some of the mathematical results might not hold.


Implementation of functionals
=============================

To define your own functional, you start by writing::

    class MyFunctional(odl.solvers.Functional):
        ...

Since `Functional` is a subclass of `Operator`, it has the *abstract method* ``domain`` which need to be explicitly overridden by any subclass.

``domain``: `Set`
    The domain of this functional, i.e., the set of elements to which this functional can be applied.

Moreover, there are several optional parameters associated with a functional. These are

``linear`` : `bool`, optional
    If `True`, the functional is considered as linear. In this case, ``domain`` and ``range`` have to be instances of `LinearSpace`, or `Field`.
    Default: ``False``
smooth : `bool`, optional
    If `True`, assume that the functional is continuously differentiable.
    Default: ``False``
concave : `bool`, optional
    If `True`, assume that the functional is concave.
    Default: ``False``
convex : `bool`, optional
    If `True`, assume that the functional is convex.
    Default: ``False``
grad_lipschitz : 'float', optional
    The Lipschitz constant of the gradient.
    Default: infinity

A functional also has a number of optional methods and properties associated with it. The default value of these are all to raise a `NotImplemetedError`. The properties are:

 * ``gradient``. This returns the gradient operator of the functional, i.e., the operator that corresponds to the mapping

.. math::

    x \\to \\nabla f(x)

where :math:`\\nabla f(x)` is the element used to evaluate derivatives in a direction :math:`d` by :math:`\\langle \\nabla f(x), d \\rangle`.
 * ``conjugate_functional``. This is the convex conjugate functional, also known as the Legendre transform or Fenchel conjugate. It is defined as 

.. math::

    f^*(x^*) = \sup_{x \in X} \{ \langle x^*,x \rangle - f(x)  \}.

For general linear spaces :math:`X`, :math:`x^* \in X^*` the (continuous/normed) dual space of :math:`X`, i.e., the space of all continuous linear functionals defined on :math:`X`. Then :math:`\langle x^*,x \rangle` is the "dual pairing", i.e., the evaluation of the linear functional :math:`x^*` in the point :math:`x`. However, Hilbert spaces are self-dual, meaning :math:`X^* = X`, and :math:`\langle x^*,x \rangle` is the inner product.

The optional method is

 * ``proximal(sigma)``. This returns the proximal operator of the functional, where ``sigma`` is a nonnegative step-size like parameter. The proximal operator is defined as

.. math::

    \text{prox}_{\sigma f}(x) = \text{arg min}_{y \in X} \{ f(y) + \frac{1}{2\sigma} \|y - x\|_2^2 \}

The `Functional` class also contains two default implementations of two help functions:

* ``derivative(point)``. Given an implementation of the gradient, this method return the (directional) derivative operator in ``point``. This is the linear operator 

.. math::
    x \to \langle x, \nabla f(point) \rangle,

where :math:`\nabla f(point)` is the gradient of the functional in the point :math:`point`.
* ``translate(shift)``. Give a functional :math:`f(.)`, this method creates the functional :math:`f(. - shift)`


Functional arithmetics
======================
It is common in applications to perform arithmetic operations with functionals, for example adding two functionals :math:`f` and :math:`g`

.. math::
   [f+g](x) = f(x) + g(x),

or multiplication of a functional by a scalar

.. math::
   [\alpha f](x) = \alpha f (x).

Another example is translating a functional with a vecotr :math:`y`

.. math::
   f(x - y),

or given an `Operator` :math:`A` whose range is the same as the domain as the functional we also have composition

.. math::
    [f * A](x) = f(A(x)). 

In some of these cases, properties and methods such as ``gradient``, ``convex_conjugate`` and ``proximal`` can be calculated automatically given a default implementation of the corresponding property in :math:`f`.

All available functional arithmetic, including which properties and methods that automatically can be calculated, is shown below. ``f``, ``g`` represent `Functional`'s, and ``A`` an `Operator` whose range is the same as the domain as the functional. `` a`` is a scalar in the field of the domain of ``f`` and ``g``, and ``y`` is a vector in the domain of ``f`` and ``g``.

+------------------+-----------------+-------------------------------------------------+
| Code             | Meaning         | Class                                           |
+==================+=================+=================================================+
| ``(f + g)(x)``   | ``f(x) + g(x)`` | `FunctionalSum`                                 |
|                  |                 | - Retains `gradient`.                           |
+------------------+-----------------+-------------------------------------------------+
| ``(f + a)(x)     | ``f(x) + a``    | `FunctionalScalarSum`                           | 
|                  |                 | - Retains all properties.                       |
+------------------+-----------------+-------------------------------------------------+
| ``(f * A)(x)``   | ``f(A(x))``     | `FunctionalComp`                                |
|                  |                 | - Retains gradient                              |
+------------------+-----------------+-------------------------------------------------+
| ``(a * f)(x)``   | ``a * f(x)``    | `FunctionalLeftScalarMult`                      |
|                  |                 | - Retains all properties, if ``a`` is positive. |
+------------------+-----------------+-------------------------------------------------+
| ``(f * a)(x)``   | ``f(a * x)``    | `FunctionalRightScalarMult`                     |
|                  |                 | - Retains all properties                        |
+------------------+-----------------+-------------------------------------------------+
| ``(v * f)(x)``   | ``v * f(x)``    | `FunctionalLeftVectorMult`                      |
|                  |                 | - Note that this is not a functional anymore.   |
+------------------+-----------------+-------------------------------------------------+
| ``(f * v)(x)``   | ``f(v * x)``    | `FunctionalRightVectorMult`                     |
|                  |                 | - Retains gradient and convex conjugate.        |
+------------------+-----------------+-------------------------------------------------+
| ``f.translate(y) | ``f(. - y)``    | `TranslatedFunctional`                          |
|                  |                 | - Retains all properties.                       |
+------------------+-----------------+-------------------------------------------------+







