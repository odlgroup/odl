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

The first assumption is made in order to simplify the concept of *convex conjugate functional*, see ``conjugate_functional`` under :ref:`implementation` for more details or the Wikipedia articles on `convex conjugate`_ or `Legendre transformation`_.

The second assumption is made in order to guarantee that the optimization problem, in some way, makes sense. For example, if the field is not (totally) ordered (like for example the complex numbers) one cannot define optimization over it in a meaningful way. Or if the field does not have the least-upper-bound property, then attainment of an optimal solution cannot be guaranteed. See, for example, the Wikipedia articles on `field`_, `ordered field`_ and `least-upper-bound property`_.

Note that it is possible to use the class without fulfilling these assumtptions, however in this case some of the mathematical results might not hold.

.. _convex conjugate: https://en.wikipedia.org/wiki/Convex_conjugate
.. _Legendre transformation: https://en.wikipedia.org/wiki/Legendre_transformation

.. _field: https://en.wikipedia.org/wiki/Field_(mathematics)
.. _ordered field: https://en.wikipedia.org/wiki/Ordered_field
.. _least-upper-bound property: https://en.wikipedia.org/wiki/Least-upper-bound_property

.. _implementation:

Implementation of functionals
=============================

To define your own functional, you start by writing::

    class MyFunctional(odl.solvers.Functional):
        ...

Since `Functional` is a subclass of `Operator`, it has the *abstract method* ``domain`` which need to be explicitly overridden by any subclass.

``domain``: `Set`
    The domain of this functional, i.e., the set of elements to which this functional can be applied.

Moreover, there are several optional parameters associated with a functional. These are

``linear``: `bool`, optional
    If `True`, the functional is considered as linear. In this case, ``domain`` and ``range`` have to be instances of `LinearSpace`, or `Field`.
    Default: ``False``
``smooth``: `bool`, optional
    If `True`, assume that the functional is continuously differentiable.
    Default: ``False``
``concave``: `bool`, optional
    If `True`, assume that the functional is concave.
    Default: ``False``
``convex``: `bool`, optional
    If `True`, assume that the functional is convex.
    Default: ``False``
``grad_lipschitz``: `float`, optional
    The Lipschitz constant of the gradient.
    Default: infinity

A functional also has a number of optional methods and properties associated with it. The default value of these are all to raise a `NotImplemetedError`. The properties are:

 * ``gradient``. This returns the gradient operator of the functional, i.e., the operator that corresponds to the mapping :math:`x \to \nabla f(x)`, where :math:`\nabla f(x)` is the element used to evaluate derivatives in a direction :math:`d` by :math:`\langle \nabla f(x), d \rangle`.
 * ``conjugate_functional``. This is the convex conjugate functional, also known as the Legendre transform or Fenchel conjugate. It is defined as :math:`f^*(x^*) = \sup_{x \in X} \{ \langle x^*,x \rangle - f(x)  \}`, where the element :math:`x^*` lives in the space :math:`X^*`, the (continuous/normed) dual space of :math:`X`, i.e., the space of all continuous linear functionals defined on :math:`X`. Moreover, the notation :math:`\langle x^*,x \rangle` is the "dual pairing" between :math:`X` and :math:`X^*`, i.e., the evaluation of the linear functional :math:`x^*` in the point :math:`x`. However, Hilbert spaces are self-dual, meaning :math:`X^* = X`, and :math:`\langle x^*,x \rangle` is then the inner product.

The optional method is:

 * ``proximal(sigma)``. This returns the proximal operator of the functional, where ``sigma`` is a nonnegative step-size like parameter. The proximal operator is defined as :math:`\text{prox}_{\sigma f}(x) = \text{arg min}_{y \in X} \{ f(y) + \frac{1}{2\sigma} \|y - x\|_2^2 \}`. 

The `Functional` class also contains two default implementations of two help functions:

* ``derivative(point)``. Given an implementation of the gradient, this method return the (directional) derivative operator in ``point``. This is the linear operator :math:`x \to \langle x, \nabla f(point) \rangle`, where :math:`\nabla f(point)` is the gradient of the functional in the point :math:`point`.
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

or given an `Operator` :math:`A` whose range is the same as the domain of the functional we also have composition

.. math::
    [f * A](x) = f(A(x)). 

In some of these cases, properties and methods such as ``gradient``, ``convex_conjugate`` and ``proximal`` can be calculated automatically given a default implementation of the corresponding property in :math:`f`.

All available functional arithmetic, including which properties and methods that automatically can be calculated, is shown below. ``f``, ``g`` represent `Functional`'s, and ``A`` an `Operator` whose range is the same as the domain as the functional. ``a`` is a scalar in the field of the domain of ``f`` and ``g``, and ``y`` is a vector in the domain of ``f`` and ``g``.

+--------------------+-----------------+-------------------------------------------------+
| Code               | Meaning         | Class                                           |
+====================+=================+=================================================+
| ``(f + g)(x)``     | ``f(x) + g(x)`` | `FunctionalSum`                                 |
|                    |                 | - Retains `gradient`.                           |
+--------------------+-----------------+-------------------------------------------------+
| ``(f + a)(x)``     | ``f(x) + a``    | `FunctionalScalarSum`                           |
|                    |                 | - Retains all properties.                       |
+--------------------+-----------------+-------------------------------------------------+
| ``(f * A)(x)``     | ``f(A(x))``     | `FunctionalComp`                                |
|                    |                 | - Retains gradient                              |
+--------------------+-----------------+-------------------------------------------------+
| ``(a * f)(x)``     | ``a * f(x)``    | `FunctionalLeftScalarMult`                      |
|                    |                 | - Retains all properties, if ``a`` is positive. |
+--------------------+-----------------+-------------------------------------------------+
| ``(f * a)(x)``     | ``f(a * x)``    | `FunctionalRightScalarMult`                     |
|                    |                 | - Retains all properties                        |
+--------------------+-----------------+-------------------------------------------------+
| ``(v * f)(x)``     | ``v * f(x)``    | `FunctionalLeftVectorMult`                      |
|                    |                 | - This is not a functional, but an operator     |
+--------------------+-----------------+-------------------------------------------------+
| ``(f * v)(x)``     | ``f(v * x)``    | `FunctionalRightVectorMult`                     |
|                    |                 | - Retains gradient and convex conjugate.        |
+--------------------+-----------------+-------------------------------------------------+
| ``f.translate(y)`` | ``f(. - y)``    | `TranslatedFunctional`                          |
|                    |                 | - Retains all properties.                       |
+--------------------+-----------------+-------------------------------------------------+


Code example
============
This section contains an example of an implementation of a functional, namely the functional :math:`\|x\|_2^2 + \langle x, y \rangle`. The code is found in the file `functional_basic_example.py`, and more implementations of other functionals can be found in `default_functionals.py`. ::

    # Here we define the functional
    class MyFunctional(odl.solvers.Functional):
        """This is my functional: ||x||_2^2 + <x, y>."""

        def __init__(self, domain, y):
            # This comand calls the init of Functional and sets a number of
            # parameters associated with a functional. All but domain have default
            # values if not set.
            super().__init__(domain=domain, linear=False, convex=True,
                             concave=False, smooth=True, grad_lipschitz=2)

            # We need to check that y is in the domain. Then we store the value of
            # y for future use.
            if y not in domain:
                raise TypeError('y is not in the domain!')
            self._y = y

        # Property that returns the linear term.
        @property
        def y(self):
            return self._y

        # Defining the _call function
        def _call(self, x):
            return x.norm()**2 + x.inner(self.y)

        # Next we define the gradient. Note that this is a property.
        @property
        def gradient(self):

            # The class corresponding to the gradient operator.
            class MyGradientOperator(odl.Operator):
                """Class that implements the gradient operator of the functional
                ``||x||_2^2 + <x,y>``.
                """

                def __init__(self, functional):
                    super().__init__(domain=functional.domain,
                                     range=functional.domain)

                    self._functional = functional

                def _call(self, x):
                    return 2.0 * x + self._functional.y

            return MyGradientOperator(functional=self)

        # Next we define the convex conjugate functional.
        @property
        def conjugate_functional(self):
            # This functional is implemented below.
            return MyFunctionalConjugate(domain=self.domain, y=self.y)


    # Here is the conjugate functional.
    class MyFunctionalConjugate(odl.solvers.Functional):
        """Conjugate functional to ``||x||_2^2 + <x,y>``.

        Calculations give that this funtional has the analytic expression
        f^*(x) = ||x||^2/2 - ||x-y||^2/4 + ||y||^2/2 - <x,y>.
        """
        def __init__(self, domain, y):
            super().__init__(domain=domain, linear=False, convex=True,
                             concave=False, smooth=True, grad_lipschitz=2)

            if y not in domain:
                raise TypeError('y is not in the domain!')
            self._y = y

        @property
        def y(self):
            return self._y

        def _call(self, x):
            return (x.norm()**2 / 2.0 - (x - self.y).norm()**2 / 4.0 +
                    self.y.norm()**2 / 2.0 - x.inner(self.y))

With this code, one can now create things like the conjugate functional of a scaled and translated version::

    my_func = MyFunctional(domain=space, linear_term=linear_term)
    (scalar * my_func).translate(translation).conjugate_functional

