.. _derivatives_in_depth:

######################################
On the different notions of derivative
######################################

The concept of a derivative is one of the core concepts of mathematical analysis, and it is essential whenever a linear approximation of a function in some point is required.
Since the notion of derivative has different meanings in different contexts, the intention of this guide is to introduce the derivative concepts as used in ODL.

In short, the derivative notions that will be discussed here are:

* **Derivative**. When we write "derivative" in ODL code and documentation, we mean the derivative of an `Operator` :math:`A : X \to Y` w.r.t to a disturbance in its argument, i.e a linear approximation of :math:`A(x + h)` for small :math:`h`.
  The derivative in a point :math:`x` is an `Operator` :math:`A'(x) : X \to Y`.

* **Gradient**. If the operator :math:`A` is a `functional`, i.e. :math:`A : X \to \mathbb{R}`, then the gradient is the direction in which :math:`A` increases the most.
  The gradient in a point :math:`x` is a vector :math:`[\nabla A](x)` in :math:`X` such that :math:`A'(x)(y) = \langle [\nabla A](x), y \rangle`.
  The gradient operator is the operator :math:`x \to [\nabla A](x)`.

* **Hessian**. The hessian in a point :math:`x` is the derivative operator of the gradient operator, i.e. :math:`H(x) = [\nabla A]'(x)`.

* **Spatial Gradient**. The spatial gradient is only defined for spaces :math:`\mathcal{F}(\Omega, \mathbb{F})` whose elements are functions over some domain :math:`\Omega \subset \mathbb{R}^d` taking values in :math:`\mathbb{R}` or :math:`\mathbb{C}`.
  It can be seen as a vectorized version of the usual gradient, taken in each point in :math:`\Omega`.

* **Subgradient**. The subgradient extends the notion of derivative to any convex functional and is used in some optimization solvers where the objective function is not differentiable.

Derivative
##########
The derivative is usually introduced for functions :math:`f: \mathbb{R} \to \mathbb{R}` via the limit

.. math::
    f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}.

Here we say that the derivative of :math:`f` in :math:`x` is :math:`f'(x)`.

This limit makes sense in one dimension, but once we start considering functions in higher dimension we get into trouble.
Consider :math:`f: \mathbb{R}^n \to \mathbb{R}^m` -- what would :math:`h` mean in this case?
An extension is the concept of a directional derivative.
The derivative of :math:`f` in :math:`x` in *direction* :math:`d` is :math:`f'(x)(d)`:

.. math::
    f'(x)(d) = \lim_{h \to 0} \frac{f(x + dh) - f(x)}{h}.

Here we see (as implied by the notation) that :math:`f'(x)` is actually an operator

.. math::
    f'(x) : \mathbb{R}^n \to \mathbb{R}^m.

This notion of derivative is called **Gâteaux derivative**.

If we add the explicit requirement that :math:`f'(x)` is a linear approximation of :math:`f` at :math:`x`, we can rewrite the definition as

.. math::
   \lim_{\|d\| \to 0} \frac{\| f(x + d) - f(x) - f'(x)(d) \|}{\|d\|} = 0,

where the limit has to be uniform in :math:`d`.
This notion naturally extends to an `Operator` :math:`f : X \to Y` between Banach spaces :math:`X` and :math:`Y` with norms :math:`\| \cdot \|_X` and :math:`\| \cdot \|_Y`, respectively.
Here :math:`f'(x)` is defined as the linear operator (if it exists) that satisfies

.. math::
   \lim_{\| d \| \to 0} \frac{\| f(x + d) - f(x) - f'(x)(d) \|_Y}{\| d \|_X} = 0.

This definition of the derivative is called the **Fréchet derivative**.
If it exists, it coincides with the Gâteaux derivative.
This is the case for most operators, but some are only differentiable in the Gâteaux sense, not in the Fréchet sense.

Another important difference between the two notions is that the Gâteaux variant (directional derivative) can be approximated by finite differences in a simple way, as it is done in ODL's `NumericalDerivative`, while there is no simple way to computationally realize the Fréchet definition.
Therefore, "derivative" in ODL generally means "Gâteaux derivative", which is the same as "Fréchet derivative" except for a few special cases.

Rules for the derivative
~~~~~~~~~~~~~~~~~~~~~~~~
Many of the usual rules for derivatives also hold for the operator derivatives, i.e.

* Linearity

  .. math::
      (a f + b g)'(x)(y) = a f'(x)(y) + b g'(x)(y)

* Chain rule

  .. math::
      (g \circ f)'(x)(y) = \Big[ g'\big(f(x)\big) \circ f'(x) \Big](y)

* Linear operators are their own derivatives. If :math:`f` is linear, then

  .. math::
     f'(x)(y) = f(y)

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~
* The derivative is implemented in ODL for `Operator`'s via the `Operator.derivative` method.
* It can be numerically computed using the `NumericalDerivative` operator.
* Many of the operator arithmetic classes implement the usual rules for the derivative, such as the chain rule, distributivity over addition etc.

Gradient
########
In the classical setting of functions :math:`f : \mathbb{R}^n \to \mathbb{R}`, the gradient is the vector

.. math::
    \nabla f =
    \begin{bmatrix}
        \dfrac{\partial f}{\partial x_1}
        \dots
        \dfrac{\partial f}{\partial x_n}
    \end{bmatrix}

This can be generalized to the setting of functionals :math:`f : X \to \mathbb{R}` mapping elements in some Banach space :math:`X` to the real numbers by noting that the Fréchet derivative can be written as

.. math::
    f'(x)(y) = \langle y, [\nabla f](x) \rangle,

where :math:`[\nabla f](x)` lies in the dual space of :math:`X`, denoted :math:`X^*`. For most spaces in ODL, the spaces are *Hilbert* spaces where :math:`X = X^*` by the `Riesz representation theorem
<https://en.wikipedia.org/wiki/Riesz_representation_theorem>`_ and hence :math:`[\nabla f](x) \in X`.

We call the (possibly nonlinear) operator :math:`x \to [\nabla f](x)` the *Gradient operator* of :math:`f`.

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~
* The gradient is implemented in ODL `Functional`'s via the `Functional.gradient` method.
* It can be numerically computed using the `NumericalGradient` operator.

Hessian
#######
For functions :math:`f : \mathbb{R}^n \to \mathbb{R}`, the Hessian in a point :math:`x` is the matrix :math:`H(x)` such that

.. math::
    H(x) =
    \begin{bmatrix}
    \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_n} \\
    \dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \dfrac{\partial^2 f}{\partial x_n\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
    \end{bmatrix}

with the derivatives are evaluated in the point :math:`x`.
It has the property that that the quadratic variation of :math:`f` is

.. math::
    f(x + d) = f(x) + \langle d, [\nabla f](x)\rangle + \frac{1}{2}\langle d, [H(x)](d)\rangle + o(\|d\|^2),

but also that the derivative of the gradient operator is

.. math::
    \nabla f(x + d) = [\nabla f](x) + [H(x)](d) + o(\|d\|).

If we take this second property as the *definition* of the Hessian, it can easily be generalized to the setting of functionals :math:`f : X \to \mathbb{R}` mapping elements in some Hilbert space :math:`X` to the real numbers.

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~
The Hessian is not explicitly implemented anywhere in ODL.
Instead it can be used in the form of the derivative of the gradient operator.
This is however not implemented for all functionals.

* For an example of a functional whose gradient has a derivative, see `RosenbrockFunctional`.
* It can be computed by taking the `NumericalDerivative` of the gradient, which can in turn be computed using the `NumericalGradient`.

Spatial Gradient
################
The spatial gradient of a function :math:`f \in \mathcal{F}(\Omega, \mathbb{R}) = \{f: \Omega \to \mathbb{R}\}` (with adequate differentiability properties) is an element in the function space :math:`\mathcal{F}(\Omega, \mathbb{R}^n)` such that for any :math:`x, d \in \Omega`:

.. math::
    \lim_{h \to 0} \frac{\| f(x + h d) - f(x) - \langle h d, \text{grad} f(x) \rangle \|}{h} = 0

It is identical to the above notion of functional gradient for the special case of functions :math:`\Omega \to \mathbb{R}`.

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~
* The spatial gradient is implemented in ODL in the `Gradient` operator.
* Several related operators such as the `PartialDerivative` and `Laplacian` are also available.

Subgradient
###########
The Subgradient (also *subderivative* or *subdifferential*) of a *convex* function :math:`f : X \to \mathbb{R}`, mapping a Banach space :math:`X` to :math:`\mathbb{R}`, is defined as the set-valued function :math:`\partial f : X \to 2^{X^*}` whose values are:

.. math::
   [\partial f](x_0) = \{c \in X^* \ s.t. \ f(x) - f(x_0) \geq \langle c , x - x_0 \rangle \forall x \in X \}.

For differentiable functions, this reduces to the singleton set containing the usual gradient.

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~
The subgradient is not explicitly implemented in ODL, but is implicitly used in the proximal operators.
See :ref:`proximal_operators` for more information.

Notes on complex spaces
#######################
All of the above definitions assume that the involved spaces are vector spaces over the field of real numbers.
For complex spaces, there are two possible ways to generalize the above concepts:

1. Complex space as the product of two real spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we indentify a space :math:`X(\mathbb{C})`, for instance :math:`L^2(\Omega, \mathbb{C})` or :math:`\mathbb{C}^n`, with the product space :math:`X(\mathbb{R})^2` using the bijective mapping

.. math::
    E(f) = \big( \Re(f),\, \Im(f) \big).

This purely geometric view is the practically more relevant one since it allows to simply adopt all rules for real spaces in the complex case.
It is endorsed in ODL unless otherwise stated.

2. Complex derivative
~~~~~~~~~~~~~~~~~~~~~
The complex derivative is a notion from `complex analysis <https://en.wikipedia.org/wiki/Complex_analysis>`_ that has vastly more far-reaching consequences than differentiability of real and imaginary parts separately.
Since complex differentiable functions are automatically infinitely many times differetiable, this derivative notion strongly restricts the class of functions to which its rules can be applied, thereby limiting the usefulness for our purposes.

For instance, the Gâteaux derivative of an operator :math:`f` between complex spaces would be defined as

.. math::
    f'(x)(y) = \lim_{z \to 0} z^{-1} \big( f(x + zy) - f(x) \big),

with the difference that here, the limit :math:`z \to 0` is understood as going along arbitrary curves in the complex plane that end up at 0.
This definition is both harder to calculate explicitly and harder to approximate numerically.

Complex <-> Real mappings
~~~~~~~~~~~~~~~~~~~~~~~~~
Some operators are defined as mapping from a complex space to a real space, or vice versa.
Typical examples are the real-to-complex Fourier transform, or taking the real part of a function or vector.
Such operators are somewhat corner cases of functional analysis that are not well covered in the literature.

A peculiar issue with this setup is that linearity in domain and range have to be checked with different sets of scalars.
In particular, testing linearity with complex scalars is invalid in real spaces, such that these kinds of operators can never be formally complex-linear, only linear in the sense of identifying a complex number with a 2-vector of real numbers.

Another issue is adjointness: When defining the adjoint with respect to the :math:`\mathbb{C} = \mathbb{R}^2` identification, "lossy" operators do not satisfy the adjoint condition fully.
For instance, the real part operator :math:`\Re: L^2(\Omega, \mathbb{C}) \to L^2(\Omega, \mathbb{R})` can be rewritten as a projection operator

.. math::
    \Re: L^2(\Omega, \mathbb{R})^2 \to L^2(\Omega, \mathbb{R}), \quad
    \Re(f) = f_1,

and as such it is linear and has the adjoint :math:`\Re^*(g) = (g, 0)`.
However, when transferring this back to the complex interpretation, we get

.. math::
    \langle \Re(f),\, g\rangle_{L^2(\Omega, \mathbb{R})} = \int \Re(f)(x)\, g(x)\, \mathrm{d}x

but

.. math::
    \langle f,\, \Re^*(g)\rangle_{L^2(\Omega, \mathbb{C})} = \int \big[ \Re(f)(x)\, g(x) + \mathrm{i}\,\Im(f)(x)\, g(x) \big] \, \mathrm{d}x.

Therefore, ODL takes the following pragmatic approach for complex <-> real operators:

- Derivatives are taken in the real sense.
  Linearity is set to `True` for an operator :math:`A: X \to Y` if :math:`A'(x) = A` for all :math:`x\in X`.
  This property can be used to optimize calculations with derivatives, since the derivative operator does not depend on the point.
  Linearity in the sense of complex vector spaces is currently not reflected by any flag in ODL.
- Even for formally non-linear derivative operators, an adjoint can be defined, which will not be complex-linear, either.
  It satisfies the adjointness test only when comparing real-valued inner products.
