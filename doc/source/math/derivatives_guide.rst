.. _derivatives_in_depth:

######################################
On the different notions of derivative
######################################

The concept of a derivative is one of the core concepts of mathematical analysis analysis, and it is essential whenever a linear approximation of a function in some point is required.
Since the notion of derivative has different meanings in different contexts, this guide has been written to introduce the different derivative concepts used in ODL.

In short, different notions of derivatives that will be discussed here are:

* **Derivative**. When we write "derivative" in ODL code and documentation, we mean the derivative of an `Operator` :math:`A : \mathcal{X} \rightarrow \mathcal{Y}` w.r.t to a disturbance in :math:`x`, i.e a linear approximation of :math:`A(x + h)` for small :math:`h`.
  The derivative in a point :math:`x` is an `Operator` :math:`A'(x) : \mathcal{X} \rightarrow \mathcal{Y}`.

* **Gradient**. If the operator :math:`A` is a `functional`, i.e. :math:`A : \mathcal{X} \rightarrow \mathbb{R}`, then the gradient is the direction in which :math:`A` increases the most.
  The gradient in a point :math:`x` is a vector :math:`[\nabla A](x)` in :math:`\mathcal{X}` such that :math:`A'(x)(y) = \langle [\nabla A](x), y \rangle`.
  The gradient operator is the operator :math:`x \rightarrow [\nabla A](x)`.

* **Hessian**. The hessian in a point :math:`x` is the derivative operator of the gradient operator, i.e. :math:`H(x) = [\nabla A]'(x)`.

* **Spatial Gradient**. The spatial gradient is only defined for spaces :math:`\mathcal{F}(\Omega, \mathbb{F})` whose elements are functions over some domain :math:`\Omega \subset \mathbb{R}^d` taking values in :math:`\mathbb{R}` or :math:`\mathbb{C}`.
  It can be seen as a vectorized version of the usual gradient, taken in each point in :math:`\Omega`.

* **Subgradient**. The subgradient extends the notion of derivative to any convex functional and is used in some optimization solvers where the objective function is not differentiable.

Derivative
##########

The derivative is usually introduced for functions :math:`f: \mathbb{R} \rightarrow \mathbb{R}` via the limit

.. math::
    f'(x) = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h}.

Here we say that the derivative of :math:`f` in :math:`x` is :math:`f'(x)`.

This limit makes sense in one dimension, but once we start considering functions in higher dimension we get into trouble.
Consider :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}^m` -- what would :math:`h` mean in this case?
An extension is the concept of a directional derivative.
The derivative of :math:`f` in :math:`x` in *direction* :math:`d` is :math:`f'(x)(d)`:

.. math::
    f'(x)(d) = \lim_{h \rightarrow 0} \frac{f(x + dh) - f(x)}{h}.

Here we see (as implied by the notation) that :math:`f'(x)` is actually an operator

.. math::
    f'(x) : \mathbb{R}^n \rightarrow \mathbb{R}^m.

We can rewrite this using the explicit requirement that :math:`f'(x)` is a linear approximation of :math:`f` at :math:`x`, i.e.

.. math::
   \lim_{\| d \| \rightarrow 0} \frac{\| f(x + d) - f(x) - f'(x)(d) \|}{\| d \|} = 0

This notion naturally extends to an `Operator` :math:`f : \mathcal{X} \rightarrow \mathcal{Y}` between Banach spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}` with norms :math:`\| \cdot \|_\mathcal{X}` and :math:`\| \cdot \|_\mathcal{Y}`, respectively.
Here :math:`f'(x)` is defined as the linear operator (if it exists) that satisfies

.. math::
   \lim_{\| d \| \rightarrow 0} \frac{\| f(x + d) - f(x) - f'(x)(d) \|_\mathcal{Y}}{\| d \|_\mathcal{X}} = 0

This definition of the derivative is called the *Fréchet derivative*.

The Gateaux derivative
~~~~~~~~~~~~~~~~~~~~~~
The concept of directional derivative can also be extended to Banach spaces, giving the *Gateaux* derivative.
The Gateaux derivative is more general than the Fréchet derivative, but is not always a linear operator. An example of a function that is Gateaux differentiable but not Fréchet differentiable is the absolute value function.
For this reason, when we write "derivative" in ODL, we generally mean the Fréchet derivative, but in some cases the Gateaux derivative can be used via duck-typing.

Rules for the Fréchet derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many of the usual rules for derivatives also hold for the Fréchet derivative, i.e.

* Linearity

  .. math::
      (a f + b y)'(x)(y) = a f'(x)(y) + b g'(x)(y)

* Chain rule

  .. math::
      (g \circ f)'(x)(y) = g'(f(x))(f'(x)(y))

* Linear operators are their own derivatives. If :math:`f` linear, then

  .. math::
     f'(x)(y) = f(y)

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~

* The derivative is implemented in ODL  for `Operator`'s via the `Operator.derivative` method.
* It can be numerically computed using the `NumericalDerivative` operator.
* Many of the operator arithmetic classes implement the usual rules for the Fréchet derivative, such as the chain rule, distributivity over addition etc.

Gradient
########
In the classical setting of functionals :math:`f : \mathbb{R}^n \rightarrow \mathbb{R}`, the gradient is the vector

.. math::
    \nabla f =
    \begin{bmatrix}
        \dfrac{\partial f}{\partial x_1}
        \dots
        \dfrac{\partial f}{\partial x_n}
    \end{bmatrix}

This can be generalized to the setting of functionals :math:`f : \mathcal{X} \rightarrow \mathbb{R}` mapping elements in some Banach space :math:`\mathcal{X}` to the real numbers by noting that the Fréchet derivative can be written as

.. math::
    f'(x)(y) = \langle y, [\nabla f](x) \rangle,

where :math:`[\nabla f](x)` lies in the dual space of :math:`\mathcal{X}`, denoted :math:`\mathcal{X}^*`. For most spaces in ODL, the spaces are *Hilbert* spaces where :math:`\mathcal{X} = \mathcal{X}^*` by the `Riesz representation theorem
<https://en.wikipedia.org/wiki/Riesz_representation_theorem>`_ and hence :math:`f'(x) \in \mathcal{X}`.

We call the (possibly nonlinear) operator :math:`x \rightarrow [\nabla f](x)` the *Gradient operator* of :math:`f`.

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~

* The gradient is implemented in ODL `Functional`'s via the `Functional.gradient` method.
* It can be numerically computed using the `NumericalGradient` operator.

Hessian
#######
In the classical setting of functionals :math:`f : \mathbb{R}^n \rightarrow \mathbb{R}`, the Hessian in a point :math:`x` is the matrix :math:`H(x)` such that

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
    f(x + d) = f(x) + \langle d, [\nabla f](x)\rangle + \langle d, [H(x)](d)\rangle + o(\|d\|^2)

but also that the derivative of the gradient operator is

.. math::
    \nabla f(x + d) = [\nabla f](x) + [H(x)](d) + o(\|d\|)

If we take this second property as the *definition* of the Hessian, it can easily be generalized to the setting of functionals :math:`f : \mathcal{X} \rightarrow \mathbb{R}` mapping elements in some Hilbert space :math:`\mathcal{X}` to the real numbers.

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~

The Hessian is not explicitly implemented anywhere in ODL.
Instead it can be used in the form of the derivative of the gradient operator.
This is however not implemented for all functionals.

* For an example of a functional whose gradient has a derivative, see `RosenbrockFunctional`.
* It can be computed by taking the `NumericalDerivative` of the gradient, which can in turn be computed using the `NumericalGradient`.

Spatial Gradient
################

The spatial gradient of a function :math:`f \in \mathcal{F}(\Omega, \mathbb{R})` is an element in the function space :math:`\mathcal{F}(\Omega, \mathbb{R}^n)` such that for any :math:`x, d \in \Omega`.

.. math::
    \lim_{h \rightarrow 0} \frac{\| f(x + h d) - f(x) - \langle h d, \text{grad} f \rangle \|}{h} = 0

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~

* The spatial gradient is implemented in ODL in the `Gradient` operator.
* Several related operators such as the `PartialDerivative` and `Laplacian` are also available.

Subgradient
###########
The Subgradient (also *subderivative* or *subdifferential*) of a *convex* function :math:`f : \mathcal{X} \rightarrow \mathbb{R}`, mapping a Banach space :math:`\mathcal{X}` to :math:`\mathbb{R}`, is defined as the set-valued function :math:`\partial f : \mathcal{X} \rightarrow 2^{\mathcal{X}^*}` whose values are:

.. math::
   [\partial f](x_0) = \{c \in \mathcal{X}^* \ s.t. \ f(x) - f(x_0) \geq \langle c , x - x_0 \rangle \forall x \in \mathcal{X} \}

for functions that are differentiable in the usual sense, this reduces to the usual gradient.


Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~

The subgradient is not explicitly implemented in odl, but is implicitly used in the proximal operators.
See :ref:`proximal_operators` for more information.