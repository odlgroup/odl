.. _derivatives_in_depth:

##############################################
A guide to the different notions of derivative
##############################################

The concept of a derivative is one of the core concepts of mathematical analysis and is used where we want a linear approximation of an arbitrary function. Since the derivative is used in different contexts to mean slightly different things, this guide has been written to clarify and introduce what the different concepts mean.

In short, different notions of derivatives that will be discussed here are:

* **Derivative**. When we write derivative in ODL code and documentation, we intend the derivative of an `Operator` :math:`A : \mathcal{X} \rightarrow \mathcal{Y}` w.r.t to a disturbance in :math:`x`, i.e a linear approximation of :math:`A(x + h)` for small :math:`h`. The derivative in a point :math:`x` is an `Operator` :math:`A'(x) : \mathcal{X} \rightarrow \mathcal{Y}`.

* **Gradient**. If the operator :math:`A` is a `functional`, i.e. :math:`A : \mathcal{X} \rightarrow \mathbb{R}`, then the gradient is the direction in which :math:`A` increases the most. The gradient in a point :math:`x` is a vector :math:`[\nabla A](x)` in :math:`\mathcal{X}` such that :math:`A'(x)(y) = \langle [\nabla A](x), y \rangle`. The gradient operator is the operator :math:`x \rightarrow [\nabla A](x)`.

* **Hessian**. The hessian in a point :math:`x` is the derivative operator of the gradient operator, i.e. :math:`H(x) = [\nabla A]'(x)`.

* **Spatial Gradient**. The spatial gradient is only defined for spaces :math:`\mathcal{F}(\Omega, \mathbb{F})` whose elements are functions over some domain :math:`\Omega \subset \mathbb{R}^d` taking values in :math:`\mathbb{R}` or :math:`\mathbb{C}`. It can be seen as a vectorized version of the usual gradient, taken in each point in :math:`\Omega`.

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
        , \dots,
        \dfrac{\partial f}{\partial x_n}
    \end{bmatrix}

This can be generalized to the setting of functionals :math:`f : \mathcal{X} \rightarrow \mathbb{R}` mapping elements in some Hilbert space :math:`\mathcal{X}` to the real numbers by noting that the derivative has a special form by the `Riesz representation theorem
<https://en.wikipedia.org/wiki/Riesz_representation_theorem>`_, namely that the linear operator :math:`f'(x)` can be represented by a vector :math:`[\nabla f](x) \in \mathcal{X}` such that for all :math:`y \in \mathcal{X}`:

.. math::
    f'(x)(y) = \langle y, [\nabla f](x) \rangle.

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

with the derivatives are evaluated in the point :math:`x`. It has the property that that the quadratic variation of :math:`f` is

.. math::
    f(x + d) = f(x) + \langle d, [\nabla f](x)\rangle + \langle d, [H(x)](d)\rangle + o(\|d\|^2)

but also that the derivative of the gradient operator is

.. math::
    \nabla f(x + d) = [\nabla f](x) + [H(x)](d) + o(\|d\|)

If we take this second property as the *definition* of the Hessian, it can easily be generalized to the setting of functionals :math:`f : \mathcal{X} \rightarrow \mathbb{R}` mapping elements in some Hilbert space :math:`\mathcal{X}` to the real numbers.

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~

The hessian is not explicitly implemented anywhere in ODL.
Instead it is taken as the derivative of the gradient operator.
This is however not implemented for all functionals.

* For an example of a functional whose gradient has a derivative, see `RosenbrockFunctional`.
* It can be computed by taking the `NumericalDerivative` of the gradient, which can in turn be computed using the `NumericalGradient`.

Spatial Gradient
################

Thus the spatial gradient of the function :math:`f`, which is an element in some function space :math:`f \in \mathcal{F}(\Omega, \mathbb{R})`, is a element in the function space :math:`\mathcal{F}(\Omega, \mathbb{R}^n)` such that for any :math:`x, d \in \Omega`.

.. math::
    \lim_{h \rightarrow 0} \frac{\| f(x + h d) - f(x) - \langle h d, grad f \rangle \|}{h} = 0

Implementations in ODL
~~~~~~~~~~~~~~~~~~~~~~

* The spatial gradient is implemented in ODL in the `Gradient` operator.
* Several related operators such as the `PartialDerivative` and `Laplacian` are also available.
