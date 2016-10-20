.. _derivatives_in_depth:

##############################################
A guide to the different notions of derivative
##############################################

The concept of a derivative is one of the core concepts of mathematical analysis and is used where we want a linear approximation of an arbitrary function. Since the derivative is used in different contexts to mean slightly different things, this guide has been written to clarify and introduce what the different concepts mean.

In short, different notions of derivatives that will be discussed here is:

* **Derivative**. When we write derivative in ODL code and documentation, we intend the derivative of a `operator` :math:`A : \mathcal{X} \rightarrow \mathcal{Y}` w.r.t to a disturbance in :math:`x`, i.e a linear approximation of :math:`A(x + h)` for small :math:`h`. The derivative in a point :math:`x` is a operator :math:`A'(x) : \mathcal{X} \rightarrow \mathcal{Y}`.

* **Gradient**. If the operator :math:`A` is a `functional`, i.e. :math:`A : \mathcal{X} \rightarrow \mathbb{R}` then the gradient is a measure of how much the output would change w.r.t. a small change in the input. The gradient in a point :math:`x` is a vector :math:`[\nabla A](x)` in :math:`\mathcal{X}` such that :math:`A'(x)(y) = \langle [\nabla A](x), y \rangle`. The gradient operator is the operator :math:`x \rightarrow [\nabla A](x)`.

* **Hessian**. The hessian in a point :math:`x` is the derivative operator of the gradient operator, i.e. :math:`H(x) = [\nabla A]'(x)`.

* **Spatial Gradient**. The spatial gradient is only defined for spaces :math:`\mathcal{F}(\Omega, \mathbb{R})` whose elements are functions over some domain :math:`\Omega \subset \mathbb{R}^n` with range in :math:`\mathbb{R}`. It can be seen as a vectorized version of the usual gradient, taken in each point in :math:`\Omega`.



Derivative
##########

It is usually introduced for functions :math:`f: \mathbb{R} \rightarrow \mathbb{R}` via the limit

.. math::
    f'(x) = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h}

Here we say that the derivative of :math:`f` in :math:`x` is :math:`f'(x)`.

This limit makes sense in one dimension, but once we start considering functions in higher dimension we get into trouble.
Consider :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}^m`, what would :math:`h` mean in this case?
A extension is the concept of a directional derivative. The derivative of :math:`f` in :math:`x` in *direction* :math:`h` is :math:`f'(x)(h)`.

.. math::
    f'(x)(h) = \lim_{\| h \| \rightarrow 0} \frac{f(x + h) - f(x)}{\| h \|}

Here we see (as implied by the notation) that :math:`f'(x)` is actually an operator

.. math::
    f'(x) : \mathbb{R}^n \rightarrow \mathbb{R}^m

However, we see that this needs not be linear, i.e. if :math:`f(x) = |x|` we get

.. math::
    f'(0)(h) =
    \begin{cases}
        1 &\text{if h>0} \\
        -1 &\text{else}
    \end{cases}

To remedy this, we require that :math:`f'(x)` is a linear approximation of :math:`f` at :math:`x`. I.e.

.. math::
   \lim_{\| h \| \rightarrow 0} \frac{\| f(x + h) - f(x) - f'(x)(h) \|}{\| h \|} = 0

This notion naturally extends to a operator :math:`A : \mathcal{X} \rightarrow \mathcal{Y}` between banach spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}` with norms :math:`\| \cdot \|_\mathcal{X}` and :math:`\| \cdot \|_\mathcal{Y}`, respectively. Here :math:`A'(x)` is defined as the linear operator, if it exists, that satisfies

.. math::
   \lim_{\| h \| \rightarrow 0} \frac{\| A(x + h) - A(x) - A'(x)(h) \|_\mathcal{Y}}{\| h \|_\mathcal{X}} = 0

This definition of the derivative is called the *Frechet derivative*. It is implemented in ODL in the `Operator.derivative` method. It can be numerically computed using the `NumericalDerivative` operator.

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
<https://en.wikipedia.org/wiki/Riesz_representation_theorem>`_, namely that the linear operator :math:`f'(x)` can be represented by a vector :math:`[\nabla f](x) \in \mathcal{X}` such that

.. math::
    f'(x)(y) = \langle y, [\nabla f](x) \rangle

We call the (possibly nonlinear) operator :math:`x \rightarrow [\nabla f](x)` element the *Gradient operator* of :math:`f`. It is implemented in ODL in the `Functional.gradient` method. It can be numerically computed using the `NumericalGradient` operator.

Hessian
#######
In the classical setting of functionals :math:`f : \mathbb{R}^n \rightarrow \mathbb{R}`, the hessian in a point :math:`x` is the matrix :math:`H` such that

.. math::
    H(x) =
    \begin{bmatrix}
    \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_n} \\
    \dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \dfrac{\partial^2 f}{\partial x_n\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
    \end{bmatrix}

Where the derivatives are taken in :math:`x`. This can be generalized to the setting of functionals :math:`f : \mathcal{X} \rightarrow \mathbb{R}` mapping elements in some Hilbert space :math:`\mathcal{X}` to the real numbers by noting that :math:`H` can be interpreted as a bi-linear operator:

.. math::
    H(x) : y, z \rightarrow y^T H x

such that the quadratic variation of :math:`f` is

.. math::
    \nabla f(x + \Delta x) \approx f(x) + \langle \Delta x, [\nabla f](x)\rangle + \langle \Delta x, [H(x)](\Delta x)\rangle

Spatial Gradient
################

Thus the spatial gradient of the element :math:`f \in \mathcal{F}(\Omega, \mathbb{R})` is a element in the function space :math:`\mathcal{F}(\Omega, \mathbb{R}^n)` such that for any :math:`x \in \Omega`.

.. math::
    \lim_{h \rightarrow 0} \frac{\| f(x + h \cdot d) - f(x) - \langle h, grad f \rangle \|}{h} = 0

It is implemented in ODL in the `Gradient` operator.
