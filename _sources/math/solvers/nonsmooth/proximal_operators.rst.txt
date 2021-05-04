.. _proximal_operators:

##################
Proximal Operators
##################

Definition
----------

Let :math:`f` be a proper convex function mapping the normed space :math:`X`
to the extended real number line :math:`(-\infty, +\infty]`. The proximal
operators of the functional :math:`f` is mapping from :math:`X\mapsto X`. It
is denoted as :math:`\mathrm{prox}_\tau[f](x)` with :math:`x\in X`  and defined by

.. math::
    \mathrm{prox}_\tau[f](x) = \arg\;\min_{y\in Y}\;f(y)+\frac{1}{2\tau} \|x-y\|_2^2

The shorter notation :math:`\mathrm{prox}_{\tau\,f}(x)`) is also common.

Properties
----------

Some properties which are useful to create or compose proximal operators:

**Separable sum**

If :math:`f` is separable across variables, i.e. :math:`f(x,y)=g(x)+h(y)`,
then

.. math:: \mathrm{prox}_\tau[f](x, y) = (\mathrm{prox}_\tau[g](x), \mathrm{prox}_\tau[h](y))

**Post-composition**

If :math:`g(x)=\alpha f(x)+a` with :math:`\alpha > 0`, then

.. math:: \mathrm{prox}_\tau[g](x) = \mathrm{prox}_{\alpha\tau}[f](x)

**Pre-composition**

If :math:`g(x)=f(\beta x+b)` with :math:`\beta\ne 0`, then

.. math::
    \mathrm{prox}_\tau[g](x) = \frac{1}{\beta} (\mathrm{prox}_{\beta^2\tau}[f](\beta x+b)-b)

**Moreau decomposition**

This is also know as the Moreau identity

.. math::
    x = \mathrm{prox}_\tau[f](x) + \frac{1}{\tau}\,\mathrm{prox}_{1/\tau}[f^*] (\frac{x}{\tau})

where :math:`f^*` is the convex conjugate of :math:`f`.

**Convec conjugate**

The convex conjugate of :math:`f` is defined as

.. math:: f^*(y) = \sup_{x\in X} \langle y,x\rangle - f(x)

where :math:`\langle\cdot,\cdot\rangle` denotes inner product. For more
on convex conjugate and convex analysis see [Roc1970]_
or `Wikipedia <https://en.wikipedia.org/wiki/Convex_conjugate>`_.

For more details on proximal operators including how to evaluate the
proximal operator of a variety of functions see [PB2014]_.


Indicator function
------------------

Indicator functions are typically used to incorporate constraints. The
indicator function for a given set :math:`S` is defined as

.. math::
    \mathrm{ind}_{S}(x) =\begin{cases}
    0 & x \in S  \\ \infty &
    x\ \notin S
    \end{cases}

**Special indicator functions**

Indicator for a box centered at origin and with width :math:`2 a`:

.. math::
    \mathrm{ind}_{\mathrm{box}(a)}(x) = \begin{cases}
    0 & \|x\|_\infty \le a\\
    \infty & \|x\|_\infty > a
    \end{cases}

where :math:`\|\cdot\|_\infty` denotes the maximum-norm.
