.. _pdhg_math:

############################################
Primal-Dual Hybrid Gradient Algorithm (PDHG)
############################################

This page introduces the mathematics behind the Primal-Dual Hybrid Gradient Algorithm.
For an applied point of view, please see :ref:`the user's guide to this method <pdhg_guide>`.

The general problem
===================

The Primal-Dual Hybrid Gradient Algorithm (PDHG) algorithm, as studied in [CP2011a]_, is a first order method for non-smooth convex optimization problems with known saddle-point structure

.. math::
    \max_{y \in Y} \min_{x \in X} \big( \langle L x, y\rangle_Y + g(x) - f^*(y) \big) ,

where :math:`X` and :math:`Y` are Hilbert spaces with inner product :math:`\langle\cdot,\cdot\rangle` and norm :math:`\|.\|_2 = \langle\cdot,\cdot\rangle^{1/2}`, :math:`L` is a continuous linear operator :math:`L: X \to Y`, :math:`g: X \to [0,+\infty]` and :math:`f: Y \to [0,+\infty]` are proper, convex and lower semi-continuous functionals, and :math:`f^*` is the convex (or Fenchel) conjugate of f, (see :term:`convex conjugate`).

The saddle-point problem is a primal-dual formulation of the primal minimization problem

.. math::
    \min_{x \in X} \big( g(x) + f(L x) \big).

The corresponding dual maximization problem is

.. math::
    \max_{y \in Y} \big( g^*(-L^* x) - f^*(y) \big)

with :math:`L^*` being the adjoint of the operator :math:`L`.


The algorithm
=============

PDHG basically consists in alternating a gradient-like ascent in the dual variable :math:`y` and a gradient-like descent in the primal variable :math:`x`.
Additionally, an over-relaxation in the primal variable is performed.

Initialization
--------------
Choose :math:`\tau > 0`, :math:`\sigma > 0`, :math:`\theta \in [0,1]`,
:math:`x_0 \in X`, :math:`y_0 \in Y`, :math:`\bar x_0 = x_0`

Iteration
---------
For :math:`n > 0` update :math:`x_n`, :math:`y_n`, and :math:`\bar x_n` as
follows:

.. math::
    y_{n+1}         &= \text{prox}_{\sigma f^*}(y_n + \sigma L \bar x_n),

    x_{n+1}         &= \text{prox}_{\tau g}(x_n - \tau  L^* y_{n+1}),

    \bar x_{n+1}    &= x_{n+1} + \theta (x_{n+1} - x_n),

Here, :math:`\text{prox}` stands for :term:`proximal operator <proximal>`.

Step sizes
----------
A simple choice of step size parameters is :math:`\tau = \sigma < \frac{1}{\|L\|}`, since the requirement :math:`\sigma \tau \|L\|^2 < 1` guarantees convergence of the algorithm.
Of course, this does not imply that this choice is anywhere near optimal, but it can serve as a good starting point.

Acceleration
------------
If :math:`g` or :math:`f^*` is uniformly convex, convergence can be accelerated using variable step sizes as follows:

Replace :math:`\tau \to \tau_n`, :math:`\sigma \to \sigma_n`, and :math:`\theta \to \theta_n` and choose :math:`\tau_0 \sigma_0 \|L\|^2 < 1` and :math:`\gamma > 0`.
After the update of the primal variable :math:`x_{n+1}` and before the update of the relaxation variable :math:`\bar x_{n+1}` use the following update scheme for relaxation and step size parameters:

.. math::
    \theta_n        &= \frac{1}{\sqrt{1 + 2 \gamma \tau_n}},

    \tau_{n+1}      &= \theta_n \tau_n,

    \sigma_{n+1}    &= \frac{\sigma_n}{\theta_n}.

Instead of choosing step size parameters, preconditioning techniques can be employed, see [CP2011b]_.
In this case the steps :math:`\tau` and :math:`\sigma` are replaced by symmetric and positive definite matrices :math:`T` and :math:`\Sigma`, respectively, and convergence holds for :math:`\| \Sigma^{1/2}\,L\, T^{1/2}\|^2 < 1`.

For more on proximal operators and algorithms see [PB2014]_.
The implementation of PDHG in ODL is along the lines of [Sid+2012]_.
