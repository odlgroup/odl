.. _chambolle_pock:

########################
Chambolle-Pock algorithm
########################

The general problem
===================

The Chambolle-Pock (CP) algorithm, as proposed in [CP2011a]_, is a first
order primal-dual hybrid-gradient method for non-smooth convex optimization
problems with known saddle-point structure

.. math::
    \max_{y \in Y} \min_{x \in X}\;\langle K x, y\rangle_Y + G(x) - F^*(y) ,

where :math:`X` and :math:`Y` are finite-dimensional Hilbert spaces with inner product
:math:`\langle\cdot,\cdot\rangle` and norm :math:`\|.\|_2 = \langle\cdot,\cdot\rangle^{1/2}`,
:math:`K` is a continuous linear operator :math:`K:X\mapsto Y`.
:math:`G:X\mapsto[0,+\infty]` and :math:`F^*:Y\mapsto[0,+\infty]`
are proper, convex, lower-semicontinuous functionals, and :math:`F^*` is the
convex (or Fenchel) conjugate of F, see below.

The saddle-point problem is a primal-dual formulation of the following
primal minimization problem

.. math:: \min_{x \in X}\;G(x) + F(K x)\;.

The corresponding dual maximization problem is

.. math::
    \max_{y \in Y}\;G(-K^* x) - F^*(y)

with :math:`K^*` being the adjoint of the operator :math:`K`.

The convex conjugate is a mapping from a normed vector space :math:`X` to its
dual space :math:`X^*`. It is defined as

.. math::
    F^*(x^*) = \sup_{x\in X}\; \langle x^*,x\rangle - F(x)\;,

with :math:`x^*\in X^*` and :math:`\langle\cdot,\cdot\rangle` denotes the dual
pairing. For Hilbert spaces, which are self-dual, we have :math:`X=X^*` and
:math:`\langle\cdot,\cdot\rangle` is the inner product. The convex conjugate
is always convex, and if F is convex, proper, and lower semi-continuous we
have :math:`F=(F^*)^*` . For more details see [Roc1970]_.


The algorithm
=============

The CP algorithm basically consists of alternating a gradient ascend in
the dual variable :math:`y` and a gradient descent in the primal variable
:math:`x`. Additionally an over-relaxation in the primal variable is performed.

**Initialization**

Choose :math:`\tau > 0`, :math:`\sigma > 0`, :math:`\theta \in [0,1]`,
:math:`x_0 \in X`, :math:`y_0 \in Y`, :math:`\bar x_0 = x_0`

**Iteration**

For :math:`n > 0` update :math:`x_n`, :math:`y_n`, and :math:`\bar x_n` as
follows

.. math:: y_{n+1} = prox_\sigma[F^*](y_n + \sigma K \bar x_n)

    x_{n+1} = prox_\tau[G](x_n - \tau  K^* y_{n+1})

    \bar x_{n+1} = x_{n+1} + \theta (x_{n+1} - x_n)


**Proximal operator**

The proximal operator, :math:`prox_\tau[H](x)`, of the functional :math:`H` with step size parameter
:math:`tau` is defined as

.. math::
    prox_\tau[H](x) = \arg\;\min_{y\in Y}\; H(y) + \frac{1}{2 \tau} \|x - y\|_2^2

**Step sizes**

A simple choice of step size parameters is :math:`\tau=\sigma<
\frac{1}{\|K\|}` with the induced operator norm

.. math:: \|K\| = \max_{x\in X}\;\{\|K x\|:\|x\| < 1\}

For :math:`\|K\|^2\sigma\tau < 1` converge of the algorithm should be
guaranteed.

**Acceleration**

If :math:`G` or :math:`F^*` is uniformly convex, convergence can be
accelerated using variable step sizes.

Replace :math:`\tau\rightarrow\tau_n`, :math:`\sigma\rightarrow\sigma_n`,
and :math:`\theta\rightarrow\theta_n` and choose
:math:`\tau_0\sigma_0\|K\|^2 < 1` and :math:`\gamma>0` . After the update of
the primal variable :math:`x_{n+1}` and before the update of the relaxation
variable :math:`\bar x_{n+1}` use the following update scheme for relaxation
and step size parameters as

.. math:: \theta_n = 1 / \sqrt{1 + 2 \gamma \tau_n}

    \tau_{n+1} = \theta_n \tau_n

    \sigma_{n+1} = \sigma_n / \theta_n

Instead of choosing step size parameters preconditioning techniques can
be employed, see [CP2011b]_. In this case the steps tau and sigma are
replaced by symmetric and positive definite matrices
:math:`\tau\rightarrow T`, :math:`\sigma\rightarrow\Sigma` and convergence
should hold for :math:`\| \Sigma^{1/2}\,K\, T^{1/2}\|^2 < 1`.

For more on proximal operators and algorithms see [PB2014]_. The
following implementation of the CP algorithm is along the lines of
[Sid+2012]_.

