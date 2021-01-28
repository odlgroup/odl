.. _linear_spaces:

#############
Linear Spaces
#############


Definition and basic properties
-------------------------------

A linear space over a `field`_ :math:`\mathbb{F}` is a set :math:`\mathcal{X}`, endorsed with the
operations of `vector addition`_ ":math:`+`" and `scalar multiplication`_ ":math:`\cdot`" which
are required to fullfill certain properties, usually called axioms. To emphasize the importance of
all ingredients, vector spaces are often written as tuples
:math:`(\mathcal{X}, \mathbb{F}, +, \cdot)`. We always assume that :math:`\mathbb{F} = \mathbb{R}` or
:math:`\mathbb{C}`.

In the following, we list the axioms, which are required to hold for arbitrary
:math:`x, y, z \in \mathcal{X}` and :math:`a, b \in \mathbb{F}`.

+--------------------------------+--------------------------------------------------------------+
|Associativity of addition       |:math:`(x + y) + z = (x + y) + z`                             |
+--------------------------------+--------------------------------------------------------------+
|Commutativity of addition       |:math:`x + y = y + x`                                         |
+--------------------------------+--------------------------------------------------------------+
|Existence of a neutral element  |:math:`0 + x = x + 0 = x`                                     |
|of addition                     |                                                              |
+--------------------------------+--------------------------------------------------------------+
|Existence of  inverse elements  |:math:`\forall x\ \exists \bar x: \bar x + x = x + \bar x = 0`|
|of addition                     |                                                              |
+--------------------------------+--------------------------------------------------------------+
|Compatibility of multiplications|:math:`a \cdot (b \cdot x) = (ab) \cdot x`                    |
+--------------------------------+--------------------------------------------------------------+
|Neutral scalar is the neutral   |:math:`1 \cdot x = x`                                         |
|element of scalar multiplication|                                                              |
+--------------------------------+--------------------------------------------------------------+
|Distributivity with respect to  |:math:`a \cdot (x + y) = a \cdot x + a \cdot y`               |
|vector addition                 |                                                              |
+--------------------------------+--------------------------------------------------------------+
|Distributivity with respect to  |:math:`(a + b) \cdot x = a \cdot x + b \cdot x`               |
|scalar addition                 |                                                              |
+--------------------------------+--------------------------------------------------------------+

Of course, the inverse element :math:`\bar x` is usually denoted with :math:`-x`.

Metric spaces
-------------
The vector space :math:`(\mathcal{X}, \mathbb{F}, +, \cdot)` is called a `metric space`_ if it is
additionally endorsed with a *distance* function or *metric*

.. math:: d: \mathcal{X} \times \mathcal{X} \to [0, \infty)

with the following properties for all :math:`x, y, z \in \mathcal{X}`:

.. math::
    :nowrap:

    \begin{align*}
      & d(x, y) = 0 \quad \Leftrightarrow \quad x = y && \text{(identity of indiscernibles)} \\
      & d(x, y) = d(y, x)  && \text{(symmetry)} \\
      & d(x, y) \leq d(x, z) + d(z, y) && \text{(subadditivity)}
    \end{align*}

We call the tuple :math:`(\mathcal{X}, \mathbb{F}, +, \cdot, d)` a `Metric space`_.

Normed spaces
-------------
A function on :math:`\mathcal{X}` intended to measure lengths of vectors is called a `norm`_

.. math:: \lVert \cdot \rVert : \mathcal{X} \to [0, \infty)

if it fulfills the following conditions for all :math:`x, y \in \mathcal{X}` and
:math:`a \in \mathbb{F}`:

.. math::
    :nowrap:

    \begin{align*}
      & \lVert x \rVert = 0 \Leftrightarrow x = 0 && \text{(positive definiteness)} \\
      & \lVert a \cdot x \rVert = \lvert a \rvert\, \lVert x \rVert && \text{(positive homegeneity)}
      \\
      & \lVert x + y \rVert \leq \lVert x \rVert + \lVert x \rVert && \text{(triangle inequality)}
    \end{align*}

A tuple :math:`(\mathcal{X}, \mathbb{F}, +, \cdot, \lVert \cdot \rVert)` fulfilling these conditions
is called `Normed vector space`_. Note that a norm induces a natural metric via
:math:`d(x, y) = \lVert x - y \rVert`.

Inner product spaces
--------------------
Measure angles and defining notions like orthogonality requires the existence of an `inner product`_

.. math:: \langle \cdot, \cdot \rangle : \mathcal{X} \times \mathcal{X} \to \mathbb{F}

with the following properties for all :math:`x, y, z \in \mathcal{X}` and :math:`a \in \mathbb{F}`:

.. math::
    :nowrap:

    \begin{align*}
      & \langle x, x \rangle \geq 0 \quad \text{and} \quad \langle x, x \rangle = 0 \Leftrightarrow
      x = 0 && \text{(positive definiteness)} \\
      & \langle a \cdot x + y, z \rangle = a \, \langle x, z \rangle + a \, \langle y, z \rangle &&
      \text{(linearity in the first argument)} \\
      & \langle x, y \rangle = \overline{\langle x, y \rangle} && \text{(conjugate symmetry)}
    \end{align*}

The tuple :math:`(\mathcal{X}, \mathbb{F}, +, \cdot, \langle \cdot \rangle)` is then called an
`Inner product space`_. Note that the inner product induces the norm
:math:`\lVert x \rVert = \sqrt{\langle x, x \rangle}`.


Cartesian spaces
----------------
We refer to the space :math:`\mathbb{F}^n` as the :math:`n`-dimensional `Cartesian space`_ over the
field :math:`\mathbb{F}`. We choose this notion since Euclidean spaces are usually associated with
the `Euclidean norm and distance`_, which are just (important) special cases. Vector addition and
scalar multiplication in :math:`\mathbb{F}^n` are, of course, realized with entry-wise addition
and scalar multiplication.

The natural inner product in :math:`\mathbb{F}^n` is defined as

.. math:: \langle x, y \rangle_{\mathbb{F}^n} := \sum_{i=1}^n x_i\, \overline{y_i}

and reduces to the well-known `dot product`_ if :math:`\mathbb{F} = \mathbb{R}`. For the norm, the
most common choices are from the family of `p-norms`_

.. math::
    \lVert x \rVert_p &:= \left( \sum_{i=1}^n \lvert x_i \rvert^p \right)^{\frac{1}{p}}
    \quad \text{if } p \in [1, \infty) \\[1ex]
    \lVert x \rVert_\infty &:= \max\big\{\lvert x_i \rvert\,|\, i \in \{1, \dots, n\} \big\}

with the standard Euclidan norm for :math:`p = 2`. As metric, one usually takes the norm-induced
distance function, although other choices are possible.

Weighted Cartesian spaces
-------------------------
In the standard definition of inner products, norms and distances, all components of a vector are
have the same weight. This can be changed by using weighted versions of those functions as described
in the following.

Let :math:`A \in \mathbb{F}^{n \times n}` be a `Hermitian`_ square and `positive definite`_ matrix,
in short :math:`A = A^* \succeq 0`. Then, a weighted inner product is defined by

.. math:: \langle x, y \rangle_A := \langle Ax, y \rangle_{\mathbb{F}^n}.

Weighted norms can be defined in different ways. For a general norm :math:`\lVert \cdot \rVert`,
a weighted version is given by

.. math:: \lVert x \rVert_A := \lVert Ax \rVert

For the :math:`p`-norms with :math:`p < \infty`, the definition is usually changed to

.. math:: \lVert x \rVert_{p, A} := \lVert A^{1/p} x \rVert,

where :math:`A^{1/p}` is the :math:`p`-th `root of the matrix`_ :math:`A`. The reason for this
definition is that for :math:`p = 2`, this version is consistent with the inner product
since :math:`\langle Ax, x \rangle = \langle A^{1/2} x, A^{1/2} x \rangle =
\lVert A^{1/2} x \rVert^2`.


Remark on matrices as operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A matrix :math:`M \in \mathbb{F}^{m \times n}` can be regarded as a `linear operator`_

.. math::
    \mathcal{M} &: \mathbb{F}^n \to \mathbb{F}^m \\
    \mathcal{M}(x) &:= M x

It is well known that in the standard case of a Euclidean space, the adjoint operator is simply
defined with the conjugate transposed matrix:

.. math::
    \mathcal{M}^* &: \mathbb{F}^m \to \mathbb{F}^n \\
    \mathcal{M}^*(y) &:= M^* y

However if the spaces :math:`\mathbb{F}^n` and :math:`\mathbb{F}^m` have weighted inner products,
this identification is no longer valid. If :math:`\mathbb{F}^{n \times n} \ni A = A^* \succeq 0`
and :math:`\mathbb{F}^{m \times m} \ni B = B^* \succeq 0` are the weighting matrices of the
inner products, we get

.. math::
    \langle \mathcal{M}(x), y \rangle_B
    &= \langle B\mathcal{M}(x), y \rangle_{\mathbb{F}^m}
    = \langle M x, B y \rangle_{\mathbb{F}^m}
    = \langle x, M^* B y \rangle_{\mathbb{F}^n} \\
    &= \langle A^{-1} A x, M^* B y \rangle_{\mathbb{F}^n}
    = \langle A x, A^{-1} M^* B y \rangle_{\mathbb{F}^n} \\
    &= \langle x, A^{-1} M^* B y \rangle_A

Thus, the adjoint of the matrix operator between the weighted spaces is rather given as
:math:`\mathcal{M}^*(y) = A^{-1} M^* B y`.

Useful Wikipedia articles
-------------------------

- `Vector space`_
- `Metric space`_
- `Normed vector space`_
- `Inner product space`_
- `Euclidean space`_

.. _Cartesian space: https://en.wikipedia.org/wiki/Cartesian_coordinate_system
.. _dot product: https://en.wikipedia.org/wiki/Dot_product
.. _Euclidean norm and distance: https://en.wikipedia.org/wiki/Euclidean_distance
.. _Euclidean space: https://en.wikipedia.org/wiki/Euclidean_space
.. _field: https://en.wikipedia.org/wiki/Field_%28mathematics%29
.. _Hermitian: https://en.wikipedia.org/wiki/Hermitian_matrix
.. _inner product: https://en.wikipedia.org/wiki/Inner_product_space
.. _Inner product space: https://en.wikipedia.org/wiki/Inner_product_space
.. _linear operator: https://en.wikipedia.org/wiki/Linear_map
.. _metric space: https://en.wikipedia.org/wiki/Metric_space
.. _Metric space: https://en.wikipedia.org/wiki/Metric_space
.. _norm: https://en.wikipedia.org/wiki/Normed_vector_space
.. _Normed vector space: https://en.wikipedia.org/wiki/Normed_vector_space
.. _p-norms: https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions
.. _positive definite: https://en.wikipedia.org/wiki/Positive-definite_matrix
.. _root of the matrix: https://en.wikipedia.org/wiki/Matrix_function
.. _scalar multiplication: https://en.wikipedia.org/wiki/Scalar_multiplication
.. _vector addition: https://en.wikipedia.org/wiki/Euclidean_vector#Addition_and_subtraction
.. _Vector space: https://en.wikipedia.org/wiki/Vector_space
