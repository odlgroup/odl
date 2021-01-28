.. _resizing_ops:

##################
Resizing Operators
##################


Introduction
============
In ODL, resizing of a discretized function is understood as the operation of shrinking or enlarging its domain in such a way that the size of the partition cells do not change.
This "constant cell size" restriction is intentional since it ensures that the underlying operation can be implemented as array resizing without resampling, thus keeping those two functionalities separate (see `Resampling`).


Basic setting
=============
Let now :math:`\mathbb{R}^n` with :math:`n \in \mathbb{N}` be the space of one-dimensional real vectors encoding values of a function defined on an interval :math:`[a, b] \subset \mathbb{R}` (see :ref:`discretizations` for details).
Since values are not manipulated, the generalization to complex-valued functions is straightforward.


Restriction operator
====================
We consider the space :math:`\mathbb{R}^m` for an :math:`m < n \in \mathbb{N}` and define the restriction operator

.. math::
    R : \mathbb{R}^n \to \mathbb{R}^m, \quad R(x) := (x_p, \dots, x_{p+m-1})
    :label: def_restrict_op

with a given index :math:`0 \leq p \leq n - m - 1`.
Its adjoint with respect to the standard inner product is easy to determine:

.. math::
    \langle R(x), y \rangle_{\mathbb{R}^m}
    &= \sum_{j=0}^{m-1} R(x)_j\, y_j
    = \sum_{j=0}^{m-1} x_{p+j}\, y_j
    = \sum_{j=p}^{p+m-1} x_j\, y_{j-p} \\
    &= \sum_{i=0}^{n-1} x_i\, R^*(y)_i

with the zero-padding operator

.. math::
    R^*(y)_i :=
    \begin{cases}
    y_{i-p} & \text{if } p \leq i \leq p + m - 1, \\
    0       & \text{else.}
    \end{cases}
    :label: zero_pad_as_restr_adj

In practice, this means that a new zero vector of size :math:`n` is created, and the values :math:`y` are filled in from index :math:`p` onwards.
It is also clear that the operator :math:`R` is right-invertible by :math:`R^*`, i.e. :math:`R R^* = \mathrm{Id}_{\mathbb{R}^m}`.
In fact, any operator of the form :math:`R^* + P`, where :math:`P` is linear and :math:`P(x)_i = 0` for :math:`i \not \in \{p, \dots, p+m-1\}` acts as a right inverse for :math:`R`.
On the other hand, :math:`R` has no left inverse since it has a non-trivial kernel (null space) :math:`\mathrm{ker} R = \{x \in \mathbb{R}^n\,|\,x_i = 0 \text{ for } i = p, \dots, p+m-1\}`.


Extension operators
===================
Now we study the opposite case of resizing, namely extending a vector.
We thus choose :math:`m > n` and consider different cases of enlarging a given vector :math:`x \in \mathbb{R}^n` to a vector in :math:`\mathbb{R}^m`.
The start index is again denoted by :math:`p` and needs to fulfill :math:`0 \leq p \leq m - n - 1`, such that a vector of length :math:`n` "fits into" a vector of length :math:`m` when starting at index :math:`p`.

It should be noted that all extension operators mentioned here are of the above form :math:`R^* + P` with :math:`P` acting on the "outer" indices only.
Hence they all act as a right inverses for the restriction operator.
This property can also be read as the fact that all extension operators are left-inverted by the restriction operator :math:`R`.

Moreover, the "mixed" case, i.e. the combination of restriction and extension which would occur e.g. for a constant index shift :math:`x \mapsto (0, \dots, 0, x_0, \dots, x_{n-p-1})`, is not considered here.
It can be represented by a combination of the two "pure" operations.


Zero padding
------------
In this most basic padding variant, one fills the missing values in the target vector with zeros, yielding the operator

.. math::
    E_{\mathrm{z}} : \mathbb{R}^n \to \mathbb{R}^m, \quad E_{\mathrm{z}}(x)_j :=
    \begin{cases}
    x_{j-p}, & \text{if } p \leq j \leq p + n - 1, \\
    0      , & \text{else}.
    \end{cases}
    :label: def_zero_pad_op

Note that this is the adjoint of the restriction operator :math:`R` defined in :eq:`def_restrict_op`.
Hence, its adjoint is given by the restriction, :math:`E_{\mathrm{z}}^* = R`.


Constant padding
----------------
In constant padding with constant :math:`c`, the extra zeros in :eq:`def_zero_pad_op` are replaced with :math:`c`.
Hence, the operator performing constant padding can be written as :math:`E_{\mathrm{c}} = E_{\mathrm{z}} + P_{\mathrm{c}}`, where the second summand is given by

.. math::
    P_{\mathrm{c}}(x) =
    \begin{cases}
    0      , & \text{if } p \leq j \leq p + n - 1, \\
    c      , & \text{else}.
    \end{cases}

Note that this operator is not linear, and its derivative is the zero operator, hence the derivative of the constant padding operator is :math:`E_{\mathrm{c}}' = E_{\mathrm{z}}`.


Periodic padding
----------------
This padding mode continues the original vector :math:`x` periodically in both directions.
For reasons of practicability, at most one whole copy is allowed on both sides, which means that the numbers :math:`n`, :math:`m` and :math:`p` need to fulfill :math:`p \leq n` ("left" padding amount) and :math:`m - (p + n) \leq n` ("right" padding amount).
The periodic padding operator is then defined as

.. math::
    E_{\mathrm{p}}(x)_j :=
    \begin{cases}
    x_{j-p + n}, & \text{if } 0 \leq j \leq p - 1,     \\
    x_{j-p},     & \text{if } p \leq j \leq p + n - 1, \\
    x_{j-p - n}, & \text{if } p + n \leq j \leq m - 1.
    \end{cases}
    :label: def_per_pad_op

Hence, one can at most get 3 full periods with :math:`m = 3n` and :math:`p = n`.
Again, this operator can be written as :math:`E_{\mathrm{p}} = E_{\mathrm{z}} + P_{\mathrm{p}}` with an operator

.. math::
    P_{\mathrm{p}}(x)_j :=
    \begin{cases}
    x_{j-p + n}, & \text{if } 0 \leq j \leq p - 1,     \\
    0,           & \text{if } p \leq j \leq p + n - 1, \\
    x_{j-p - n}, & \text{if } p + n \leq j \leq m - 1.
    \end{cases}

For the adjoint of :math:`P_{\mathrm{p}}`, we calculate

.. math::
    \langle P_{\mathrm{p}}(x), y \rangle_{\mathbb{R}^m}
    &= \sum_{j=0}^{p-1} x_{j-p+n}\, y_j + \sum_{j=p+n}^{m-1} x_{j-p-n}\, y_j \\
    &= \sum_{i=n-p}^{n-1} x_i\, y_{i+p-n} + \sum_{i=0}^{m-n-p-1} x_i\, y_{i+p+n} \\
    &= \sum_{i=0}^{n-1} x_i\, \big( P_{\mathrm{p},1}^*(y) + P_{\mathrm{p},2}^*(y) \big)

with

.. math::
    P_{\mathrm{p},1}^*(y)_i :=
    \begin{cases}
    y_{i+p-n}, & \text{if } n - p \leq i \leq n - 1, \\
    0,         & \text{else},
    \end{cases}

and

.. math::
    P_{\mathrm{p},2}^*(y)_i :=
    \begin{cases}
    y_{i+p+n}, & \text{if } 0 \leq i \leq m - n - p - 1, \\
    0,         & \text{else}.
    \end{cases}

In practice, this means that that besides copying the values from the indices :math:`p, \dots, p+n-1` of a vector :math:`y \in \mathbb{R}^m` to a new vector :math:`x \in \mathbb{R}^n`, the values corresponding to the other indices are added to the vector :math:`x` as follows.
The *first* :math:`m - n - p - 1` entries of :math:`y` (negative means 0) are added to the *last* :math:`m - n - p - 1` entries of :math:`x`, in the same ascending order.
The *last* :math:`p` entries of :math:`y` are added to the *first* :math:`p` entries of :math:`x`, again keeping the order.
This procedure can be interpreted as "folding back" the periodized structure of :math:`y` into a single period :math:`x` by adding the values from the two side periods.


Symmetric padding
-----------------
In symmetric padding mode, a given vector is extended by mirroring at the outmost nodes to the desired extent.
By convention, the outmost values are not repeated, and as in periodic mode, the input vector is re-used at most once on both sides.
Since the outmost values are not doubled, the numbers :math:`n`, :math:`m` and :math:`p` need to fulfill the relations :math:`p \leq n - 1` ("left" padding amount) and :math:`m - (p + n) \leq n - 1` ("right" padding amount).
Now the symmetric padding operator is defined as

.. math::
    E_{\mathrm{s}}(x)_j :=
    \begin{cases}
    x_{p-j},      & \text{if } 0 \leq j \leq p - 1,      \\
    x_{j-p},      & \text{if } p \leq j \leq p + n - 1,  \\
    x_{2n-2+p-j}, & \text{if } p + n \leq j \leq m - 1.
    \end{cases}
    :label: def_sym_pad_op

This operator is the sum of the zero-padding operator :math:`E_{\mathrm{z}}` and

.. math::
    P_{\mathrm{s}}(x)_j :=
    \begin{cases}
    x_{p-j},      & \text{if } 0 \leq j \leq p - 1,      \\
    0,            & \text{if } p \leq j \leq p + n - 1,  \\
    x_{2n-2+p-j}, & \text{if } p + n \leq j \leq m - 1.
    \end{cases}

For its adjoint, we compute

.. math::
    \langle P_{\mathrm{s}}(x), y \rangle_{\mathbb{R}^m}
    &= \sum_{j=0}^{p-1} x_{p-j}\, y_j + \sum_{j=p+n}^{m-1} x_{2n-2+p-j}\, y_j \\
    &= \sum_{i=1}^p x_i\, y_{p-i} + \sum_{i=2n-1+p-m}^{n-2} x_i\, y_{2n-2+p-i} \\
    &= \sum_{i=0}^{n-1} x_i\, \big( P_{\mathrm{s},1}^*(y) + P_{\mathrm{s},2}^*(y) \big)

with

.. math::
    P_{\mathrm{s},1}^*(y)_i :=
    \begin{cases}
    y_{p-i},   & \text{if } 1 \leq i \leq p, \\
    0,         & \text{else},
    \end{cases}

and

.. math::
    P_{\mathrm{s},2}^*(y)_i :=
    \begin{cases}
    y_{2n-2+p-i}, & \text{if } 2n - 1 + p - m \leq i \leq n - 2, \\
    0,            & \text{else}.
    \end{cases}

Note that the index condition :math:`m - (p + n) \leq n - 1` is equivalent to :math:`2n - 1 + p - m \geq 0`, hence the index range in the definition of :math:`P_{\mathrm{s},2}^*` is well-defined.

Practically, the evaluation of :math:`E_{\mathrm{s}}^*` consists in copying the "main" part of :math:`y \in \mathbb{R}^m` corresponding to the indices :math:`p, \dots, p + n - 1` to :math:`x \in \mathbb{R}^n` and updating the vector additively as follows.
The values at indices 1 to :math:`p` are updated with the values of :math:`y` mirrored at the index position :math:`p`, i.e. in reversed order.
The values at the indices :math:`2n - 1 + p - m` to :math:`n - 2` are updated with the values of :math:`y` mirrored at the position :math:`2n + 2 - p`, again in reversed order.
This procedure can be interpreted as "mirroring back" the outer two parts of the vector :math:`y` at the indices :math:`p` and :math:`2n + 2 - p`, adding those parts to the "main" vector.


Order 0 padding
---------------
Padding with order 0 consistency means continuing the vector constantly beyond its boundaries, i.e.

.. math::
    E_{\mathrm{o0}}(x)_j :=
    \begin{cases}
    x_0,     & \text{if } 0 \leq j \leq p - 1,      \\
    x_{j-p}, & \text{if } p \leq j \leq p + n - 1,  \\
    x_{n-1}, & \text{if } p + n \leq j \leq m - 1.
    \end{cases}
    :label: def_order0_pad_op

This operator is the sum of the zero-padding operator and

.. math::
    P_{\mathrm{o0}}(x)_j :=
    \begin{cases}
    x_0,     & \text{if } 0 \leq j \leq p - 1,      \\
    0,       & \text{if } p \leq j \leq p + n - 1,  \\
    x_{n-1}, & \text{if } p + n \leq j \leq m - 1.
    \end{cases}

We calculate the adjoint of :math:`P_{\mathrm{o0}}`:

.. math::
    \langle P_{\mathrm{o0}}(x), y \rangle_{\mathbb{R}^m}
    &= \sum_{j=0}^{p-1} x_0\, y_j + \sum_{j=p+n}^{m-1} x_{n-1}\, y_j \\
    &= x_0 \sum_{j=0}^{p-1} y_j + x_{n-1} \sum_{j=p+n}^{m-1} y_j \\
    &= x_0 M_{\mathrm{l},0}(y) + x_{n-1} M_{\mathrm{r},0}(y)

with the zero'th order moments

.. math::
    M_{\mathrm{l},0}(y) := \sum_{j=0}^{p-1} y_j, \quad M_{\mathrm{r},0}(y) := \sum_{j=p+n}^{m-1} y_j.

Hence, we get

.. math::
    P_{\mathrm{o0}}^*(y)_i :=
    \begin{cases}
    M_{\mathrm{l},0}(y), & \text{if } i = 0,     \\
    M_{\mathrm{r},0}(y), & \text{if } i = n - 1, \\
    0,                   & \text{else},
    \end{cases}

with the convention that the sum of the two values is taken in the case that $n = 1$, i.e. both first cases are the same.
Hence, after constructing the restriction :math:`x \in \mathbb{R}^n` of a vector :math:`y \in \mathbb{R}^m` to the main part :math:`p, \dots, p + n - 1`, the sum of the entries to the left are added to :math:`x_0`, and the sum of the entries to the right are added to :math:`x_{n-1}`.


Order 1 padding
---------------
In this padding mode, a given vector is continued with constant slope instead of constant value, i.e.

.. math::
 E_{\mathrm{o1}}(x)_j :=
 \begin{cases}
  x_0 + (j - p)(x_1 - x_0),                     & \text{if } 0 \leq j \leq p - 1,      \\
  x_{j-p},                                      & \text{if } p \leq j \leq p + n - 1,  \\
  x_{n-1} + (j - p - n + 1)(x_{n-1} - x_{n-2}), & \text{if } p + n \leq j \leq m - 1.
 \end{cases}
 :label: def_order1_pad_op

We can write this operator as :math:`E_{\mathrm{o1}} = E_{\mathrm{o0}} + S_{\mathrm{o1}}` with the order-1 specific part

.. math::
    S_{\mathrm{o1}}(x)_j :=
    \begin{cases}
    (j - p)(x_1 - x_0),                 & \text{if } 0 \leq j \leq p - 1,      \\
    0,                                  & \text{if } p \leq j \leq p + n - 1,  \\
    (j - p - n + 1)(x_{n-1} - x_{n-2}), & \text{if } p + n \leq j \leq m - 1.
    \end{cases}

For its adjoint, we get

.. math::
 \langle S_{\mathrm{o1}}(x), y \rangle_{\mathbb{R}^m}
 &= \sum_{j=0}^{p-1} (j - p)(x_1 - x_0)\, y_j +
    \sum_{j=p+n}^{m-1} (j - p - n + 1)(x_{n-1} - x_{n-2})\, y_j \\
 &= x_0 (-M_{\mathrm{l}}(y)) + x_1 M_{\mathrm{l}}(y) +
    x_{n-2}(-M_{\mathrm{r}}(y)) + x_{n-1} M_{\mathrm{r}}(y)

with the first order moments

.. math::
 M_{\mathrm{l},1}(y) := \sum_{j=0}^{p-1} (j - p)\, y_j, \quad
 M_{\mathrm{r},1}(y) := \sum_{j=p+n}^{m-1} (j - p - n + 1)\, y_j.

Hence, the order-1 specific operator has the adjoint

.. math::
    S_{\mathrm{o1}}^*(y)_i :=
    \begin{cases}
    -M_{\mathrm{l},1}(y), & \text{if } i = 0,     \\
    M_{\mathrm{l},1}(y),  & \text{if } i = 1,     \\
    -M_{\mathrm{r},1}(y), & \text{if } i = n - 2, \\
    M_{\mathrm{r},1}(y),  & \text{if } i = n - 1, \\
    0,                  & \text{else},
    \end{cases}

with the convention of summing values for overlapping cases, i.e. if :math:`i \in \{1, 2\}`.
In practice, the adjoint for the order 1 padding case is applied by computing the zero'th and first order moments of :math:`y` and adding them to the two outmost entries of :math:`x` according to the above rule.


Generalization to arbitrary dimension
=====================================
Fortunately, all operations are completely separable with respect to (coordinate) axes, i.e. resizing in higher-dimensional spaces can be written as a series of one-dimensional resizing operations.
One particular issue should be mentioned with the extension operators and their adjoints, though.
When extending a small, e.g., two-dimensional array to a larger size, there is an ambiguity in how the corner blocks should be handled.
One possibility would be use the small array size for the extension in both axes, which would leave the corner blocks untouched (initialized to 0 usually):

.. image:: images/resize_small.svg
   :width: 100%

However, this is not the behavior one would often want in practice.
Instead, it is much more reasonable to also fill the corners in the same way the "inner" parts have been extended:

.. image:: images/resize_large.svg
   :width: 100%

This latter behavior is implemented in the resizing operators in ODL.

The adjoint operators of these "corner-filling" resizing operator are given by reversing the unfolding pattern, i.e. by "folding in" the large array axis by axis according to the adjoint formula for the given padding mode.
This way, the corners also contribute to the final result, which leads to the correct adjoint of the 2D resizing operator.
Of course, the same principle can easily be generalized to arbitrary dimension.
