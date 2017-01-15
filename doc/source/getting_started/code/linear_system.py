
# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.


# ## Problem 1: Linear system
#
# We consider the linear system
#
# $$
# A x = b,\quad A \in \mathbb{R}^{m \times n}
# $$
#
# for given $b\in\mathbb{R}^m$ and want to compute a least squares
# solution, i.e.
#
# $$
# x^\ast = \mathrm{arg}\min_{x\in\mathbb{R}^n} \frac{1}{2} \lVert A x - b
# \rVert_2^2.
# $$
#
# We choose a ridiculously small problem ($n = 3$, $m = 2$) for
# demonstration purposes.

import odl
import numpy as np  # Package for n-dimensional array handling


# ### Definition of the spaces

# $\mathscr{X} := \mathbb{R}^3$, $\mathscr{Y} := \mathbb{R}^3$

X = odl.Rn(3)
Y = odl.Rn(2)


# We can create elements of the spaces from given lists / arrays or with
# special methods:

x1 = X.element([1, 2, 4])
array = np.linspace(-1, 1, 3)
x2 = X.element(array)  # [-1, 0, 1]
x3 = X.one()  # [1, 1, 1]


# These vectors have access to all linear space functionality, e.g.
# arithmetic

x1 + x2 - 1   # (1, 2, 4) + (-1, 0, 1) - 1


x2 / x1  # (-1, 0, 1) / (1, 2, 4)   element-wise


# By default, we have the standard inner product
# $$
# \langle x, y \rangle = \sum_{j=1}^n x_j\, \overline{y_j}
# $$
# and its induced norm:

x1.inner(x3)  # <(1, 2, 4), (1, 1, 1)>


x2.norm()  # sqrt(1^2 + 0^2 + 1^2)


# That can be changed, of course:

Z = odl.Rn(3, exponent=1)  # 1-norm
z = Z.element([1, 2, 3])
z.norm()


# ### Setup of data and system matrix

# We use
# $$
# A =
# \begin{pmatrix}
#   1 & 2 & 0 \\
#   -1 & 0 & 1
# \end{pmatrix},
# \qquad
# b =
# \begin{pmatrix}
#   1\\
#   -1
# \end{pmatrix}.
# $$
#

b = Y.element([1, -1])


A = np.array([[1.0, 2.0, 0.0],
              [-1.0, 0.0, 1.0]])


# ### Make an operator from the matrix

# **Mathematical definition:**
#
# $$
# \mathcal{A}: \mathscr{X} \to \mathscr{Y} \\
# \mathcal{A}(x) := Ax
# $$
#
# The `Operator` class is the universal interface to solvers. For
# matrices, we can use the predefined `MatVecOperator`:

# dom and ran can be omitted - then domain and range are inferred
A_op = odl.MatVecOperator(A, dom=X, ran=Y)


# To evaluate the operator, one simply calls it like a function:

x = X.element([1, 1, 1])
A_op(x)


# `A_op` is a bounded linear operator between Hilbert spaces, so it has an
# adjoint, given by the transposed matrix:
#
# $$
# \mathcal{A}^\ast: \mathscr{Y} \to \mathscr{X} \\
# \mathcal{A}^\ast(y) = A^{\mathrm{T}} y \\
# A^{\mathrm{T}} =
# \begin{pmatrix}
#   1 & -1 \\
#   2 & 0 \\
#   0 & 1
# \end{pmatrix}
# $$
#

A_op.adjoint


A_op.adjoint.matrix


# It can be evaluated on any element of $\mathscr{Y}$ - in fact, one can
# check if the adjoint is correct, in the sense that $\langle
# \mathcal{A}(x), b \rangle_{\mathscr{Y}} = \langle x, \mathcal{A}^* (b)
# \rangle_{\mathscr{X}}$:

A_op(x).inner(b) == x.inner(A_op.adjoint(b))


# Operators can be added, multiplied by scalars (left and right), composed
# etc.:

A_op_times_two = A_op * 2
A_op_times_two(x)


B = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]])
B_op = odl.MatVecOperator(B)

sum_op = A_op + B_op
sum_op(x)


# More complex example: $\mathcal{T} = \mathcal{A}^\ast \mathcal{A} + 2
# \mathcal{I}_{\mathscr{X}}$

# Tikhonov-type operator
T_op = (A_op.adjoint * A_op + 2 * odl.IdentityOperator(X))
T_op(x)


# ### Calling the solver

x = X.zero()  # Start value
odl.solvers.conjugate_gradient_normal(A_op, x, b, niter=3)


x


# Checking the result - we expect $\mathcal{A}^*(\mathcal{A}(x^*) - b) =
# 0$:

A_op.adjoint(A_op(x) - b)
