###############
Getting started
###############

Welcome to the ODL users guide, this guide is intended to give you a simple introduction to ODL and how to work with it. If you need help on a specific function you should look at its documentation.

Operators
---------
Operators in ODL are represented by the abstract `Operator`
class. As an *abstract class*, it cannot be used directly but must be
subclassed for concrete implementation. To define your own operator,
you start by writing::

    class MyOperator(odl.Operator):
        ...

`Operator` has a couple of *abstract methods* which need to
be explicitly overridden by any subclass, namely

``domain``: `Set`
    Set of elements to which the operator can be applied
``range`` `Set`
    Set in which the operator takes values

As a simple example, you can implement the matrix multiplication
operator

    :math:`\mathcal{A}: \mathbb{R}^m \\to \mathbb{R}^n, \quad
    \mathcal{A}(x) = Ax`

for a matrix :math:`A\\in \mathbb{R}^{n\\times m}` as follows::

    from builtins import super
    import numpy as np

    class MatVecOperator(odl.Operator):
        def __init__(self, matrix):
            assert isinstance(matrix, np.ndarray)
            self.matrix = matrix
            dom = odl.Rn(matrix.shape[1])
            ran = odl.Rn(matrix.shape[0])
            super().__init__(dom, ran)

In addition, an `Operator` needs at least one way of
evaluation, *in-place* or *out-of-place*.

- In-place evaluation means that the operator is evaluated on a
  ``domain`` element, and the result is written to an
  *already existing* ``range`` element. To implement
  this behavior, create the (private) ``Operator._apply``
  method with the following signature, here given for the above
  example::

    class MatVecOperator(odl.Operator):
        ...
        def _apply(self, x, out):
            self.matrix.dot(x, out=out.asarray())

  In-place evaluation is usually more efficient and should be used
  *whenever possible*.

- Out-of-place evaluation means that the
  operator is evaluated on a ``domain`` element, and
  the result is written to a *newly allocated*
  ``range`` element. To implement this
  behavior, create the (private) ``Operator._call`` method
  with the following signature, here given for the above example::

    class MatVecOperator(odl.Operator):
        ...
        def _call(self, x):
            return self.range.element(self.matrix.dot(x))

  Out-of-place evaluation is usually less efficient since it requires
  allocation of an array and a full copy and should be *generally
  avoided*.

**Important:** Do not call these methods directly. Use the call pattern
``operator(x)`` or ``operator(x, out=y)``, e.g.::

    matrix = np.array([[1, 0], [0, 1], [1, 1]])
    operator = MatVecOperator(matrix)
    x = odl.Rn(2).one()
    y = odl.Rn(3).element()

    # Out-of-place evaluation
    y = operator(x)

    # In-place evaluation
    operator(x, out=y)

This public calling interface is type-checked, so the private methods
do not need to implement type checks.