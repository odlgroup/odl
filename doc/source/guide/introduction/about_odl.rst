#########
About ODL
#########

ODL is a Python library for inverse problems. It contains all the abstract mathematical tools needed from analysis, such as sets, vector spaces and operators as well as optimized implementations of many of the most common of these.


Set
===

A `Set` is the fundamental building block of odl objects. It is intended to mirror the mathematical concept of a `set
<https://en.wikipedia.org/wiki/Set_(mathematics)>`_, and has methods to check if an element is contained in the set

>>> interv = Interval(0, 1)
>>> 0.5 in interv
True
>>> 2.0 in interv
False

The most commonly used sets in odl is the `RealNumbers` the set of all `real numbers
<https://en.wikipedia.org/wiki/Real_number>`_ and `IntervalProd`, arbitrary `n-dimensional hypercubes
<https://en.wikipedia.org/wiki/Hypercube>`_. Several convenience sub-classes such as `Interval`,  `Rectangle` and `Cuboid` are also provided.


LinearSpace
===========

A `LinearSpace` is a very important subclass of `Set` and is a general implementation of a mathematical `vector space
<https://en.wikipedia.org/wiki/Vector_space>`_. In odl there are a few kinds of spaces that you will face. 

.. _Function Space: https://en.wikipedia.org/wiki/Function_space
.. _Lp: https://en.wikipedia.org/wiki/Lp_space
.. _Linear operators: https://en.wikipedia.org/wiki/Bounded_operator#Properties_of_the_space_of_bounded_linear_operators
.. _CUDA: https://en.wikipedia.org/wiki/CUDA

* Continuous function spaces such as `FunctionSpace`, intended to represent mathematical `Function space`_ such as the lebesgue space `Lp`_ or the space of `Linear operators`_. These are mostly used to represent abstract concepts, and are seldomly used in actual computation.

* :math:`\mathbb{F}^n` type spaces such as `Rn` and `Cn`, but also the `CUDA`_ accelerated version `CudaRn`.

* Discretizations of continous spaces. This may for example be a discretization of a cube using voxels. All discretizations inherit form `RawDiscretization`, but the most important is `DiscreteLp`.

* In addition to this, there are utility spaces such as `ProductSpace` which allows the composition of several spaces into a larger space.

In addition to the spaces, all elements in the spaces inherit from `LinearSpaceVector`. Using these vectors, most standard mathematical operations can be expressed

>>> r3 = Rn(3)
>>> x = r3.element([1, 2, 3])
>>> y = r3.element([4, 5, 6])

Arithmetic such as addition and multiplication by scalars

>>> x + y
Rn(3).element([5.0, 7.0, 9.0])

Inner product etc are defined

>>> r3.inner(x, y)
32.0


See also in depth guide on :ref:`linearspace_in_depth`.

Operator
========

A operator is a `function
<https://en.wikipedia.org/wiki/Function_(mathematics)>`_ from some `Set` to another. In odl these inherit from the abstract class `Operator`.

See also in depth guide on :ref:`operators_in_depth`.


Discretizations
===============

Discretizations ...

Diagnostics
===========

Odl also offers tools to verify the correctness of operators and spaces. Examples include verifying that a `Operator`'s derivative is correct:

>>> op = MyOperator()
>>> odl.diagnostics.OperatorTest(op).derivative()

or verifying that a `LinearSpace` satisfies all expected properties

>>> r3 = odl.Rn(5)
>>> odl.diagnostics.SpaceTest(r3).run_tests()

See `SpaceTest` and `OperatorTest` for more details.
