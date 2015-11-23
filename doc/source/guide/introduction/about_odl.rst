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
<https://en.wikipedia.org/wiki/Vector_space>`_. In odl there are two "kinds" of spaces that you will face. The first is F^n type spaces such as `Rn` and `Cn`, but also the `CUDA
<https://en.wikipedia.org/wiki/CUDA>`_ accelerated version `CudaRn`.