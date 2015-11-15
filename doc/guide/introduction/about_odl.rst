#########
About ODL
#########

ODL is a Python library for inverse problems. It contains all the abstract mathematical tools needed from analysis, such as sets, vector spaces and operators as well as optimized implementations of many of the most common of these.

Set
===

A :py:class:`~odl.set.sets.Set` is the fundamental building block of odl objects. It is intended to mirror the mathematical concept of a `real numbers
<https://en.wikipedia.org/wiki/Set_(mathematics)>`_, and has methods to check if an element is contained in the set

>>> interv = Interval(0, 1)
>>> 0.5 in interv
True
>>> 2.0 in interv
False

The most commonly used sets in odl is the :py:class:`~odl.RealNumbers` the set of all `real numbers
<https://en.wikipedia.org/wiki/Real_number>`_ and :py:class:`~odl.IntervalProd`, arbitrary `n-dimensional hypercubes
<https://en.wikipedia.org/wiki/Hypercube>`_. Several convenience sub-classes such as :py:class:`~odl.Interval`,  :py:class:`~odl.set.domain.Rectangle` and :py:class:`~odl.Cuboid` are also provided.

LinearSpace
===========
A :py:class:`~odl.set.space.LinearSpace` is a very important subclass of :py:class:`~odl.set.sets.Set` and is a general implementation of a mathematical `vector space
<https://en.wikipedia.org/wiki/Vector_space>`_. In odl there are two "kinds" of spaces that you will face. The first is F^n type spaces such as :py:class:`~odl.Rn` and :py:class:`~odl.Cn`, but also the `CUDA
<https://en.wikipedia.org/wiki/CUDA>`_ accelerated version :py:class:`~odl.CudaRn`.