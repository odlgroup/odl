space
=====

Abstract linear vector spaces.

The classes in this module represent abstract mathematical concepts
of vector spaces. They cannot be used directly but are rather intended
to be sub-classed by concrete space implementations. The spaces
provide default implementations of the most important vector space
operations. See the documentation of the respective classes for more
details.

The abstract `LinearSpace` class is intended for quick prototyping.
It has a number of abstract methods which must be overridden by a
subclass. On the other hand, it provides automatic error checking
and numerous attributes and methods for convenience.


.. currentmodule:: odl.set.space



Classes
-------

.. autosummary::
   :toctree: generated/

   ~odl.set.space.LinearSpace
   ~odl.set.space.LinearSpaceVector
   ~odl.set.space.UniversalSpace


