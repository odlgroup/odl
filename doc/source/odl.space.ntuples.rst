odl.space.ntuples package
=========================

CPU implementations of ``n``-dimensional Cartesian spaces.

This is a default implementation of :math:`A^n` for an arbitrary set
:math:`A` as well as the real and complex spaces :math:`R^n` and
:math:`C^n`. The data is represented by NumPy arrays.


.. currentmodule:: odl.space.ntuples



Functions
---------

.. autosummary::
   :toctree: generated/

   _blas_is_applicable
   _inner_default
   _lincomb
   _norm_default
   _pnorm_default
   _pnorm_diagweight
   _repr_space_funcs
   _weighting
   weighted_dist
   weighted_inner
   weighted_norm


Classes
-------

.. autosummary::
   :toctree: generated/

   Cn
   CnVector
   Fn
   FnConstWeighting
   FnCustomDist
   FnCustomInnerProduct
   FnCustomNorm
   FnMatrixWeighting
   FnNoWeighting
   FnVector
   FnVectorWeighting
   MatVecOperator
   Ntuples
   NtuplesVector
   Rn
   RnVector

