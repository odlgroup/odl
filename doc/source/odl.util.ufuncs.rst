ufuncs
======

 UFuncs for ODL vectors.

These functions are internal and should only be used as methods on
`NtuplesBaseVector` type spaces.

See `numpy.ufuncs
<http://docs.scipy.org/doc/numpy-1.10.0/reference/ufuncs.html#universal-functions-ufunc>`_
for more information.

Notes
-----
The default implementation of these methods make heavy use of the
`NtuplesBaseVector.__array__` to extract a `numpy.ndarray` from the vector,
and then apply a ufunc to it. Afterwards, `NtuplesBaseVector.__array_wrap__`
is used to re-wrap the data into the appropriate space.


.. currentmodule:: odl.util.ufuncs



Classes
-------

.. autosummary::
   :toctree: generated/

   ~odl.util.ufuncs.CudaNtuplesVectorUFuncs
   ~odl.util.ufuncs.DiscreteLpVectorUFuncs
   ~odl.util.ufuncs.NtuplesBaseVectorUFuncs
   ~odl.util.ufuncs.NtuplesVectorUFuncs


Functions
---------

.. autosummary::
   :toctree: generated/

   ~odl.util.ufuncs.method
   ~odl.util.ufuncs.wrap_method_base
   ~odl.util.ufuncs.wrap_method_discretelp
   ~odl.util.ufuncs.wrap_method_ntuples

