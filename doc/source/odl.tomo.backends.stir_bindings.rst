stir_bindings
=============

Backend for STIR: Software for Tomographic Reconstruction

Back and forward projectors for PET.

`ForwardProjectorByBinWrapper` and `BackProjectorByBinWrapper` are general
objects of STIR projectors and back-projectors, these can be used to wrap a
given projector.

`projector_from_file` allows users a easy way to create a
`ForwardProjectorByBinWrapper` by giving file paths to the required templates.

See the STIR `webpage
<http://stir.sourceforge.net/>`_ for more information.


.. currentmodule:: odl.tomo.backends.stir_bindings



Classes
-------

.. autosummary::
   :toctree: generated/

   ~odl.tomo.backends.stir_bindings.BackProjectorByBinWrapper
   ~odl.tomo.backends.stir_bindings.ForwardProjectorByBinWrapper
   ~odl.tomo.backends.stir_bindings.StirVerbosity


Functions
---------

.. autosummary::
   :toctree: generated/

   ~odl.tomo.backends.stir_bindings.stir_projector_from_file

