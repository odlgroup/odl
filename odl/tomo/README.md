# odl.tomo

This directory contains all of the source code related tomographic reconstruction.

## Content

* [analytic](analytic) Analytic reconstruction methods such as filtered back-projection. Also contains various utilities like `parker_weighting`.
* [backends](backends) Bindings to external libraries.
* [geometry](geometry) Definitions of projection geometries.
* [operators](operators) Defines the `RayTransform` operator and its adjoint ("back-projection").
* [util](util) Utilities used internally.