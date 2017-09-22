# Examples

This directory contains examples of how to use various ODL functionality.

## Content

* [deform](deform) How to create and use the operators in the `odl.deform` package.
* [diagnostics](diagnostics) How to use `SpaceTest` and `OperatorTest` to test operators and spaces.
* [operator](operator) Examples on how to create sub-classes of `Operator` and call these.
* [solvers](solvers) How to use the various solvers included in the ODL package. This folder contains some of the more complete examples of ODL functionality, all the way from creating e.g. a `RayTransform` to using it to solve an inverse problem.
* [space](space) Demonstrations of general functionality of spaces and how to create a new space.
* [tomo](tomo) Specific examples on how to create and call `RayTransform` for various geometries. Also contains examples of the various analytic reconstruction methods (but see `solvers` for more examples) and performance examples.
* [trafos](trafos) Examples on how to create and call `FourierTransform` and `WaveletTransform`. See `solvers` for examples on how they can be used in inverse problems.
* [ufunc_ops](ufunc_ops) Examples on how to use functionality like `sin` as an operator.
* [visualization](visualization) Demonstrates the various visualization functionality included in ODL, including 1d, 2d and slice views, in addition to real-time updates.