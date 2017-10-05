# ODL

This directory contains all of the source code related to ODL.

Briefly, the hierarchy of submodules is:

* [set](set) and [operator](operator) contains the core abstract functionality of ODL needed to define sets, vector spaces and operators acting on these.
* [space](space) and [discr](discr) contains the standard spaces such as Rn and the set of discretized functions on some domain.
* [solvers](solvers) defines equation solvers as well as various optimization algorithms.
* [tomo](tomo), [trafos](trafos), [deform](deform) contains application specific operators like Fourier transforms, Wavelet transforms, deformations, ray transforms, etc.

## Content

This is a brief description of the content of each submodule, see the individual modules for a more detailed description.

* [contrib](contrib) Sub-package for immature and/or very specific code. Examples includes vendor specific geometries, bindings to less used libraries and very new solution methods.
* [deform](deform) Functionality related to deformations. Defines the free function `linear_deform` which deforms a function according to a vector-field of displacements. Also defines the operators `LinDeformFixedTempl` and `LinDeformFixedDisp`.
* [diagnostics](diagnostics) Automated tests for user defined operators and spaces. `SpaceTest` verifies that various properties of linear spaces work as expected, while `OperatorTest` does the same for operators.
* [discr](discr) Discretizations of function-spaces. The main abstract class is `Discretization`, while `DiscreteLp` is an abstract class defining discretized functions on some hypercube. In addition, the classes `RectGrid` and `RectPartition` are used to exactly define what discretization is used under the hood. Finally this submodule defines several utilities like `uniform_discr` and `uniform_partition` which serve to create the most common special cases.
* [operator](operator) Operators between sets. Defines the class `Operator` which is the main abstract class used for any mapping between two `Set`'s. Further defines several general classes of operators applicable to general spaces.
* [phantom](phantom) Standardized test images. Functions for generating standardized test examples such as `shepp_logan`.
* [set](set) Sets of objects. Defines the abstract class `Set` and `LinearSpace` as well as some concrete implementations such as `RealNumbers`.
* [solvers](solvers) Solution of equations and optimization. Contains both general solvers for problems of the form `A(x) = b` where `A` is an `Operator` as well as solvers of minimization problems. In addition, it defines the class `Functional` with several concrete implementations such as `L2Norm`.
* [space](space) Concrete vector spaces. Contains concrete implementations of `LinearSpace`, including `NumpyTensorSpace` and `ProductSpace`.
* [test](test) Tests for the ODL package. This contains automated tests for all other ODL functionality. In general, users should not be calling anything from this submoduel.
* [tomo](tomo) Tomography. Defines the operator `RayTransform` as well as `Geometry` along with subclasses and utilities. Also defines problem dependent direct reconstruction such as `fbp_op`.
* [trafos](trafos) Transformations between spaces. Defines `FourierTransform` and `WaveletTransform`.
* [ufunc_ops](ufunc_ops) Ufuncs as operators. Defines operators like the `sin` and `abs` functions.
* [util](util) Utilities. Functionality mainly intended to be used by other ODL functions such as linear algebra and visualization.