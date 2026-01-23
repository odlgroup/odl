# Core
This folder contains the core ODL code.

* [Array-API support](array_API_support): Code to implement the array-API defined functions as ODL functions. Also contains the ArrayBackend Dataclass to handle multi-backends.

* [diagnostics](diagnostics): Automated tests for user-defined operators and spaces. `SpaceTest` verifies that various properties of linear spaces work as expected, while `OperatorTest` does the same for operators.

* [discr](discr): Contains the set of discretized functions on some domain.

* [operator](operator): Operators between sets. Defines the class `Operator` which is the main abstract class used for any mapping between two `Set`'s. Further defines several general classes of operators applicable to general spaces.

* [phantom](phantom): Standardized test images. Functions for generating standardized test examples such as `shepp_logan`.

* [set](set): Sets of objects. Defines the abstract class `Set` and `LinearSpace` as well as some concrete implementations such as `RealNumbers`.

* [space](space): Concrete vector spaces. Contains concrete implementations of `LinearSpace`, including `NumpyTensorSpace` and `ProductSpace`.

* [sparse](sparse): Multi-backend sparse arrays handling. 

* [util](util) Utilities. Functionality mainly intended to be used by other ODL functions such as linear algebra and visualization.
