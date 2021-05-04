.. _release_notes:

.. tocdepth: 0

#############
Release Notes
#############

Upcoming release
================

ODL 0.7.0 Release Notes (2018-09-09)
====================================
This release is a big one as it includes the cumulative work over a period of 1 1/2 years.
It is planned to be the last release before version 1.0.0 where we expect to land a number of exciting new features.

Highlights
----------

Native multi-indexing of ODL space elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``DiscreteLpElement`` and ``Tensor`` (renamed from ``FnBaseVector``) data structures now natively support almost all kinds of Numpy "fancy" indexing.
Likewise, the spaces ``DiscreteLp`` and ``Tensorspace`` (renamed from ``FnBase``) have more advanced indexing capabilities as well.
Up to few exceptions, ``elem[indices] in space[indices]`` is always fulfilled.
Alongside, ``ProductSpace`` and its elements also gained more advanced indexing capabilities, in particular in the case of power spaces.

Furthermore, integration with Numpy has been further improved with the implementation of the ``__array_ufunc__`` interface.
This allows to transparently use ODL objects in calls to Numpy UFuncs, e.g., ``np.cos(odl_obj, out=odl_obj)`` or ``np.add.reduce(odl_in, axis=0, out=odl_out)`` — both these examples were not possible with the ``__array__`` and ``__array_wrap__`` interfaces.

Unfortunately, this changeset makes the ``odlcuda`` plugin unusable since it only supports linear indexing.
A much more powerful replacement based on CuPy will be added in version 1.0.0.

Integration with deep learning frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ODL is now integrated with three major deep learning frameworks: `TensorFlow <https://www.tensorflow.org/>`_, `PyTorch <https://pytorch.org/>`_ and `Theano <http://www.deeplearning.net/software/theano/>`_.
In particular, ODL ``Operator`` and ``Functional`` objects can be used as layers in neural networks, with support for automatic differentiation and backpropagation.
This makes a lot of (inverse) problems that ODL can handle well, e.g., tomography, accessible to the computation engines of the deep learning field, and opens up a wide range of possibilities to combine the two.

The implementation of this functionality and examples of its usage can be found in the packages `tensorflow <https://github.com/odlgroup/odl/tree/master/odl/contrib/tensorflow>`_, `torch <https://github.com/odlgroup/odl/tree/master/odl/contrib/torch>`_ and `theano <https://github.com/odlgroup/odl/tree/master/odl/contrib/theano>`_ in the ``odl.contrib`` sub-package (see below).

New ``contrib`` sub-package
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The core ODL library is intended to stay focused on general-purpose classes and data structures, and good code quality is a major goal.
This implies that contributions need to undergo scrutiny in a review process, and that some contributions might not be a good fit if they are too specific for certain applications.

For this reason, we have created a new `contrib <https://github.com/odlgroup/odl/tree/master/odl/contrib>`_ sub-package that is intended for exactly this kind of code.
As of writing this, ``contrib`` already contains a number of highly useful modules:

- `datasets <https://github.com/odlgroup/odl/tree/master/odl/contrib/datasets>`_: Loaders and utility code for publicly available datasets (currently FIPS CT, Mayo clinic human CT, Tu Graz MRI and some image data)
- `fom <https://github.com/odlgroup/odl/tree/master/odl/contrib/fom>`_: Implementations of Figures-of-Merit for image quality assessment
- `mrc <https://github.com/odlgroup/odl/tree/master/odl/contrib/mrc>`_: Reader and writer for the MRC 2014 data format in electron microscopy
- `param_opt <https://github.com/odlgroup/odl/tree/master/odl/contrib/param_opt>`_: Optimization strategies for method hyperparameters
- `pyshearlab <https://github.com/odlgroup/odl/tree/master/odl/contrib/pyshearlab>`_: Integration of the `pyshearlab <https://github.com/stefanloock/pyshearlab>`_ Python library for shearlet decomposition and analysis
- `shearlab <https://github.com/odlgroup/odl/tree/master/odl/contrib/shearlab>`_: Integration of the `Shearlab.jl <https://github.com/arsenal9971/Shearlab.jl>`_ Julia shearlet library
- `solvers <https://github.com/odlgroup/odl/tree/master/odl/contrib/solvers>`_: More exotic functionals and optimization methods than in the core ODL library
- `tomo <https://github.com/odlgroup/odl/tree/master/odl/contrib/tomo>`_: Vendor- or application-specific geometries (currently Elekta ICON and XIV)
- `tensorflow <https://github.com/odlgroup/odl/tree/master/odl/contrib/tensorflow>`_: Integration of ODL with TensorFlow
- `theano <https://github.com/odlgroup/odl/tree/master/odl/contrib/theano>`_: Integration of ODL with Theano
- `torch <https://github.com/odlgroup/odl/tree/master/odl/contrib/torch>`_: Integration of ODL with

Overhaul of tomographic geometries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The classes for representing tomographic geometries in ``odl.tomo`` have undergone a major update, resulting in a consistent definition of coordinate systems across all cases, `proper documentation <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_, vectorization and broadcasting semantics in all methods that compute vectors, and significant speed-up of backprojection due to better axis handling.
Additionally, factory functions ``cone_beam_geometry`` and ``helical_geometry`` have been added as a simpler and more accessible way to create cone beam geometries.

-----

New features
------------
- Function ``pkg_supports`` for tracking package features (:pull:`976`).
- Class ``CallbackShowConvergence`` for tracking values of functionals in a plot (:pull:`832`).
- Context manager ``NumpyRandomSeed`` for setting and resetting the random seed, to get reproducible randomness (:pull:`1003`).
- Parameter ``seed`` in noise phantoms for reproducible results (:pull:`1003`).
- Function ``as_scipy_functional`` that allows using ``Functional`` instances and their gradients in SciPy's optimization methods (:pull:`1004`).
- New ``text`` phantom to create images from arbitrary text (:pull:`1009`, :pull:`1072`).
- Class ``CallbackPrintHardwareUsage`` for monitoring of OS resources during an optimization loop (:pull:`1024`).
- New ``odl.contrib`` sub-package as a place for user-contributed code that lives outside the ODL core, but is still bundled with it (:pull:`1020`).
- Class ``FiniteSet`` with some simple set logic (:pull:`865`).
- Alternative constructor ``frommatrix`` for tomographic geometries which takes a matrix that rotates (and scales) the default coordinate system. This is an advanced interface that gives full control over the initialization (:pull:`968`).
- Factory function ``cone_beam_geometry`` as a simple interface to cone beam geometries (:pull:`968`).
- Class ``FunctionalQuadraticPerturb`` that supersedes ``FunctionalLinearPerturb``, with an additional quadratic terms and the usual rules for gradient and proximal (:pull:`1066`).
- Method ``Operator.norm`` that allows to implement exact (constant) values for operator norms, as well as estimating them with a power iteration (:pull:`1067`).
- Two phantoms ``smooth_cuboid`` and ``tgv_phantom`` (:pull:`1081`, :pull:`1082`, :pull:`1041`).
- Operator ``ComplexModulus``, often used in MRI and phase contrast imaging (:pull:`1041`).
- Optimization method ``adam`` that is popular in the machine learning community (:pull:`972`).
- Class ``CallbackProgressBar`` for prettier progress display in solvers (:pull:`1097`).
- Additional ``axis`` parameter in the ``squeeze`` methods on ``RectGrid`` and ``RectPartition`` for axis-specific squeezing (:pull:`1110`).
- Tomographic ``Geometry`` classes now support indexing ``geom[indices]`` for extraction of sub-geometries. This is particularly useful for reconstruction methods that split up the forward operator, e.g., Kaczmarz (:pull:`1110`).
- Additional ``gamma_dual`` parameter in the ``pdhg`` solver (renamed from ``chambolle_pock_solver``) for doing acceleration in the dual variable instead of the primal (:pull:`1092`).
- Function ``linear_deform`` now exposed (:pull:`1140`).
- Phantom ``uniform_noise`` (:pull:`1148`).
- Optimization method ``admm_linearized`` implementing the linearized version of the ADMM (Alternating Direction Method of Multipliers) (:pull:`1198`).
- Functional ``Huber``,  a smoothed version of the L1 Norm (:pull:`1191`).
- Functional ``BregmanDistance`` and a method ``Functional.bregman`` as helpers to implement "Bregmanized" versions of regularization methods (:pull:`1267`, :pull:`1340`).
- Optimization method ``adupdates``, an implementation of the Alternating Dual method of McGaffin and Fessler for nonsmooth optimization (:pull:`1243`).
- Helper function ``helical_geometry`` to quickly create helical cone beam geometries (:pull:`1157`).
- Helper functions ``douglas_rachford_pd_stepsize`` and ``pdhg_stepsize`` for automatically computing step-size-like parameters for solvers that ensure theoretical convergence (:pull:`1286`, :pull:`1360`).
- Optimization methods ``dca``, ``prox_dca`` and ``doubleprox_dca`` for difference-of-convex type problems (:pull:`1307`).
- Functionals ``IndicatorSimplex`` and ``IndicatorSumConstraint`` with proximals, for restraining solutions of optimization problems to simplices (:pull:`1347`).

Updates/additions to ``contrib``
--------------------------------
- New ``datasets`` sub-package for code to programatically load publicly available datasets from the web; initially containing two FIPS datasets for X-ray CT, Mayo clinic real human CT data, three MRI datasets from TU Graz, as well as some images for image processing applications (:pull:`992`, :pull:`1041`, :pull:`1193`, :pull:`1211`, :pull:`1352`, :pull:`1321`, :pull:`1367`, :pull:`1383`, :pull:`1421`).
- New ``tomo`` sub-package for application- or device-specific geometries and projection operators; initially populated with implementations for the Elekta ICON and XVI CT systems (:pull:`1035`, :pull:`1125`, :pull:`1138`).
- New ``fom`` sub-package for figures-of-merit (FOMs) that measure image quality (:pull:`1018`, :pull:`972`, :pull:`1116`, :pull:`1128`, :pull:`1108`, :pull:`1126`, :pull:`1144`, :pull:`1163`, :pull:`1280`, :pull:`1419`).
- New ``solvers`` sub-package for application-specific solvers and experimental optimization code; initally contains a nonlocal means functional (:pull:`1052`).
- New ``tensorflow`` sub-package featuring seamless two-way integration of ODL and Tensorflow. This allows ODL operators and functionals to be used as layers in neural networks, which opens up a big range of (inverse problems) applications to the world of deep learning.
  Conversely, Tensorflow computation graphs can be treated as ODL vector space elements and, e.g., be fed to ODL solvers, resulting in an abstract representation of the result as a new computation graph (:pull:`972`, :pull:`1271`, :pull:`1366`).
- New ``theano`` sub-package featuring support for ODL operators and functionals as ``theano.Op``. Unfortunately, this has limited usefulness since the Theano project has been stopped (:pull:`1098`).
- New ``pytorch`` sub-package integrating ODL with PyTorch, such that operators and functionals can be used in PyTorch neural nets, with similar implications as for the ``tensorflow`` integration, although only one-way (:pull:`1109`, :pull:`1160`, :pull:`1393`).
- New ``pyshearlab`` sub-package implementing bindings for the pyshearlab library for shearlet decomposition and analysis in 2D (:pull:`1115`).
- New ``solvers.spdhg`` sub-package containing a stochastic version of the PDHG optimizer (:pull:`1194`, :pull:`1326`).
- New ``shearlab`` sub-package with a wrapper for the Julia package ``Shearlab.jl`` that implements shearlet decomposition and analysis (:pull:`1322`, :pull:`1372`).
- New ``param_opt`` sub-package for parameter optimization strategies, e.g. regularization parameters in inverse problems (:pull:`1280`).
- Bugfix: MRC headers with invalid axis order entries are now handled properly (:pull:`990`).

Improvements
------------
- Anisotropic voxels are now supported in 3D tomographic projections with the ASTRA toolbox (:pull:`976`).
- Zero-dimensional grids, partitions and ``DiscreteLp`` instances are now supported. They come up once in a while, e.g., during splitting or when building up something axis by axis (:pull:`995`).
- ``DiscreteLp`` can now have a mixture of uniform and non-uniform axes, and (most) operators that take an ``axis`` argument work with this. A major use case is ranges of tomographic projections with non-uniform angles (:pull:`996`, :pull:`1000`).
- An annoying ``ComplexWarning`` in ``ProductSpace.inner`` was silenced by correct code (:pull:`1005`).
- ``Operator`` now disallows returning a different ``out`` than was passed in. This catches erroneous code that would allocate a new element regardless and return that, instead of using the provided ``out`` element (:pull:`1007`).
- FFTs now use the fastest available backend by default, instead of defaulting to Numpy's FFT (:pull:`1006`).
- Many classes now make more use of caching of their computed properties to save the computational cost. Some of those properties are on hot code paths and make a big difference for the final runtime of typical code. Furthermore, heavily used functions with only a small number of possible inputs make use of an LRU input cache (:pull:`1012`).
- The performance of the ``douglas_rachford_pd`` solver was improved by the use of a temporary and in-place arithmetic (:pull:`1012`).
- Linear combination in :math:`R^n` like spaces uses BLAS only for arrays of more than 50000 entries; below that threshold, a naive implementation tends to be faster (:pull:`1012`).
- All ``Callback`` classes now support the ``step`` parameter (:pull:`1021`).
- The ``pdhg`` solver (then ``chambolle_pock_solver``) precomputes proximals for a 25 % speed-up (:pull:`1027`).
- The ``indices`` sequence in ``show`` methods now takes ``None`` entries as ``slice(None)``, thereby mirroring the behavior of the ``coords`` parameter (:pull:`1029`).
- Several functions (``parker_weighting``, ``fpb_filter``, the ASTRA CUDA wrappers) got performance tweaks (:pull:`1035`).
- A number of code paths have been made faster by removing redundant checks, getting rid of ``abc``, caching, etc. (:pull:`1043`).
- The whole system of tomographic geometries was overhauled with better internal consistency, clearer definitions of coordinate systems, vectorization of methods, and, most importantly, proper documentation (:pull:`968`, :pull:`1159`).
- The ``indicate_proj_axis`` phantom can now be used in 2D as well (:pull:`968`).
- The ODL to ASTRA geometry translation tries as hard as possible to make the data layout beneficial for performance (less axis swapping). In 3D, this gives a whopping 15x speedup compared to the previous implementation (:pull:`968`).
- The duration of ``import odl`` was decreased with a number of optimizations, most of them consisting in lazy loading of modules or lazy evaluation of expressions that are not strictly needed at import time (:pull:`1090`, :pull:`1112`, :pull:`1402`).
- ``ProductSpaceElement`` now implements the ``__array__`` interface if its ``space`` is a power space (:pull:`972`).
- A mutex was added to the ASTRA CUDA wrapper classes, to avoid race conditions between threads, e.g. when using ``tensorflow`` (:pull:`972`).
- Calls to ``super`` have been carefully revised and unified, either as ``super(<class_name>, self).<attr>`` for collaborative multiple inheritance, or as hard-wired ``OtherClass.<attr>`` if a very specific attribute should be used. As an aside, remnants of the slow ``super`` from the ``future`` module have been removed (:pull:`1161`).
- ``Detector`` subclasses can opt out of bounds checking with the new ``check_bounds`` parameter (:pull:`1059`).
- ``CallbackPrintIteration`` now passes through keyword args to the ``print`` function, and the ``CallbackPrintTiming`` has gotten a ``cumulative`` parameter (:pull:`1176`).
- Printing of ODL space elements, operators and others has been improved, and the implementation has been simplified with helper functions (:pull:`1203`).
- The internal representation of vector spaces and similar structures has been significantly simplified. Before, there were a number of ``*Set`` and ``*Space`` classes alongside, where the former was a more general version of the latter with less structure and fewer capabilities. This separation has been removed in favor of *duck-typing*: if it quacks like a space (e.g. has an inner product), it is a space (:pull:`1205`).
- A number of operators (differential operators like ``Gradient`` and pointwise vector field operators like ``PointwiseNorm``) have been equipped with the capability of customizing their ranges (:pull:`1216`).
- Phantoms now take two additional parameters ``min_pt`` and ``max_pt`` that allow restricting their extent to a subvolume if both are given, or shift the phantom if only one of them is given (:pull:`1223`).
- ``KullbackLeiblerCrossEntropy.proximal`` now works with complex spaces (:pull:`1088`).
- The ``insert`` method of ``IntervalProd``, ``RectGrid`` and ``RectPartition`` now takes an arbitrary number of objects to insert (:pull:`1088`).
- Numpy ``ufunc`` operators with 2 disparate output data types are now supported (:pull:`1088`).
- ``ProductSpace.shape`` now recursively determines the axes and its sizes in case of power spaces. The ``size`` and ``ndim`` properties work accordingly, i.e., ``len(pspace)`` is no longer necessarily the same as ``pspace.ndim``, as for Numpy arrays (:pull:`1088`).
- ``ProductSpace`` and its elements now support indexing with integers, slices, tuples and lists (:pull:`1088`).
- The ``TensorSpace`` class (replacement for ``FnBase``) and its element class ``Tensor`` (and by analogy also ``DiscreteLp`` and its elements) now fully and natively support Numpy "fancy" indexing up to very few exceptions (:pull:`1088`).
- ``Tensor`` and ``DiscreteLpElement`` support the Numpy 1.13 ``__array_ufunc__`` interface which allows classes to take control over how ufuncs are evaluated. With this interface, it is possible to transparently perform in-place operations like ``np.cos(odl_obj, out=odl_obj)``, which was not possible with ``__array__`` and ``__array_wrap__`` before. Furthermore, other methods of Numpy ufuncs are available, e.g. ``np.add.reduce(odl_in, axis=0, out=odl_out)`` (:pull:`1088`).
- A non-discretized ``FunctionSpace`` can now be vector- or tensor-valued, using a Numpy ``dtype`` with shape, e.g., ``np.dtype((float, (2, 3)))`` (:pull:`1088`).
- The ``element`` methods of ``TensorSpace`` and ``DiscreteLp`` have a new ``order`` parameter to determine the array memory layout (:pull:`1088`).
- ``ProductSpaceElement.asarray`` has been added (:pull:`1152`).
- ``SeparableSum`` now accepts vector-valued step sizes, and several functionals (e.g. ``L1Norm``) takes pointwise step sizes, with full support for proximal, convex conjuage etc. (:pull:`1166`).
- ``KullbackLeibler.convex_conj`` now works on product spaces (:pull:`1287`).
- Generation of the sparse matrix containing the operators in ``ProductSpaceOperator`` is now more robust and disallows malformed constructions like ``ProductSpaceOperator([A, B])`` with matrices that are not 2D (:pull:`1293`, :pull:`1295`).
- ``ProductSpace`` and ``ProductSpaceElement`` now implement ``real_space``, ``complex_space``, ``real``, ``imag``, ``conj``, ``astype`` and ``__array_wrap__`` where applicable (:pull:`1288`).
- ``matrix_representation`` now works with arbitrary tensor spaces as domain and range of an operator. The result will be a tensor with the sum of the number of axes in domain and range (:pull:`1308`).
- Optimizations for common cases in ``PointwiseNorm`` have been added, making the code run 1.5-2 times faster in typical conditions (:pull:`1318`).
- Several complex-to-real operators like ``ComplexModulus`` now have a ``derivative`` that implements the :math:`\mathbb{C} = \mathbb{R}^2` interpretation. Furthermore, linearity is interpreted in the same sense, allowing optimization of certain operations (:pull:`1324`, :pull:`1331`).
- The colorbar in plots from ``show`` can new be turned off with the ``colorbar`` flag (:pull:`1343`).
- ``FunctionSpace`` and ``ProductSpace`` now have properties ``is_real`` and ``is_complex`` (:pull:`1348`).
- ``power_method_opnorm`` now starts from a noise element, making it easier to use with operators that have null spaces, like ``Gradient`` (:pull:`1286`).
- The default of the ``omega`` relaxation parameter in the ``landweber`` solver has been changed from 1 to ``1 / op.norm(estimate=True) ** 2``, which theoretically guarantees convergence (:pull:`1286`).
- For the solvers ``douglas_rachford_pd`` and ``pdhg``, the step-size-like parameters have been made optional, with the default values being computed automatically using some heuristics and the bound that guarantees convergence (:pull:`1286`).
- The ``LpNorm`` proximal now also supports exponent infinity (:pull:`1347`).
- Filters for FBP reconstruction can now be given as arrays to ``fbp_op`` (:pull:`1379`).
- ``ProductSpace`` and its element type now implement ``nbytes`` (:pull:`1410`).

Bugfixes
--------
- Resolve an issue with negative indices resulting in a truncated image in ``ellipsoid_phantom`` (:pull:`998`).
- ``MultiplyOperator.adjoint`` now works for scalar domain and range (:pull:`987`).
- ``ReductionOperator._call`` now properly unwraps the result before returning it (:pull:`1012`, :pull:`1010`).
- Fix the issue of ``0 * log(0)`` producing ``NaN`` in ``KullbackLeibler`` (:pull:`1042`).
- Sometimes, titles of figures resulting from ``show`` would be clipped. This is now fixed (:pull:`1045`).
- ``Parallel3dEulerGeometry`` now actually works with ASTRA projectors (:pull:`968`).
- Fix a rounding error preventing colorbar ticks to show up in ``show`` (:pull:`1063`).
- ``DiscreteLp.astype`` now propagates its axis labels as expected (:pull:`1073`).
- Resolve an issue with wrong inner products on non-uniformly discretized spaces (:pull:`1096`).
- ``CallbackStore`` now works with objects that do have a ``copy`` method but do implement ``__copy__`` (:pull:`1094`).
- ``RayTransform`` and FBP operators used the wrong projection space weighting if the reconstruction space was unweighted. This was fixed, but the patch has been superseded by :pull:`1088` (:pull:`1099`, :pull:`1102`).
- Fix ``LinearSpace.zeros`` using the wrong order of arguments (:pull:`972`).
- ``ProductSpaceElement`` now has a (space pass-through) ``shape`` property (:pull:`972`).
- Resolve several issues with complex spaces in optimization problems (:pull:`1120`).
- The tick labels in ``show`` are now "NaN-proof" (:pull:`1092`, :pull:`1158`, :pull:`1088`).
- Fix a bug in ``nonuniform_partition`` that caused length-1 inputs to crash the function (:pull:`1141`).
- Fix ``DiscreteLpElement.real`` (and ``.imag``) sometimes returning a copy instead of a view (:pull:`1155`).
- Fix ``ConeFlatGeometry`` not propagating ``pitch`` in its ``__getitem__`` method (:pull:`1173`).
- Fix a bug in ``parker_weighting`` caused by the change of geometry definitions (:pull:`1175`).
- Resolve an issue with wrong results of the L1 convex conjugate proximal when input and output were aliased (:pull:`1182`).
- Correct the implementation of ``Operator{Left,Right}VectorMult.adjoint`` for complex spaces (:pull:`1192`).
- Add a workaround for the fact BLAS internally works with 32-bit integers as indices, which goes wrong for very large arrays (:pull:`1190`).
- Fix Numpy errors not recognizing ``builtins.int`` from the ``future`` library as valid ``dtype`` by disallowing that object as ``dtype`` internally (:pull:`1205`).
- Resolve a number of minor issues with geometry methods' broadcasting (:pull:`1210`).
- Correct handling of degenerate (size 1) axes in Fourier transform range inference (:pull:`1208`).
- Fix a bug in ``OperatorSum`` and ``OperatorPointwiseProduct`` that resulted in wrong outputs for aliased input and output objects (:pull:`1225`).
- Fix the broken ``field`` determination for ``ProductSpace(space, 0)`` (:pull:`1088`).
- Add back the string dtypes in ``NumpyTensorSpace.available_dtypes`` (:pull:`1236`, :pull:`1294`).
- Disallow bool conversion of ``Tensor`` with ``size > 1`` (:pull:`1235`).
- Fix a sign flip error in 2D geometries (:pull:`1245`).
- Blacklisted several patch versions of NumPy 1.14 due to bugs in new-style array printing that result in failing doctests (:pull:`1265`).
- Correct the implementations of ``PointwiseNorm.derivative`` and ``GroupL1Norm.gradient`` to account for division-by-zero errors (:pull:`1070`).
- Fix issue in ``NumpyTensor.lincomb`` when one of the scalars is NaN (:pull:`1272`).
- Fix indexing into ``RectPartition.byaxis`` producing a wrong result with integers (:pull:`1284`).
- Resolve ``space.astype(float)`` failing for ``space.dtype == bool`` (:pull:`1285`).
- Add a missing check for scalar ``sigma`` in ``FunctionalQuadraticPerturb.proximal`` (:pull:`1283`).
- Fix an error in the adjoint of ``SamplingOperator`` triggered by a ``sampling_points`` argument of length 1 (:pull:`1351`).
- Make ``DiscreteLpElement.show`` use the correct interpolation scheme (:pull:`1375`).
- Fix checking of pyFFTW versions to also support Git revision versions (:pull:`1373`).
- Correct the implementation of ``MultiplyOperator.adjoint`` for complex spaces (:pull:`1390`).
- Replace the improper and potentially ambiguous indexing with tuple indexing as signalled by the Numpy deprecation warning (:pull:`1420`).

API Changes
-----------
- Functions and attributes related to convex conjugates now use ``convex_conj`` as name part instead of ``cconj`` (:pull:`1048`).
- ``ParallelGeometry`` was renamed to ``ParallelBeamGeometry`` (:pull:`968`).
- ``HelicalConeFlatGeometry`` was renamed to ``ConeFlatGeometry``, and ``CircularConeFlatGeometry`` was removed as special case (:pull:`968`).
- ``pitch_offset`` in 3D cone beam geometries was renamed to ``offset_along_axis`` (:pull:`968`).
- ``ellipsoid_phantom`` now takes angles in radians instead of degrees (:pull:`972`).
- The ``L1Norm.gradient`` operator now implements the (ad-hoc) ``derivative`` method, returning ``ZeroOperator`` (:pull:`972`).
- The base class for solver callbacks was renamed from ``SolverCallback`` to ``Callback`` (:pull:`1097`).
- The ``chambolle_pock_solver`` has been renamed to ``pdhg`` (Primal-Dual Hybrid Gradient), along with all references to "Chambolle-Pock" (:pull:`1092`).
- The ``gamma`` parameter in ``pdhg`` (see one above) has been renamed to ``gamma_primal``, since one can now alternatively specify a ``gamma_dual`` acceleration parameter (:pull:`1092`).
- As a result of merging internal ``*Set`` and ``*Space`` classes, a number of arguments to internal class constructors like ``FunctionSpaceMapping`` have been renamed accordingly (:pull:`1205`)
- Remove the (dubious) ``dist_using_inner`` optimization of vector spaces (:pull:`1214`).
- The class ``Ntuples`` has been merged into ``FnBase``, but both have been superseded by :pull:`1088` (:pull:`1205`, :pull:`1216`).
- The ``writable_array`` context manager no longer takes an arbitrary number of positional arguments as pass-through, only keyword arguments (:pull:`1088`).
- ``LinearSpaceElement`` and ``ProductSpaceElement`` are no longer available in the top-level ``odl`` namespace (:pull:`1088`).
- The ``NoWeighting`` classes have been removed due to their odd behavior. For the time being, no weighting is equivalent to weighting with constant 1.0, but this will change a bit in the future (:pull:`1088`).
- The classes ``FnBase`` and ``NumpyFn`` have been removed in favor of ``TensorSpace`` and ``NumpyTensorSpace``. Likewise, the ``fn`` factory function is now called ``tensor_space``, and any other name associated with ``fn`` has been renamed accordingly (:pull:`1088`).
- The ``uspace`` and ``dspace`` properties of ``Discretization`` have been renamed to ``fspace`` ("function space") and ``tspace`` ("tensor space"), respectively (:pull:`1088`).
- With mandatory multi-indexing support for ``TensorSpace`` implementations, the old ``CudaFn`` class is no longer supported. The next release 1.0.0 will have a much more powerful replacement using CuPy, see :pull:`1401` (:pull:`1088`).
- The meanings of the parameters ``f`` and ``g`` has been switched in ``pdhg`` to make the interface match the rest of the solvers (:pull:`1286`).
- Bindings to the STIR reconstruction software have been overhauled and moved out of the core into a separate repository (:pull:`1403`).


ODL 0.6.0 Release Notes (2017-04-20)
====================================
Besides many small improvements and additions, this release is the first one under the new Mozilla Public License 2.0 (MPL-2.0).

New features
------------
- The Kaczmarz method has been added to the ``solvers`` (:pull:`840`).
- Most immutable types now have a ``__hash__`` method (:pull:`840`).
- A variant of the Conjugate Gradient solver for non-linear problems has been added (:pull:`554`).
- There is now an example for tomographic reconstruction using Total Generalized Variation (TGV). (:pull:`883`).
- Power spaces can now be created using the ``**`` operator, e.g., ``odl.rn(3) ** 4``.
  Likewise, product spaces can be created using multiplication ``*``, i.e., ``odl.rn(3) * odl.rn(4)`` (:pull:`882`).
- A ``SamplingOperator`` for the extraction of values at given indices from arrays has been added, along with its adjoint ``WeightedSumSamplingOperator`` (:pull:`940`).
- Callbacks can now be composed with operators, which can be useful, e.g., for transforming the current iterate before displaying it (:pull:`954`).
- ``RayTransform`` (and thus also ``fbp_op``) can now be directly used on spaces of complex functions (:pull:`970`).

Improvements
------------
- In ``CallbackPrintIteration``, a step number between displays can now be specified (:pull:`871`).
- ``OperatorPointwiseProduct`` got its missing ``derivative`` (:pull:`877`).
- ``SeparableSum`` functionals can now be indexed to retrieve the constituents (:pull:`898`).
- Better self-printing of callbacks (:pull:`881`).
- ``ProductSpaceOperator`` and subclasses now have ``size`` and ``__len__``, and the parent also has ``shape``.
  Also self-printing of these operators is now better (:pull:`901`).
- Arithmetic methods of ``LinearSpace`` have become more permissive in the sense that operations like ``space_element + raw_array`` now works if the array can be cast to an element of the same space (:pull:`902`).
- There is now a (work-in-progress) document on the release process with the aim to avoid errors (:pull:`872`).
- The MRC extended header implementation is now much simpler (:pull:`917`).
- The ``show_discrete_data`` workhorse is now more robust towards arrays with ``inf`` and ``nan`` entries regarding colorbar settings (:pull:`921`).
- The ``title`` in ``CallbackShow`` are now interpreted as format string with iteration number inserted, which enables updating the figure title in real time (:pull:`923`).
- Installation instructions have been arranged in a better way, grouped after different ways of installing (:pull:`884`).
- A performance comparison example pure ASTRA vs. ODL with ASTRA for 3d cone beam has been added (:pull:`912`).
- ``OperatorComp`` avoids an operator evaluation in ``derivative`` in the case when the left operator is linear (:pull:`957`).
- ``FunctionalComp`` now has a default implementation of ``gradient.derivative`` if the operator in the composition is linear (:pull:`956`).
- The ``saveto`` parameter of ``CallbackShow`` can now be a callable that returns the file name to save to when called on the current iteration number (:pull:`955`).

Changes
-------
- The ``sphinxext`` submodule has been from upstream (:pull:`846`).
- The renames ``TensorGrid`` -> ``RectGrid`` and ``uniform_sampling`` -> ``uniform_grid`` have been made, and separate class ``RegularGrid`` has been removed in favor of treating regular grids as a special case of ``RectGrid``.
  Instances of ``RectGrid`` have a new property ``is_uniform`` for this purpose.
  Furthermore, uniformity of ``RectPartition`` and ``RectGrid`` is exposed as property per axis using ``is_uniform_byaxis`` (:pull:`841`).
- ``extent`` of grids and partitions is now a property instead of a method (:pull:`889`).
- The number of iterations in solvers is no longer optional since the old default 1 didn't make much sense (:pull:`888`).
- The ``nlevels`` argument of ``WaveletTransform`` is now optional, and the default is the maximum number of levels as determined by the new function ``pywt_max_nlevels`` (:pull:`880`).
- ``MatVecOperator`` is now called ``MatrixOperator`` and has been moved to the ``tensor_ops`` module.
  This solves a circular dependency issue with ODL subpackages (:pull:`911`).
- All step parameters of callbacks are now called just ``step`` (:pull:`929`).
- The ``impl`` name for the scikit-image back-end in ``RayTransform`` has been changed from ``scikit`` to ``skimage`` (:pull:`970`).
- ODL is now licensed under the Mozilla Public License 2.0 (:pull:`977`).

Bugfixes
--------
- Fix an argument order error in the gradient of ``QuadraticForm`` (:pull:`868`).
- Lots of small documentation fixes where ", optional" was forgotten in the Parameters section (:pull:`554`).
- Fix an indexing bug in the ``indicate_proj_axis`` phantom (:pull:`878`).
- Fix wrong inheritance order in ``FileReaderRawBinaryWithHeader`` that lead to wrong ``header_size`` (:pull:`893`).
- Comparison of arbitrary objects in Python 2 is now disabled for a some ODL classes where it doesn't make sense (:pull:`933`).
- Fix a bug in the angle calculation of the scikit-image back-end for Ray transforms (:pull:`947`).
- Fix issue with wrong integer type in ``as_scipy_operator`` (:pull:`960`).
- Fix wrong scaling in ``RayTransform`` and adjoint with unweighted spaces (:pull:`958`).
- Fix normalization bug of ``min_pt`` and ``max_pt`` parameters in ``RectPartition`` (:pull:`971`).
- Fix an issue with ``*args`` in ``CallbackShow`` that lead to the ``title`` argument provided twice (:pull:`981`).
- Fix an unconditional ``pytest`` import that lead to an ``ImportError`` if pytest was not installed (:pull:`982`).


ODL 0.5.3 Release Notes (2017-01-17)
====================================

Lots of small improvements and feature additions in this release.
Most notable are the remarkable performance improvements to the ASTRA bindings (up to 10x), the addition of ``fbp_op`` to create filtered back-projection operators with several filter and windowing options, as well as further performance improvements to operator compositions and the ``show`` methods.

New features
------------
- Add the ``SeparableSum(func, n)`` syntax for n-times repetition of the same summand (:pull:`685`).
- Add the Ordered Subsets MLEM solver ``odl.solvers.osmlem`` for faster EM reconstruction (:pull:`647`).
- Add ``GroupL1Norm`` and ``IndicatorGroupL1UnitBall`` for mixed L1-Lp norm regularization (:pull:`620`).
- Add ``fbp_op`` helper to create filtered back-projection operators for a range of geometries (:pull:`703`).
- Add 2-dimensional FORBILD phantom (:pull:`694`, :pull:`804`, :pull:`820`).
- Add ``IndicatorZero`` functional in favor of of ``ConstantFunctionalConvexConj`` (:pull:`707`).
- Add reader for MRC data files and for custom binary formats with fixed header (:pull:`716`).
- Add ``NuclearNorm`` functional for multi-channel regularization (:pull:`691`).
- Add ``CallbackPrint`` for printing of intermediate results in iterative solvers (:pull:`691`).
- Expose Numpy ufuncs as operators in the new ``ufunc_ops`` subpackage (:pull:`576`).
- Add ``ScalingFunctional`` and ``IdentityFunctional`` (:pull:`576`).
- Add ``RealPart``, ``ImagPart`` and ``ComplexEmbedding`` operators (:pull:`706`).
- Add ``PointwiseSum`` operator for vector fields (:pull:`754`).
- Add ``LineSearchFromIterNum`` for using a pre-defined mapping from iteration number to step size (:pull:`752`).
- Add ``axis_labels`` option to ``DiscreteLp`` for custom labels in plots (:pull:`770`).
- Add Defrise phantom for cone beam geometry testing (:pull:`756`).
- Add ``filter`` option to ``fbp_op`` and ``tam_danielson_window`` and ``parker_weighting`` helpers for helical/cone geometries (:pull:`756`, :pull:`806`, :pull:`825`).
- Add ISTA (``proximal_gradient``) and FISTA (``accelerated_proximal_gradient``) algorithms, among others useful for L1 regularization (:pull:`758`).
- Add ``salt_pepper_noise`` helper function (:pull:`758`).
- Expose FBP filtering as operator ``fbp_filter_op`` (:pull:`780`).
- Add ``parallel_beam_geometry`` helper for creation of simple test geometries (:pull:`775`).
- Add ``MoreauEnvelope`` functional for smoothed regularization (:pull:`763`).
- Add ``saveto`` option to ``CallbackShow`` to store plots of iterates (:pull:`708`).
- Add ``CallbackSaveToDisk`` and ``CallbackSleep`` (:pull:`798`).
- Add a utility ``signature_string`` for robust generation of strings for ``repr`` or ``str`` (:pull:`808`).

Improvements
------------
- New documentation on the operator derivative notion in ODL (:pull:`668`).
- Add largescale tests for the convex conjugates of functionals (:pull:`744`).
- Add ``domain`` parameter to ``LinDeformFixedTempl`` for better extensibility (:pull:`748`).
- Add example for sparse tomography with TV regularization using the Douglas-Rachford solver (:pull:`746`).
- Add support for 1/r^2 scaling in cone beam backprojection with ASTRA 1.8 using a helper function for rescaling (:pull:`749`).
- Improve performance of operator scaling in certain cases (:pull:`576`).
- Add documentation on testing in ODL (:pull:`704`).
- Replace occurrences of ``numpy.matrix`` objects (:pull:`778`).
- Implement Numpy-style indexing for ``ProductSpaceElement`` objects (:pull:`774`).
- Greatly improve efficiency of ``show`` by updating the figure in place instead of re-creating (:pull:`789`).
- Improve efficiency of operator derivatives by short-circuiting in case of a linear operator (:pull:`796`).
- Implement simple indexing for ``ProducSpaceOperator`` (:pull:`815`).
- Add caching to ASTRA projectors, thus making algorithms run much faster (:pull:`802`).

Changes
-------
- Rename ``vector_field_space`` to ``tangent_bundle`` in vector spaces (more adequate for complex spaces) (:pull:`702`).
- Rename ``show`` parameter of ``show`` methods to ``force_show`` (:pull:`771`).
- Rename ``elem.ufunc`` to ``elem.ufuncs`` where implemented (:pull:`809`).
- Remove "Base" from weighting base classes and rename ``weight`` parameter to ``weighting`` for consistency (:pull:`810`).
- Move ``tensor_ops`` module from ``odl.discr`` to ``odl.operator`` for more general application (:pull:`813`).
- Rename ``ellipse`` to ``ellipsoid`` in names intended for 3D cases (:pull:`816`).
- Pick the fastest available implementation in ``RayTransform`` by default instead of ``astra_cpu`` (:pull:`826`).

Bugfixes
--------
- Prevent ASTRA cubic voxel check from failing due to numerical rounding errors (:pull:`721`).
- Implement the missing ``__ne__`` in ``RectPartition`` (:pull:`748`).
- Correct adjoint of ``WaveletTransform`` (:pull:`758`).
- Fix issue with creation of phantoms in a space with degenerate shape (:pull:`777`).
- Fix issue with Windows paths in ``collect_ignore``.
- Fix bad dict lookup with ``RayTransform.adjoint.adjoint``.
- Fix rounding issue in a couple of indicator functionals.
- Several bugfixes in ``show`` methods.
- Fixes to outdated example code.

ODL 0.5.2 Release Notes (2016-11-02)
====================================

Another maintenance release that fixes a number of issues with installation and testing, see :issue:`674`, :issue:`679`, and :pull:`692` and :pull:`696`.


ODL 0.5.1 Release Notes (2016-10-24)
====================================

This is a maintenance release since the test suite was not bundled with PyPI and Conda packages as intended already in 0.5.0.
From this version on, users can run ``python -c "import odl; odl.test()"`` with all types of installations (from PyPI, Conda or from source).


ODL 0.5.0 Release Notes (2016-10-21)
====================================

This release features a new important top level class ``Functional`` that is intended to be used in optimization methods.
Beyond its parent ``Operator``, it provides special methods and properties like ``gradient`` or ``proximal`` which are useful in advanced smooth or non-smooth optimization schemes.
The interfaces of all solvers in ``odl.solvers`` have been updated to make use of functionals instead of their proximals, gradients etc. directly.

Further notable changes are the implementation of an ``as_writable_array`` context manager that exposes arbitrary array storage as writable Numpy arrays, and the generalization of the wavelet transform to arbitrary dimensions.

See below for a complete list of changes.


New features
------------
- Add ``Functional`` class to the solvers package. (:pull:`498`)
  ``Functional`` is a subclass of odl ``Operator`` and intended to help in formulating and solving optimization problems.
  It contains optimization specific features like ``proximal`` and ``convex_conj``, and built-in intelligence for handling things like translation, scaling of argument or scaling of functional.
  * Migrate all solvers to work with ``Functional``'s instead of raw proximals etc. (:pull:`587`)
  * ``FunctionalProduct`` and ``FunctionalQuotient`` which allow evaluation of the product/quotient of functions and also provides a gradient through the Leibniz/quotient rules. (:pull:`586`)
  * ``FunctionalDefaultConvexConjugate`` which acts as a default for ``Functional.convex_conj``, providing it with a proximal property. (:pull:`588`)
  * ``IndicatorBox`` and ``IndicatorNonnegativity`` which are indicator functions on a box shaped set and the set of nonnegative numbers, respectively. They return 0 if all points in a vector are inside the box, and infinity otherwise. (:pull:`589`)
  * Add ``Functional``s for ``KullbackLeibler`` and ``KullbackLeiblerCrossEntropy``, together with corresponding convex conjugates (:pull:`627`).
  Also add proximal operator for the convex conjugate of cross entropy Kullback-Leibler divergence, called ``proximal_cconj_kl_cross_entropy`` (:pull:`561`)
- Add ``ResizingOperator`` for shrinking and extending (padding) of discretized functions, including a variety of padding methods. (:pull:`499`)
- Add ``as_writable_array`` that allows casting arbitrary array-likes to a numpy array and then storing the results later on. This is
  intended to be used with odl vectors that may not be stored in numpy format (like cuda vectors), but can be used with other types like lists.
  (:pull:`524`)
- Allow ASTRA backend to be used with arbitrary dtypes. (:pull:`524`)
- Add ``reset`` to ``SolverCallback`` that resets the callback to its initial state. (:issue:`552`)
- Add ``nonuniform_partition`` utility that creates a partition with non-uniformly spaced points.
  This is useful e.g. when the angles of a tomography problem are not exactly uniform. (:pull:`558`)
- Add ``Functional`` class to the solvers package.
  ``Functional`` is a subclass of odl ``Operator`` and intended to help in formulating and solving optimization problems.
  It contains optimization specific features like ``proximal`` and ``convex_conj``, and built-in intelligence for handling things like translation, scaling of argument or scaling of functional. (:pull:`498`)
- Add ``FunctionalProduct`` and ``FunctionalQuotient`` which allow evaluation of the product/quotient of functions and also provides a gradient through the Leibniz/quotient rules. (:pull:`586`)
- Add ``FunctionalDefaultConvexConjugate`` which acts as a default for ``Functional.convex_conj``, providing it with a proximal property. (:pull:`588`)
- Add ``IndicatorBox`` and ``IndicatorNonnegativity`` which are indicator functions on a box shaped set and the set of nonnegative numbers, respectively. They return 0 if all points in a vector are inside the box, and infinity otherwise. (:pull:`589`)
- Add proximal operator for the convex conjugate of cross entropy Kullback-Leibler divergence, called ``proximal_cconj_kl_cross_entropy`` (:pull:`561`)
- Add ``Functional``'s for ``KullbackLeibler`` and ``KullbackLeiblerCrossEntropy``, together with corresponding convex conjugates (:pull:`627`)
- Add tutorial style example. (:pull:`521`)
- Add MLEM solver. (:pull:`497`)
- Add ``MatVecOperator.inverse``. (:pull:`608`)
- Add the ``Rosenbrock`` standard test functional. (:pull:`602`)
- Add broadcasting of vector arithmetic involving ``ProductSpace`` vectors. (:pull:`555`)
- Add ``phantoms.poisson_noise``. (:pull:`630`)
- Add ``NumericalGradient`` and ``NumericalDerivative`` that numerically compute gradient and derivative of ``Operator``'s and ``Functional``'s. (:pull:`624`)

Improvements
------------
- Add intelligence to ``power_method_opnorm`` so it can terminate early by checking if consecutive iterates are close. (:pull:`527`)
- Add ``BroadcastOperator(op, n)``, ``ReductionOperator(op, n)`` and ``DiagonalOperator(op, n)`` syntax.
  This is equivalent to ``BroadcastOperator(*([op] * n))`` etc, i.e. create ``n`` copies of the operator. (:pull:`532`)
- Allow showing subsets of the whole volume in ``DiscreteLpElement.show``. Previously this allowed slices to be shown, but the new version allows subsets such as ``0 < x < 3`` to be shown as well. (:pull:`574`)
- Add ``Solvercallback.reset()`` which allows users to reset a callback to its initial state. Applicable if users want to reuse a callback in another solver. (:pull:`553`)
- ``WaveletTransform`` and related operators now work in arbitrary dimensions. (:pull:`547`)
- Several documentation improvements. Including:

  * Move documentation from ``_call`` to ``__init__``. (:pull:`549`)
  * Major review of minor style issues. (:pull:`534`)
  * Typeset math in proximals. (:pull:`580`)

- Improved installation docs and update of Chambolle-Pock documentation. (:pull:`121`)

Changes
--------
- Change definition of ``LinearSpaceVector.multiply`` to match the definition used by Numpy. (:pull:`509`)
- Rename the parameters ``padding_method`` in ``diff_ops.py`` and ``mode`` in ``wavelet.py`` to ``pad_mode``.
  The parameter ``padding_value`` is now called ``pad_const``. (:pull:`511`)
- Expose ``ellipse_phantom`` and ``shepp_logan_ellipses`` to ``odl.phantom``. (:pull:`529`)
- Unify the names of minimum (``min_pt``), maximum (``max_pt``) and middle (``mid_pt``) points as well as number of points (``shape``) in grids, interval products and factory functions for discretized spaces. (:pull:`541`)
- Remove ``simple_operator`` since it was never used and did not follow the ODL style. (:pull:`543`)
  The parameter ``padding_value`` is now called ``pad_const``.
- Remove ``Interval``, ``Rectangle`` and ``Cuboid`` since they were confusing (Capitalized name but not a class) and barely ever used.
  Users should instead use ``IntervalProd`` in all cases. (:pull:`537`)
- The following classes have been renamed (:pull:`560`):

  * ``LinearSpaceVector`` -> ``LinearSpaceElement``
  * ``DiscreteLpVector`` -> ``DiscreteLpElement``
  * ``ProductSpaceVector`` -> ``ProductSpaceElement``
  * ``DiscretizedSetVector`` -> ``DiscretizedSetElement``
  * ``DiscretizedSpaceVector`` -> ``DiscretizedSpaceElement``
  * ``FunctionSetVector`` -> ``FunctionSetElement``
  * ``FunctionSpaceVector`` -> ``FunctionSpaceElement``

- Change parameter style of differential operators from having a ``pad_mode`` and a separate ``edge_order`` argument that were mutually exclusive to a single ``pad_mode`` that covers all cases.
  Also added several new pad modes to the differential operators. (:pull:`548`)
- Switch from RTD documentation hosting to gh-pages and let Travis CI build and deploy the documentation. (:pull:`536`)
- Update name of ``proximal_zero`` to ``proximal_const_func``. (:pull:`582`)
- Move unit tests from top level ``test/`` to ``odl/test/`` folder and distribute them with the source. (:pull:`638`)
- Update pytest dependency to [>3.0] and use new featuers. (:pull:`653`)
- Add pytest option ``--documentation`` to test all doctest examples in the online documentation.
- Remove the ``pip install odl[all]`` option since it fails by default.


Bugfixes
--------
- Fix ``python -c "import odl; odl.test()"`` not working on Windows. (:pull:`508`)
- Fix a ``TypeError`` being raised in ``OperatorTest`` when running ``optest.ajoint()`` without specifying an operator norm. (:pull:`525`)
- Fix scaling of scikit ray transform for non full scan. (:pull:`523`)
- Fix bug causing classes to not be vectorizable. (:pull:`604`)
- Fix rounding problem in some proximals (:pull:`661`)

ODL 0.4.0 Release Notes (2016-08-17)
====================================

This release marks the addition of the ``deform`` package to ODL, adding functionality for the deformation
of ``DiscreteLp`` elements.

New features
------------
- Add ``deform`` package with linearized deformations (:pull:`488`)
- Add option to interface with ProxImaL solvers using ODL operators. (:pull:`494`)


ODL 0.3.1 Release Notes (2016-08-15)
====================================

This release mainly fixes an issue that made it impossible to ``pip install odl`` with version 0.3.0.
It also adds the first really advanced solvers based on forward-backward and Douglas-Rachford
splitting.

New features
------------
- New solvers based on the Douglas-Rachford and forward-backward splitting schemes. (:pull:`478`,
  :pull:`480`)
- ``NormOperator`` and ``DistOperator`` added. (:pull:`487`)
- Single-element ``NtuplesBase`` vectors can now be converted to ``float``, ``complex`` etc.
  (:pull:`493`)


Improvements
------------
- ``DiscreteLp.element()`` now allows non-vectorized and 1D scalar functions as input. (:pull:`476`)
- Speed improvements in the unit tests. (:pull:`479`)
- Uniformization of ``__init__()`` docstrings and many further documentation and naming improvements.
  (:pull:`489`, :pull:`482`, :pull:`491`)
- Clearer separation between attributes that are intended as part of the subclassing API and those
  that are not. (:pull:`471`)
- Chambolle-Pock solver accepts also non-linear operators and has better documentation now.
  (:pull:`490`)
- Clean-up of imports. (:pull:`492`)
- All solvers now check that the given start value ``x`` is in ``op.domain``. (:pull:`502`)
- Add test for in-place evaluation of the ray transform. (:pull:`500`)

Bugfixes
--------
- Axes in ``show()`` methods of several classes now use the correct corner coordinates, the old ones
  were off by half a grid cell in some situations. (:pull:`477`).
- Catch case in ``power_method_opnorm()`` when iteration goes to zero. (:pull:`495`)


ODL 0.3.0 Release Notes (2016-06-29)
====================================

This release marks the removal of ``odlpp`` from the core library. It has instead been moved to a separate library, ``odlcuda``.

New features
------------
- To enable cuda backends for the odl spaces, an entry point ``'odl.space'`` has been added where external libraries can hook in to add ``FnBase`` and ``NtuplesBase`` type spaces.
- Add pytest fixtures ``'fn_impl'`` and ``'ntuple_impl'`` to the test config ``conf.py``. These can now be accessed from any test.
- Allow creation of general spaces using the ``fn``, ``cn`` and ``rn`` factories. These functions now take an ``impl`` parameter which defaults to ``'numpy'`` but with odlcuda installed it may also be set to ``'cuda'``. The old numpy specific ``Fn``, ``Cn`` and ``Rn`` functions have been removed.

Changes
-------
- Move all CUDA specfic code out of the library into odlcuda. This means that ``cu_ntuples.py`` and related files have been removed.
- Rename ``ntuples.py`` to ``npy_ntuples.py``.
- Add ``Numpy`` to the numy based spaces. They are now named ``NumpyFn`` and ``NumpyNtuples``.
- Prepend ``npy_`` to all methods specific to ``ntuples`` such as weightings.

ODL 0.2.4 Release Notes (2016-06-28)
====================================

New features
------------
- Add ``uniform_discr_fromdiscr`` (:pull:`467`).
- Add conda build files (:commit:`86ff166`).

Bugfixes
--------
- Fix bug in submarine phantom with non-centered space (:pull:`469`).
- Fix crash when plotting in 1d (:commit:`3255fa3`).

Changes
-------
- Move phantoms to new module odl.phantom (:pull:`469`).
- Rename ``RectPartition.is_uniform`` to ``RectPartition.is_uniform``
  (:pull:`468`).

ODL 0.2.3 Release Notes (2016-06-12)
====================================

New features
------------
- ``uniform_sampling`` now supports the ``nodes_on_bdry`` option introduced in ``RectPartition``
  (:pull:`308`).
- ``DiscreteLpVector.show`` has a new ``coords`` option that allows to slice by coordinate instead
  of by index (:pull:`309`).
- New ``uniform_discr_fromintv`` to discretize an existing ``IntervalProd`` instance
  (:pull:`318`).
- The ``operator.oputils`` module has a new function ``as_scipy_operator`` which exposes a linear
  ODL operator as a ``scipy.sparse.linalg.LinearOperator``. This way, an ODL operator can be used
  seamlessly in SciPy's sparse solvers (:pull:`324`).
- New ``Resampling`` operator to resample data between different discretizations (:pull:`328`).
- New ``PowerOperator`` taking the power of an input function (:pull:`338`).
- First pointwise operators acting on vector fields: ``PointwiseInner`` and ``PointwiseNorm``
  (:pull:`346`).
- Examples for FBP reconstruction (:pull:`364`) and TV regularization using the Chambolle-Pock
  method (:pull:`352`).
- New ``scikit-image`` based implementation of ``RayTransform`` for 2D parallel beam tomography
  (:pull:`352`).
- ``RectPartition`` has a new method ``append`` for simple extension (:pull:`370`).
- The ODL unit tests can now be run with ``odl.test()`` (:pull:`373`).
- Proximal of the Kullback-Leibler data discrepancy functional (:pull:`289`).
- Support for SPECT using ``ParallelHoleCollimatorGeometry`` (:pull:`304`).
- A range of new proximal operators (:pull:`401`) and some calculus rules (:pull:`422`) have been added,
  e.g. the proximal of the convex conjugate or of a translated functional.
- Functions with parameters can now be sampled by passing the parameter values to the sampling
  operator. The same is true for the ``element`` method of a discrete function space (:pull:`406`).
- ``ProducSpaceOperator`` can now be indexed directly, returning the operator component(s)
  corresponding to the index (:pull:`407`).
- ``RectPartition`` now supports "almost-fancy" indexing, i.e. indexing via integer, slice, tuple
  or list in the style of NumPy (:pull:`386`).
- When evaluating a ``FunctionSetVector``, the result is tried to be broadcast if necessary
  (:pull:`438`).
- ``uniform_partition`` now has a more flexible way of initialization using ``begin``, ``end``,
  ``num_nodes`` and ``cell_sides`` (3 of 4 required) (:pull:`444`).

Improvements
------------
- Product spaces now utilize the same weighting class hierarchy as ``Rn`` type spaces, which makes
  the weight handling much more transparent and robust (:pull:`320`).
- Major refactor of the ``diagnostics`` module, with better output, improved derivative test and
  a simpler and more extensible way to generate example vectors in spaces (:pull:`338`).
- 3D Shepp-Logan phantom sliced in the middle is now exactly the same as the 2D Shepp-Logan phantom
  (:pull:`368`).
- Improved usage of test parametrization, making decoration of each test function obsolete. Also
  the printed messages are better (:pull:`371`).
- ``OperatorLeftScalarMult`` and ``OperatorRightScalarMult`` now have proper inverses (:pull:`388`).
- Better behavior of display methods if arrays contain ``inf`` or ``NaN`` (:pull:`376`).
- Adjoints of Fourier transform operators are now correctly handled (:pull:`396`).
- Differential operators now have consistent boundary behavior (:pull:`405`).
- Repeated scalar multiplication with an operator accumulates the scalars instead of creating a new
  operator each time (:pull:`429`).
- Examples have undergone a major cleanup (:pull:`431`).
- Addition of ``__len__`` at several places where it was missing (:pull:`425`).

Bugfixes
--------
- The result of the evaluation of a ``FunctionSpaceVector`` is now automatically cast to the correct
  output data type (:pull:`331`).
- ``inf`` values are now properly treated in ``BacktrackingLineSearch`` (:pull:`348`).
- Fix for result not being written to a CUDA array in interpolation (:pull:`361`).
- Evaluation of ``FunctionSpaceVector`` now works properly in the one-dimensional case
  (:pull:`362`).
- Rotation by 90 degrees / wrong orientation of 2D parallel and fan beam projectors
  and back-projectors fixed (:pull:`436`).

Changes
-------
- ``odl.set.pspace`` was moved to ``odl.space.pspace`` (:pull:`320`)
- Parameter ``ord`` in norms etc. has been renamed to ``exponent`` (:pull:`320`)
- ``restriction`` and ``extension`` operators and parameters have been renamed to ``sampling``
  and ``interpolation``, respectively (:pull:`337`).
- Differential operators like ``Gradient`` and ``Laplacian`` have been moved from
  ``odl.discr.discr_ops`` to ``odl.discr.diff_ops`` (:pull:`377`)
- The initialization patterns of ``Gradient`` and ``Divergence`` were unified to allow specification
  of domain or range or both (:pull:`377`).
- ``RawDiscretization`` and ``Discretization`` were renamed to ``DiscretizedSet`` and
  ``DiscretizedSpace``, resp. (:pull:`406`).
- Diagonal "operator matrices" are now implemented with a class ``DiagonalOperator`` instead of
  the factory function ``diagonal_operator`` (:pull:`407`).
- The ``...Partial`` classes have been renamed to ``Callback...``. Parameters of solvers are now
  ``callback`` instead of ``partial`` (:pull:`430`).
- Occurrences of ``dom`` and ``ran`` as initialization parameters of operators have been changed
  to ``domain`` and ``range`` throughout (:pull:`433`).
- Assignments ``x = x.space.element(x)`` are now required to be no-ops (:pull:`439`)


ODL 0.2.2 Release Notes (2016-03-11)
====================================

From this release on, ODL can be installed through ``pip`` directly from the Python package index.


ODL 0.2.1 Release Notes (2016-03-11)
====================================

Fix for the version number in setup.py.


ODL 0.2 Release Notes (2016-03-11)
==================================

This release features the Fourier transform as major addition, along with some minor improvements and fixes.

New Features
------------

- Add ``FourierTransform`` and ``DiscreteFourierTransform``, where the latter is the fully discrete version not accounting for shift and scaling, and the former approximates the integral transform by taking shifted and scaled grids into account. (:pull:`120`)
- The ``weighting`` attribute in ``FnBase`` is now public and can be used to initialize a new space.
- The ``FnBase`` classes now have a ``default_dtype`` static method.
- A ``discr_sequence_space`` has been added as a simple implementation of finite sequences with
  multi-indexing.
- ``DiscreteLp`` and ``FunctionSpace`` elements now have ``real`` and ``imag`` with setters as well as a
  ``conj()`` method.
- ``FunctionSpace`` explicitly handles output data type and allows this attribute to be chosen during
  initialization.
- ``FunctionSpace``, ``FnBase`` and ``DiscreteLp`` spaces support creation of a copy with different data type
  via the ``astype()`` method.
- New ``conj_exponent()`` utility to get the conjugate of a given exponent.


Improvements
------------

- Handle some not-so-unlikely corner cases where vectorized functions don't behave as they should.
  In particular, make 1D functions work when expressions like ``t[t > 0]`` are used.
- ``x ** 0`` evaluates to the ``one()`` space element if implemented.

Changes
-------

- Move `fast_1d_tensor_mult` to the ``numerics.py`` module.

ODL 0.1 Release Notes (2016-03-08)
==================================

First official release.


.. _Discrete Fourier Transform: https://en.wikipedia.org/wiki/Discrete_Fourier_transform
.. _FFTW: http://fftw.org/
.. _Fourier Transform: https://en.wikipedia.org/wiki/Fourier_transform
.. _Numpy's FFTPACK based transform: http://docs.scipy.org/doc/numpy/reference/routines.fft.html
.. _pyFFTW: https://pypi.python.org/pypi/pyFFTW
