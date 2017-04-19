.. _release_notes:

.. tocdepth: 0

#############
Release Notes
#############

Upcoming release
================

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
- Added test for in-place evaluation of the ray transform. (:pull:`500`)

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
- To enable cuda backends for the odl spaces, an entry point ``'odl.space'`` has been added where external libraries can hook in to add `FnBase` and `NtuplesBase` type spaces.
- Add pytest fixtures ``'fn_impl'`` and ``'ntuple_impl'`` to the test config ``conf.py``. These can now be accessed from any test.
- Allow creation of general spaces using the ``fn``, ``cn`` and ``rn`` factories. These functions now take an ``impl`` parameter which defaults to ``'numpy'`` but with odlcuda installed it may also be set to ``'cuda'``. The old numpy specific ``Fn``, ``Cn`` and ``Rn`` functions have been removed.

Changes
-------
- Moved all CUDA specfic code out of the library into odlcuda. This means that ``cu_ntuples.py`` and related files have been removed.
- Rename ``ntuples.py`` to ``npy_ntuples.py``.
- Added ``Numpy`` to the numy based spaces. They are now named ``NumpyFn`` and ``NumpyNtuples``.
- Prepended ``npy_`` to all methods specific to ``ntuples`` such as weightings.

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
- The ``weighting`` attribute in `FnBase` is now public and can be used to initialize a new space.
- The `FnBase` classes now have a ``default_dtype`` static method.
- A `discr_sequence_space` has been added as a simple implementation of finite sequences with
  multi-indexing.
- `DiscreteLp` and `FunctionSpace` elements now have ``real`` and ``imag`` with setters as well as a
  ``conj()`` method.
- `FunctionSpace` explicitly handles output data type and allows this attribute to be chosen during
  initialization.
- `FunctionSpace`, `FnBase` and `DiscreteLp` spaces support creation of a copy with different data type
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
