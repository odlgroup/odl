.. _release_notes:

.. tocdepth: 0

#############
Release Notes
#############

Next release
============

New features
------------
- Add ``ResizingOperator`` for shrinking and extending (padding) of discretized functions, including a variety of padding methods. (:pull:`499`)
- Add ``as_writable_array`` that allows casting arbitrary array-likes to a numpy array and then storing the results later on. This is
  intended to be used with odl vectors that may not be stored in numpy format (like cuda vectors), but can be used with other types like lists.
  (:pull:`524`)
- Allow ASTRA backend to be used with arbitrary dtypes. (:pull:`524`)
- Add ``reset`` to ``SolverCallback`` that resets the callback to its initial state. (:issue:`552`)
- Add ``nonuniform_partition`` utility that creates a partition with non-uniformly spaced points.
  This is useful e.g. when the angles of a tomography problem are not exactly uniform.

Improvements
------------
- Add intelligence to ``power_method_opnorm`` so it can terminate early by checking if consecutive iterates are close. (:pull:`527`)
- Add ``BroadcastOperator(op, n)``, ``ReductionOperator(op, n)`` and ``DiagonalOperator(op, n)`` syntax.
  This is equivalent to ``BroadcastOperator(*([op] * n))`` etc, i.e. create ``n`` copies of the operator. (:pull:`532`)

Changes
--------
- Changed definition of ``LinearSpaceVector.multiply`` to match the definition used by Numpy. (:pull:`509`)
- The parameters ``padding_method`` in ``diff_ops.py`` and ``mode`` in ``wavelet.py`` have been renamed to ``pad_mode``.
  The parameter ``padding_value`` is now called ``pad_const``. (:pull:`511`)
- Expose ``ellipse_phantom`` and ``shepp_logan_ellipses`` to ``odl.phantom``. (:pull:`529`)
- Unify the names of minimum (``min_pt``), maximum (``max_pt``) and middle (``mid_pt``) points as well as number of points (``shape``) in grids, interval products and factory functions for discretized spaces. (:pull:`541`)
- Removed ``simple_operator`` since it was never used and did not follow the ODL style. (:pull:`543`)
  The parameter ``padding_value`` is now called ``pad_const``.
- Removed ``Interval``, ``Rectangle`` and ``Cuboid`` since they were confusing (Capitalized name but not a Cunction) and barely ever used.
  Users should instead use ``IntervalProd`` in all cases. (:pull:`537`)
- The following classes have been renamed:
  * `LinearSpaceVector` -> `LinearSpaceElement`
  * `DiscreteLpVector` -> `DiscreteLpElement`
  * `ProductSpaceVector` -> `ProductSpaceElement`
  * `DiscretizedSetVector` -> `DiscretizedSetElement`
  * `DiscretizedSpaceVector` -> `DiscretizedSpaceElement`
  * `FunctionSetVector` -> `FunctionSetElement`
  * `FunctionSpaceVector` -> `FunctionSpaceElement`
- Changed parameter style of differential operators from having a `pad_mode` and a separate `edge_order` argument that were mutually exclusive to a single `pad_mode` that covers all cases. Also Added several new pad modes to the differential operators. (:pull:`548`)

Bugfixes
--------
- Fixed ``python -c "import odl; odl.test()"`` not working on Windows. (:pull:`508`)
- Fixed a ``TypeError`` being raised in ``OperatorTest`` when running ``optest.ajoint()`` without specifying an operator norm. (:pull:`525`)


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

This release adds the functionality of the **Fourier Transform** in arbitrary dimensions. The
operator comes in two different flavors: the "bare", trigonometric-sum-only
`Discrete Fourier Transform`_ and the discretization of the continuous `Fourier Transform`_.

New Features
------------

Fourier Transform (FT)
~~~~~~~~~~~~~~~~~~~~~~

The FT is an :term:`operator` mapping a function to its transformed version (shown for 1d):

.. math::
    \widehat{f}(\xi) = \mathcal{F}(f)(\xi) = (2\pi)^{-\frac{1}{2}}
    \int_{\mathbb{R}} f(x)\ e^{-i x \xi} \, \mathrm{d}x, \quad \xi\in\mathbb{R}.

This implementation acts on discretized functions and accounts for scaling and shift of the
underlying grid as well as the type of discretization used. Supported backends are `Numpy's
FFTPACK based transform`_ and `pyFFTW`_ (Python wrapper for `FFTW`_). The implementation has full
support for the wrapped backends, including

- Forward and backward transforms,
- Half-complex transfroms, i.e. real-to-complex transforms where roughly only half of the
  coefficients need to be stored,
- Partial transforms along selected axes,
- Computation of efficient FFT plans (pyFFTW only).

Discrete Fourier Transform (DFT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This operator merely calculates the trigonometric sum

.. math::
    \hat f_j = \sum_{k=0}^{n-1} f_k\, e^{-i 2\pi jk/n},\quad j=0, \dots, n-1

without accounting for shift and scaling of the underlying grid. It supports the same features of
the wrapped backends as the FT.

Further additions
~~~~~~~~~~~~~~~~~

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
  The main issue was the way Python 2 treats comparisons of tuples against scalars (Python 3 raises
  an exception which is correctly handled by the subsequent code). In Python 2, the following
  happens::

    >>> t = ()
    >>> t > 0
    True
    >>> t = (-1,)
    >>> t > 0
    True

  This is especially unfortunate if used as ``t[t > 0]`` in 1d functions, when ``t`` is a
  :term:`meshgrid` sequence (of 1 element). In this case, ``t > 0`` evaluates to ``True``, which
  is treated as ``1`` in the index expression, which in turn will raise an ``IndexError`` since the
  sequence has only length one. This situation is now properly caught.

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
