.. _release_notes:

.. tocdepth: 0

#############
Release Notes
#############

ODL 0.3.0 Release Notes (2016-06-29)
====================================

This release marks the removal of odlpp from the core library. It has instead been moved to a separate library, odlcuda.

New features
------------
- To enable cuda backends for the odl spaces, an entry point ``'odl.space'`` has been added where external libraries can hook in to add `FnBase` and `NtuplesBase` type spaces.
- Add pytest fixtures ``'fn_impl'`` and ``'ntuple_impl'`` to the test config ``conf.py``. These can now be accessed from any test.
- Allow creation of general spaces using the ``fn``, ``cn`` and ``rn`` methods. This functions now take an ``impl`` parameter which defaults to ``'numpy'`` but with odlcuda installed it may also be set to ``'cuda'``. The old numpy specific ``Fn``, ``Cn`` and ``Rn`` functions have been removed.

Changes
-------
- Moved all CUDA specfic code out of the library into odlcuda. This means that ``cu_ntuples.py`` and related files have been removed.
- rename ``ntuples.py`` to ``npy_ntuples.py``.
- Added ``Numpy`` to the numy based spaces. They are now named ``NumpyFn`` and ``NumpyNtuples``.
- Prepended ``npy_`` to all methods specific to ``ntuples`` such as weightings.

ODL 0.2.4 Release Notes (2016-06-28)
====================================

New features
------------
- Add ``uniform_discr_fromdiscr`` (`PR 467`).
- Add conda build files (`commit 86ff166`).

Bugfixes
--------
- Fix bug in submarine phantom with non-centered space (`PR 469`).
- Fix crash when plotting in 1d (`commit 3255fa3`).

Changes
-------
- Move phantoms to new module odl.phantom (`PR 469`).
- Rename ``RectPartition.is_uniform`` to ``RectPartition.is_uniform``
  (`PR 468`).

ODL 0.2.3 Release Notes (2016-06-12)
====================================

New features
------------
- ``uniform_sampling`` now supports the ``nodes_on_bdry`` option introduced in ``RectPartition``
  (`PR 308`_).
- ``DiscreteLpVector.show`` has a new ``coords`` option that allows to slice by coordinate instead
  of by index (`PR 309`_).
- New ``uniform_discr_fromintv`` to discretize an existing ``IntervalProd`` instance
  (`PR 318`_).
- The ``operator.oputils`` module has a new function ``as_scipy_operator`` which exposes a linear
  ODL operator as a ``scipy.sparse.linalg.LinearOperator``. This way, an ODL operator can be used
  seamlessly in SciPy's sparse solvers (`PR 324`_).
- New ``Resampling`` operator to resample data between different discretizations (`PR 328`_).
- New ``PowerOperator`` taking the power of an input function (`PR 338`_).
- First pointwise operators acting on vector fields: ``PointwiseInner`` and ``PointwiseNorm``
  (`PR 346`_).
- Examples for FBP reconstruction (`PR 364`_) and TV regularization using the Chambolle-Pock
  method (`PR 352`_).
- New ``scikit-image`` based implementation of ``RayTransform`` for 2D parallel beam tomography
  (`PR 352`_).
- ``RectPartition`` has a new method ``append`` for simple extension (`PR 370`_).
- The ODL unit tests can now be run with ``odl.test()`` (`PR 373`_).
- Proximal of the Kullback-Leibler data discrepancy functional (`PR 289`_).
- Support for SPECT using ``ParallelHoleCollimatorGeometry`` (`PR 304`_).
- A range of new proximal operators (`PR 401`_) and some calculus rules (`PR 422`_) have been added,
  e.g. the proximal of the convex conjugate or of a translated functional.
- Functions with parameters can now be sampled by passing the parameter values to the sampling
  operator. The same is true for the ``element`` method of a discrete function space (`PR 406`_).
- ``ProducSpaceOperator`` can now be indexed directly, returning the operator component(s)
  corresponding to the index (`PR 407`_).
- ``RectPartition`` now supports "almost-fancy" indexing, i.e. indexing via integer, slice, tuple
  or list in the style of NumPy (`PR 386`_).
- When evaluating a ``FunctionSetVector``, the result is tried to be broadcast if necessary
  (`PR 438`_).
- ``uniform_partition`` now has a more flexible way of initialization using ``begin``, ``end``,
  ``num_nodes`` and ``cell_sides`` (3 of 4 required) (`PR 444`_).

Improvements
------------
- Product spaces now utilize the same weighting class hierarchy as ``Rn`` type spaces, which makes
  the weight handling much more transparent and robust (`PR 320`_).
- Major refactor of the ``diagnostics`` module, with better output, improved derivative test and
  a simpler and more extensible way to generate example vectors in spaces (`PR 338`_).
- 3D Shepp-Logan phantom sliced in the middle is now exactly the same as the 2D Shepp-Logan phantom
  (`PR 368`_).
- Improved usage of test parametrization, making decoration of each test function obsolete. Also
  the printed messages are better (`PR 371`_).
- ``OperatorLeftScalarMult`` and ``OperatorRightScalarMult`` now have proper inverses (`PR 388`_).
- Better behavior of display methods if arrays contain ``inf`` or ``NaN`` (`PR 376`_).
- Adjoints of Fourier transform operators are now correctly handled (`PR 396`_).
- Differential operators now have consistent boundary behavior (`PR 405`_).
- Repeated scalar multiplication with an operator accumulates the scalars instead of creating a new
  operator each time (`PR 429`_).
- Examples have undergone a major cleanup (`PR 431`_).
- Addition of ``__len__`` at several places where it was missing (`PR 425`_).

Bugfixes
--------
- The result of the evaluation of a ``FunctionSpaceVector`` is now automatically cast to the correct
  output data type (`PR 331`_).
- ``inf`` values are now properly treated in ``BacktrackingLineSearch`` (`PR 348`_).
- Fix for result not being written to a CUDA array in interpolation (`PR 361`_).
- Evaluation of ``FunctionSpaceVector`` now works properly in the one-dimensional case
  (`PR 362`_).
- Rotation by 90 degrees / wrong orientation of 2D parallel and fan beam projectors
  and back-projectors fixed (`PR 436`_).

Changes
-------
- ``odl.set.pspace`` was moved to ``odl.space.pspace`` (`PR 320`_)
- Parameter ``ord`` in norms etc. has been renamed to ``exponent`` (`PR 320`_)
- ``restriction`` and ``extension`` operators and parameters have been renamed to ``sampling``
  and ``interpolation``, respectively (`PR 337`_).
- Differential operators like ``Gradient`` and ``Laplacian`` have been moved from
  ``odl.discr.discr_ops`` to ``odl.discr.diff_ops`` (`PR 377`_)
- The initialization patterns of ``Gradient`` and ``Divergence`` were unified to allow specification
  of domain or range or both (`PR 377`_).
- ``RawDiscretization`` and ``Discretization`` were renamed to ``DiscretizedSet`` and
  ``DiscretizedSpace``, resp. (`PR 406`_).
- Diagonal "operator matrices" are now implemented with a class ``DiagonalOperator`` instead of
  the factory function ``diagonal_operator`` (`PR 407`_).
- The ``...Partial`` classes have been renamed to ``Callback...``. Parameters of solvers are now
  ``callback`` instead of ``partial`` (`PR 430`_).
- Occurrences of ``dom`` and ``ran`` as initialization parameters of operators have been changed
  to ``domain`` and ``range`` throughout (`PR 433`_).
- Assignments ``x = x.space.element(x)`` are now required to be no-ops (`PR 439`_)


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

.. include:: prs.inc
