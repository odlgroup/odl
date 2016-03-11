#############
Release Notes
#############


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
