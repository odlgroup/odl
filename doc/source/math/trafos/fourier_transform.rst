.. _fourier_transform:

#################
Fourier Transform
#################


Background
==========

Definition and basic properties
-------------------------------

The Fourier Transform (FT) of a function :math:`f` belonging to the `Lebesgue Space`_
:math:`L^1(\mathbb{R})` is defined as

.. math::
    \widehat{f}(\xi) = \mathcal{F}(f)(\xi) = (2\pi)^{-\frac{1}{2}}
    \int_{\mathbb{R}} f(x)\ e^{-i x \xi} \, \mathrm{d}x.
    :label: def_fourier     

By unique continuation, the bounded FT operator can be
`extended <https://en.wikipedia.org/wiki/Fourier_transform#On_Lp_spaces>`_ to
:math:`L^p(\mathbb{R})` for :math:`p \in [1, 2]`, yielding a mapping

.. math::
    \mathcal{F}: L^p(\mathbb{R}) \longrightarrow L^q(\mathbb{R}), \quad q = \frac{p}{p-1},

where :math:`q` is the conjugate exponent of :math:`p` (for :math:`p=1` one sets :math:`q=\infty`).
Finite exponents larger than 2 also allow the extension of the operator but require the notion of
`Distributions`_ to characterize its range. See [SW71]_ for further details.

The inverse of :math:`\mathcal{F}` on its range is given by the formula

.. math::
    \widetilde{\phi}(x) = \mathcal{F}^{-1}(\phi)(x) = (2\pi)^{-\frac{1}{2}}
    \int_{\mathbb{R}} \phi(\xi)\ e^{i \xi x}\, \mathrm{d}\xi.
    :label: def_fourier_inverse

For :math:`p = 2`, the conjugate exponent is :math:`q = 2`, and the FT is a unitary
operator on :math:`L`2(\mathbb{R})` according to `Parseval's Identity`_

.. math::
    \int_{\mathbb{R}} \lvert f(x)\rvert^2\, \mathrm{d}x =
    \int_{\mathbb{R}} \lvert \widetilde{f}(\xi) \rvert^2\, \mathrm{d}\xi,

which implies that its adjoint is its inverse, :math:`\mathcal{F}^* = \mathcal{F}^{-1}`.

Further Properties
------------------

.. math::
    \mathcal{F^{-1}}(\phi) = \mathcal{F}(\check\phi) = \mathcal{F}(\phi)(-\cdot)
    = \overline{\mathcal{F}(\overline{\phi})} = \mathcal{F}^3(\phi),
    \quad \check\phi(x) = \phi(-x),
    :label: fourier_properties

    \mathcal{F}\big(f(\cdot - b)\big)(\xi) = e^{-i b \xi} \widehat{f}(\xi),

    \mathcal{F}\big(f(a \cdot)\big)(\xi) = a^{-1} \widehat{f}(a^{-1}\xi),

    \frac{\mathrm{d}}{\mathrm{d} \xi} \widehat{f}(\xi) = \mathcal{F}(-i x f)(\xi)
    
    \mathcal{F}(f')(\xi) = i \xi \widehat{f}(\xi).
    
The first identity implies in particular that for real-valued :math:`f`, it is
:math:`\overline{\mathcal{F}(\phi)}(\xi) = \mathcal{F}(\phi)(-\xi)`, i.e. the FT is
completely known already from the its values in a half-space only. This property is later exploited
to reduce storage.

In :math:`d` dimensions, the FT is defined as

.. math::
    \mathcal{F}(f)(\xi) = (2\pi)^{-\frac{d}{2}}
    \int_{\mathbb{R}^d} f(x)\ e^{-i x^{\mathrm{T}}\xi} \, \mathrm{d}x

with the usual inner product :math:`x^{\mathrm{T}}\xi = \sum_{k=1}^d x_k \xi_k` in
:math:`\mathbb{R}^d`. The identities :eq:`fourier_properties` also hold in this case with obvious
modifications.


Discretized Fourier Transform
=============================

General case
------------

The approach taken in ODL for the discretization of the FT follows immediately from the way
:ref:`discretizations` are defined, but the original inspiration for it came from the book
[P+07]_, Section 13.9 "Computing Fourier Integrals Using the FFT".

Discretization of the Fourier transform operator means evaluating the Fourier integral
:eq:`def_fourier` on a discretized function

.. math:: f(x) = \sum_{k=0}^{n-1} f_k \phi_k(x)
    :label: discr_function

with coefficients :math:`\bar f = (f_0, \dots, f_{n-1}) \in \mathbb{C}^n` and functions
:math:`\phi_0, \dots, \phi_{n-1}`. This approach follows from the way , but can be 
We consider in particular functions generated from a single
kernel :math:`\phi` via

.. math:: \phi_k(x) = \phi\left( \frac{x - x_k}{s_k} \right),

where :math:`x_0 < \dots < x_{n-1}` are sampling points and :math:`s_k > 0` scaling factors. Using
the shift and scaling properties in :eq:`fourier_properties` yields

.. math::
    \widehat{f}(\xi) = \sum_{k=0}^{n-1} f_k \widehat{\phi_k}(\xi) =
    \sum_{k=0}^{n-1} f_k\, s_k \widehat{\phi}(s_k\xi) e^{-i x_k \xi}.
    :label: discr_fourier_general

There exist methods for the fast approximation of such sums for a general choice of frequency
samples :math:`\xi_m`, e.g. `NFFT`_.

Regular grids
-------------

For regular grids

.. math:: x_k = x_0 + ks, \quad \xi_j = \xi_0 + j\sigma,
    :label: regular_grids

the evaluation of the integral can be written in the form which uses trigonometric sums
as `computed in FFTW`_ or `in Numpy`_:

.. math:: \hat f_j = \sum_{k=0}^{n-1} f_k e^{-i 2\pi jk/n}.
    :label: fft_sum

Hence, the Fourier integral evaluation can be built around established libraries with simple pre-
and post-processing steps.

With regular grids, the discretized integral :eq:`discr_fourier_general` evaluated at
:math:`\xi = \xi_j`, can be expanded to

.. math::
    \widehat{f}(\xi_j) = s \widehat{\phi}(s\xi_j) e^{-i x_0\xi_j}
    \sum_{k=0}^{n-1} f_k\, e^{-i k s \xi_0}\, e^{-i jk s\sigma}

To reach the form :eq:`fft_sum`, the factor depending on both indices :math:`j` and :math:`k`
must agree with the corresponding factor in the FFT sum. This is achieved by setting

.. math:: \sigma = \frac{2\pi}{ns},
    :label: reciprocal_stride

finally yielding the representation

.. math::
    \hat f_j = \widehat{f}(\xi_j) = s \widehat{\phi}(s\xi_j) e^{-i x_0\xi_j}
    \sum_{k=0}^{n-1} f_k\, e^{-i k s \xi_0}\, e^{-i 2\pi jk/n}.
    :label: discr_fourier_final

Choice of :math:`\xi_0`
-----------------------

There is a certain degree of freedom in the choice of the most negative frequency :math:`\xi_0`.
Usually one wants to center the Fourier space grid around zero since most information is typically
concentrated there. Point-symmetric grids are the standard choice, however sometimes one explicitly
wants to include (for even :math:`n`) or exclude (for odd :math:`n`) the zero frequency from the
grid, which is achieved by shifting the frequency :math:`xi_0` by :math:`-\sigma/2`. This results in
two possible choices

.. math::
    \xi_{0, \mathrm{n}} = -\frac{\pi}{s} + \frac{\pi}{sn} \quad \text{(no shift)},

    \xi_{0, \mathrm{s}} = -\frac{\pi}{s} \quad \text{(shift)}.

For the shifted frequency, the pre-processing factor in the sum in
:eq:`discr_fourier_final` can be simplified to

.. math:: e^{-i k s \xi_0} = e^{i k \pi} = (-1)^k,

which is favorable for real-valued input :math:`\bar f` since this first operation preserves
this property. For half-complex transforms, shifting is required.

The factor :math:`\widehat{\phi}(s\xi_j)`
-----------------------------------------

In :eq:`discr_fourier_final`, the FT of the kernel :math:`\phi` appears as post-processing factor.
We give the explicit formulas for the two standard discretizations currently used in ODL, which
are nearest neighbor interpolation

.. math::
    \phi_{\mathrm{nn}}(x) =
    \begin{cases}
        1, & \text{if } -1/2 \leq x < 1/2, \\
        0, & \text{else,}
    \end{cases}

and linear interpolation

.. math::
    \phi_{\mathrm{lin}}(x) =
    \begin{cases}
        1 - \lvert x \rvert, & \text{if } -1 \leq x \leq 1, \\
        0, & \text{else.}
    \end{cases}

Their Fourier transforms are given by

.. math::
    \widehat{\phi_{\mathrm{nn}}}(\xi) = (2\pi)^{-1/2} \mathrm{sinc}(\xi/2),

    \widehat{\phi_{\mathrm{lin}}}(\xi) = (2\pi)^{-1/2} \mathrm{sinc}^2(\xi/2).

Since their arguments :math:`s\xi_j = s\xi_0 + 2\pi/n` lie between :math:`-\pi` and :math:`\pi`,
these functions introduce only a slight taper towards higher frequencies given the fact that the
first zeros lie at :math:`\pm 2\pi`.


Useful Wikipedia articles
=========================

- `Fourier Transform`_
- `Lebesgue Space`_
- `Distributions`_
- `Parseval's Identity`_

.. _Fourier Transform: https://en.wikipedia.org/wiki/Fourier_Transform
.. _Lebesgue Space: https://en.wikipedia.org/wiki/Lp_space
.. _Distributions: https://en.wikipedia.org/wiki/Distribution_(mathematics)
.. _Parseval's Identity: https://en.wikipedia.org/wiki/Parseval's_identity
.. _NFFT: https://github.com/NFFT/nfft
.. _computed in FFTW: http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html
.. _in Numpy: http://docs.scipy.org/doc/numpy/reference/routines.fft.html#implementation-details

.. [SW71] Stein, Elias and Weiss, Guido (1971). Introduction to
   Fourier Analysis on Euclidean Spaces. Princeton, N.J.:
   Princeton University Press. ISBN 978-0-691-08078-9

.. [P+07] Press, William H, Teukolsky, Saul A, Vetterling, William T,
   and Flannery, Brian P. *Numerical Recipes in C - The Art of
   Scientific Computing* (Volume 3). Cambridge University Press,
   2007.
