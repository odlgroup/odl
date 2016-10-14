[![PyPI version](https://badge.fury.io/py/odl.svg)](https://badge.fury.io/py/odl)
[![Build Status](https://travis-ci.org/odlgroup/odl.svg?branch=master)](https://travis-ci.org/odlgroup/odl?branch=master)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](http://opensource.org/licenses/GPL-3.0)

ODL
===

Operator Discretization Library (ODL) is a Python library for fast prototyping focusing on (but not restricted to) inverse problems. ODL is being developed at [KTH Royal Institute of Technology](https://www.kth.se/en/sci/institutioner/math).

The main intent of ODL is to enable mathematicians and applied scientists to use different numerical methods on real-world problems without having to implement all necessary parts from the bottom up.
This is reached by an `Operator` structure which encapsulates all application-specific parts, and a high-level formulation of solvers which usually expect an operator, data and additional parameters.
The main advantages of this approach are that

1. Different problems can be solved with the same method (e.g. TV regularization) by simply switching operator and data.
2. The same problem can be solved with different methods by simply calling into different solvers.
3. Solvers and application-specific code need to be written only once, in one place, and can be tested individually.
4. Adding new applications or solution methods becomes a much easier task.

Features
========

- Efficient and well-tested data containers based on [NumPy](https://github.com/numpy/numpy) (default) or CUDA (optional)
- Objects to represent mathematical notions like vector spaces and operators, including properties as expected from mathematics (inner product, norm, operator composition, ...)
- Convenience functionality for operators like arithmetic, composition, operator matrices etc., which satisfy the known mathematical rules.
- Out-of-the-box support for frequently used operators like scaling, partial derivative, gradient, Fourier transform etc.
- Support for tomographic imaging with a unified geometry representation and bindings to external libraries for efficient computation of projections and back-projections.
- Standardized tests to validate implementations against expected behavior of the corresponding mathematical object, e.g. if a user-defined norm satisfies `norm(x + y) <= norm(x) + norm(y)` for a number of input vectors `x` and `y`.

Documentation
=============

Introduction, mathematical details and the API reference can be found at the [ODL documentation page](https://odlgroup.github.io/odl/).


Installation
============
For basic installation without extra dependencies, run

    pip install odl

You can check that everything was installed properly by running

    python -c "import odl; odl.test()"

This requires [pytest](http://pytest.org/latest/). See the [installation](http://odlgroup.github.io/odl/guide/introduction/installing.html) documentation for further information.


Optional Dependencies
---------------------
Several ODL features work better with external packages installed. This is a list of optional dependencies and links for further information. Some of the packages can be installed along ODL through pip by calling

    pip install odl[FOO, BAR, ...]

where FOO, BAR, etc are the dependencies. Others require more complicated installation procedures, in that case, consult the respective package for instructions.

| Package | Purpose | Dependency |
|---------|---------|------------|
| [odlcuda](https://github.com/odlgroup/odlcuda/)   | CUDA backend |  |
| [ASTRA](https://github.com/astra-toolbox/astra-toolbox)    | Tomographic forward/backward projectors in [RayTransform](http://odlgroup.github.io/odl/generated/odl.tomo.operators.ray_trafo.RayTransform.html) |  |
| [scikit-image](http://scikit-image.org/)    | 2D parallel beam forward/backward projectors in [RayTransform](http://odlgroup.github.io/odl/generated/odl.tomo.operators.ray_trafo.RayTransform.html) | scikit |
| [FFTW](https://github.com/pyFFTW/pyFFTW)   | Accelerated [FourierTransform](http://odlgroup.github.io/odl/generated/odl.trafos.fourier.FourierTransform.html) | fftw |
| [PyWavelets](https://github.com/PyWavelets/pywt)   | Computation of the  [WaveletTransform](http://odlgroup.github.io/odl/generated/odl.trafos.wavelet.WaveletTransform.html) | pywavelets |
| [matplotlib](http://matplotlib.org/)   | Visualization through the [show](http://odlgroup.github.io/odl/generated/odl.discr.lp_discr.DiscreteLpElement.show.html) command | show |
| [proximal](http://github.com/comp-imaging/ProxImaL)   | Solution of some convex optimization problems | proximal |
| [pytest](http://pytest.org/latest/)   | Unit tests | testing |


Compatibility
-------------
ODL is compatible to Python 2 and 3 through the `future` library. It is intended to work on all major platforms (GNU/Linux / Mac / Windows).

Currently (2016-06-10) known working combinations are:

| Platform     | Python          |
|--------------|-----------------|
| Windows 7    | 2.7             |
| Windows 10   | 2.7             |
| Ubuntu 14.04 | 2.7             |
| Ubuntu 15.10 | 2.7 / 3.4 / 3.5 |
| Ubuntu 16.04 | 2.7 / 3.4 / 3.5 |
| Fedora 22    | 2.7 / 3.4       |
| Mac OSX      | 3.5             |

License
-------
GPL Version 3 or later. See [LICENSE](LICENSE) file.

If you would like to get the code under a different license, please contact the developers.

ODL development group
---------------------
To contact the developers either [write an issue on github](https://github.com/odlgroup/odl/issues/new) or send an email to odl@math.kth.se

##### Main developers
- Jonas Adler ([@adler-j](https://github.com/adler-j))
- Holger Kohr ([@kohr-h](https://github.com/kohr-h))

##### Contributors
- Ozan Öktem ([@ozanoktem](https://github.com/ozanoktem))
- Kati Niinimäki ([@niinimaki](https://github.com/niinimaki))
- Axel Ringh ([@aringh](https://github.com/aringh))
- Julian Moosmann ([@moosmann](https://github.com/moosmann))
- Chong Chen ([@chongchenmath](https://github.com/chongchenmath))
- Willem Jan Palenstijn ([@wjp](https://github.com/wjp))

Funding
-------
Development of ODL is financially supported by the Swedish Foundation for Strategic Research as part of the project "Low complexity image reconstruction in medical imaging".

Some development time has also been financed by [Elekta](https://www.elekta.com/).
