[![PyPI version](https://badge.fury.io/py/odl.svg)](https://badge.fury.io/py/odl)
[![Build Status](https://travis-ci.org/odlgroup/odl.svg?branch=master)](https://travis-ci.org/odlgroup/odl?branch=master)
[![Documentation Status](https://readthedocs.org/projects/odl/badge/?version=latest)](http://odl.readthedocs.org/?badge=latest)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](http://opensource.org/licenses/GPL-3.0)

# ODL

Operator Discretization Library (ODL) is a python library for fast prototyping focusing on (but not restricted to) inverse problems. ODL is being developed at [KTH, Royal Institute of Technology](https://www.kth.se/en/sci/institutioner/math).

The main intent of ODL is to enable mathematicians and applied scientists to use different numerical methods on real-world problems without having to implement all necessary parts from the bottom up.
This is reached by an `Operator` structure which encapsulates all application-specific parts, and a high-level formulation of solvers which usually expect an operator, data and additional parameters.
The main advantages of this approach is that

1. Different problems can be solved with the same method (e.g. TV regularization) by simply switching operator and data.
2. The same problem can be solved with different methods by simply calling into different solvers.
3. Solvers and application-specific code need to be written only once, in one place, and can be tested individually.
4. Adding new applications or solution methods becomes a much easier task.



Features
========

- Efficient and well-tested data containers based on [NumPy](https://github.com/numpy/numpy) (default) or CUDA (optional)
- Objects to represent mathematical notions like vector spaces and operators including properties as expected from mathematics (inner product, norm, operator composition, ...)
- Convenience functionality for operators like arithmetic, composition, operator matrices etc., which satisfy the known mathematical rules.
- Out-of-the-box support for frequently used operators like scaling, partial derivative, gradient, Fourier transform etc.
- Support for tomographic imaging with a unified geometry representation and bindings to external libraries for efficient computation of projections and back-projections.
- Standardized tests to validate implementations against expected behavior of the corresponding mathematical object, e.g. if a user-defined norm satisfies `norm(x + y) <= norm(x) + norm(y)` for a number of input vectors `x` and `y`.

Installation
============

Requirements
------------

First, you need to [install pip](https://pip.pypa.io/en/stable/installing/#installation) to be able to install Python packages from the [Python package index](https://pypi.python.org/pypi). Usually, `pip` is bundled with Python, so there is no action required. Check if simply running `pip` shows a help message or an error.

**General remark:** Try to install as many dependencies as possible through your distribution package manager (on Linux) or with a complete scientific python distribution like [Anaconda](https://www.continuum.io/anaconda). Packages installed through `pip` can cause conflicts with identical packages installed by a different mechanism (our basic dependencies are largely unproblematic, though).

#### Minimal dependencies

- [numpy](https://github.com/numpy/numpy) >= 1.9
- [scipy](https://github.com/scipy/scipy) >= 0.14
- [python-future](https://pypi.python.org/pypi/future/) >= 0.14

These packages are automatically installed through `pip` alongside ODL if they are not found on the system or the version requirement is not met.


Installing ODL through `pip`
----------------------------

For basic installation without extra dependencies, run

    pip install odl

If you don't have administrator rights on your machine, you can install packages locally as a user:

    pip install --user odl

#### Extra dependencies

Additional dependencies can be specified in square brackets, separated by commas (no spaces!):

    pip install odl[dep1,dep2]

Possible choices:

- `all` : Install with all extra dependencies.
- `show` : Install [matplotlib](http://matplotlib.org/) to enable displaying capabilities.
- `fft` : Install [pyFFTW](https://pypi.python.org/pypi/pyFFTW) for fast Fourier transforms. Note that this requires the [FFTW](http://fftw.org/) library to be available on your system.
  Note also that even without this dependency, FFTs can be computed with Numpy's FFT library.
- `wavelets` : Install [PyWavelets](https://pypi.python.org/pypi/PyWavelets) for wavelet transforms.

Installing from source
----------------------
This installation method is intended for developers who want to make changes to the code. It assumes that the [Git](http://www.git-scm.com/) version control system is available on your system.

To get ODL, clone the repository with the command

    git clone <repo_uri>

where `<repo_uri>` is the clone link from the top navigation bar on the [ODL GitHub page](https://github.com/odlgroup/odl). No GitHub account is required for this step.

For installation in a local user folder, enter the top-level directory of the cloned repository and run

    pip install --user --editable .

**Don't forget the "." (dot) at the end** - it refers to the current directory, the location from where `pip` is supposed to install ODL.

We recommend the `--editable` option (can be shortened to `-e`) since it installs a link instead of copying the files to your Python packages location. This way local changes to the code (e.g. after a `git pull`) take immediate effect without reinstall.

If you prefer a system-wide installation, leave out the `--user` option. To do this, you need administrator rights.

#### Extra dependencies

You can give additional dependencies separated by commas in square brackets right after the ".":

    pip install --user -e .[dep1,dep2]

The simplest choice is `all`, which pulls all dependencies available through `pip`. See above for a list of
possible values.

Developers have an additional choice `testing`, which installs all dependencies related to unit
testing. See the next section.

Running unit tests
------------------
Unit tests in ODL are based on [pytest](https://pypi.python.org/pypi/pytest), and coverage reports are created by the [coverage](https://pypi.python.org/pypi/coverage/) module. These packages are installed if you run

    pip install --user -e .[testing]

Now you can check that everything was installed properly by running

    py.test

Compiled extensions
===================

CUDA backend for linear arrays
------------------------------
If you also wish to use the (optional) CUDA extensions you need to run

    git submodule update --init --recursive
    cd odlpp

From here follow the instructions in [odlpp](https://github.com/odlgroup/odlpp) to build and install it.

ASTRA for X-ray tomography
--------------------------
To calculate forward and backward projections for image reconstruction in X-ray tomography, install the
[ASTRA tomography toolbox](https://github.com/astra-toolbox/astra-toolbox). ASTRA projectors are fully supported
in ODL.

STIR for emission tomography
----------------------------
For applications in emission tomography, i.e. PET or SPECT, install [STIR](https://github.com/UCL/STIR). Support
for STIR is currently very limited.


List of optional dependencies
=============================

- [matplotlib](http://matplotlib.org/) for plotting.
- [pyFFTW](https://pypi.python.org/pypi/pyFFTW) for fast Fourier transforms via FFTW.
- [PyWavelets](https://pypi.python.org/pypi/PyWavelets) for wavelet transforms.
- [odlpp](https://github.com/odlgroup/odlpp) for CUDA support (not required for ASTRA GPU projectors).
- [ASTRA](https://github.com/astra-toolbox/astra-toolbox) >= 1.7.1 for X-ray tomography support.
- [STIR](https://github.com/UCL/STIR) for emission tomography support.
- [pytest](https://pypi.python.org/pypi/pytest) >= 2.7.0 for unit tests
- [coverage](https://pypi.python.org/pypi/coverage/) >= 4.0.0 for test coverage report

Compatibility
=============
The code is compatible to Python 2 and 3 through the `future` library. It is intended to work on all major platforms (GNU/Linux / Mac / Windows).

Current status (2016-03-11) is

| Platform     | Python | Works | CUDA  |
|--------------|--------|-------|-------|
| Windows 7    | 2.7    | ✔     | ✔     |
| Windows 10   | 2.7    | ✔     | ✔     |
| Ubuntu 14.04 | 2.7    | ✔     | ✔     |
| Fedora 22    | 2.7    | ✔     | x (1) |
| Fedora 22    | 3.4    | ✔     | x (1) |
| Mac OSX      | 3.5    | ✔     | ??    |

(1) The GCC 5.x compiler is not compatible with current CUDA (7.5)

License
=======
GPL Version 3 or later. See LICENSE file.

If you would like to get the code under a different license, please contact the developers.

ODL development group
=====================

##### Main developers

- Jonas Adler (@adler-j)
- Holger Kohr (@kohr-h)

##### Further core developers

- Julian Moosmann (@moosmann)
- Kati Niinimäki (@niinimaki)
- Axel Ringh (@aringh)
- Ozan Öktem (@ozanoktem)


Funding
=======

Development of ODL is financially supported by the Swedish Foundation for Strategic Research as part of the project "Low complexity image reconstruction in medical imaging".
