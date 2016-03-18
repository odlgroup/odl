[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](http://opensource.org/licenses/GPL-3.0)
[![Build Status](https://travis-ci.org/odlgroup/odl.svg?branch=master)](https://travis-ci.org/odlgroup/odl?branch=master)
[![Coverage Status](https://coveralls.io/repos/odlgroup/odl/badge.svg?branch=master&service=github)](https://coveralls.io/github/odlgroup/odl?branch=master)
[![Documentation Status](https://readthedocs.org/projects/odl/badge/?version=latest)](http://odl.readthedocs.org/?badge=latest)

ODL
===

Operator Discretization Library (ODL) is a python library for fast prototyping focusing on (but not restricted to) inverse problems. ODL is being developed at [KTH, Royal Institute of Technology](https://www.kth.se/en/sci/institutioner/math).

The main intent of ODL is to enable mathematicians and applied scientists to use different numerical methods on real-world problems without having to implement all necessary parts from the bottom up.
This is reached by an `Operator` structure which encapsulates all application-specific parts, and a high-level formulation of solvers which usually expect an operator, data and additional parameters.
The main advantages of this approach is that

1. Different problems can be solved with the same method (e.g. TV regularization) by simply switching operator and data.
2. The same problem can be solved with different methods by simply calling into different solvers.
3. Solvers and application-specific code needs to be written only once, in one place, and can be tested individually.
4. Adding new applications or solution methods becomes a much easier task.



Features
--------

- Efficient and well-tested data containers based on
  [NumPy](https://github.com/numpy/numpy) (default) or CUDA (optional)
- Objects to represent mathematical notions like vector spaces and operators including
  properties as expected from mathematics (inner product, norm, operator composition, ...)
- Standardized tests to validate implementations against expected behavior of the
  corresponding mathematical object, e.g. if a user-defined norm satisfies
  `norm(x + y) <= norm(x) + norm(y)` for a number of input vectors `x` and `y`
- Convenience functionality for operators like arithmetic, composition, operator matrices etc.,
  which satisfy the known mathematical rules

Installation
------------

To get ODL, clone the repository with the command

```
user$ git clone <repo>
```

where `<repo>` is the clone link from the right navigation bar on the [ODL GitHub page](https://github.com/odlgroup/odl).

For installation in a local user folder, go to the ODL folder and run

```
user$ pip install --user -e .
```

in the top level folder of the package. For system-wide installation, run (as root)

```
root# pip install -e .
```

We recommend to use the `-e` option since it installs a link instead of copying the files to
your Python packages location. This way local changes to the code (e.g. after a `git pull`) take
immediate effect without reinstall.

After installing, you can check that everything was installed properly by running

```
user$ py.test
```

To do this, you need the [pytest](https://pypi.python.org/pypi/pytest) package installed. You can install ODL with the extra `testing` dependency, i.e. invoke

```
pip install --user -e .[testing]
```
to get `pytest` installed as a dependency.

##### CUDA

If you also wish to use the (optional) CUDA extensions you need to run

```
user$ git submodule update --init --recursive
user$ cd odlpp
```

From here follow the instructions in [odlpp](https://github.com/odlgroup/odlpp) and install it.

Requirements
------------

- [setuptools](https://pypi.python.org/pypi/setuptools) needed for installation.
- [numpy](https://github.com/numpy/numpy) >= 1.9
- [scipy](https://github.com/scipy/scipy) >= 0.14
- [python-future](https://pypi.python.org/pypi/future/) >= 0.14
- [matplotlib](http://matplotlib.org/) for plotting.

Optional
--------

- [odlpp](https://github.com/odlgroup/odlpp) for GPU support.
- [pytest](https://pypi.python.org/pypi/pytest) >= 2.7.0 for unit tests
- [coverage](https://pypi.python.org/pypi/coverage/) >= 4.0.0 for test coverage report

Compatibility
-------------
The code is written in python 2/3 through the `future` library. It is intended to work on all major platforms (GNU/Linux / Mac / Windows).

Current status (2015-11-05) is

| Platform     | Python | Works | CUDA  |
|--------------|--------|-------|-------|
| Windows 7    | 2.7    | ✔     | ✔     |
| Ubuntu 14.04 | 2.7    | ✔     | ✔     |
| Fedora 22    | 2.7    | ✔     | x (1) |
| Fedora 22    | 3.4    | ✔     | x (1) |
| Mac OSX      | ??     | ??    | ??    |

(1) The GCC 5.x compiler is not compatible with current CUDA (7.5)

License
-------

GPL Version 3. See LICENSE file.

If you would like to get the code under a different license, please contact the
developers.

ODL development group
---------------------

##### Main developers

- Jonas Adler (@adler-j)
- Holger Kohr (@kohr-h)

##### Further core developers

- Julian Moosmann (@moosmann)
- Kati Niinimäki (@niinimaki)
- Axel Ringh (@aringh)
- Ozan Öktem (@ozanoktem)


Funding
-------

Development of ODL is financially supported by the Swedish Foundation for Strategic Research as part of the project "Low complexity image reconstruction in medical imaging".
