[![Build Status](https://travis-ci.org/odlgroup/odl.svg?branch=master)](https://travis-ci.org/odlgroup/odl?branch=master)
[![Coverage Status](https://coveralls.io/repos/odlgroup/odl/badge.svg?branch=master&service=github)](https://coveralls.io/github/odlgroup/odl?branch=master)

ODL
===

Operator Discretization Library (ODL) is a python library for fast prototyping focusing on (but not restricted to) inverse problems. ODL is being developed at [KTH, Royal Institute of Technology](https://www.kth.se/en/sci/institutioner/math).

The main intent of ODL is to enable mathematicians and applied scientists to use different numerical methods on real-world problems without having to implement all necessary parts from the bottom up.
ODL provides some of the most heavily used building blocks for numerical algorithms out of the box, which enables users to focus on real scientific issues.

Features
--------

- Efficient and well-tested data containers based on
  [NumPy](https://github.com/numpy/numpy) (default) or CUDA (optional)
- Objects to represent mathematical notions like vector spaces and operators including
  properties as expected from mathematics (inner product, norm, operator composition, ...)
- Standardized tests to validate implementations against expected behavior of the
  corresponding mathematical object, e.g. if a user-defined norm satisfies
  `norm(x + y) <= norm(x) + norm(y)` for a number of input vectors `x` and `y`.

Installation
------------

To get ODL, clone the repository with the command

```sh
user$ git clone <repo>
```

where `<repo>` is the clone link from the right navigation bar on the [ODL GitHub page](https://github.com/odlgroup/odl).

For installation in a local user folder, go to the ODL folder and run

```sh
user$ pip install --user .
```

in the root folder of the package. For system-wide installation, run (as root)

```sh
root# pip install .
```

If you intend to make changes to the code, you should add the `--editable` option to the `pip` command.
This way, a link to your working directory is installed in your Python installation site rather than a copy of the code, which makes local changes take immediate effect.

After installing, you should check that everything was installed properly by running

```sh
user$ python run_tests.py
```

##### CUDA

If you also wish to use the (optional) CUDA extensions you need to run

```sh
user$ git submodule update --init --recursive
user$ cd odlpp
```

From here follow the instructions in [odlpp](https://github.com/odlgroup/odlpp) and install it. You then need to re-install ODL.

Requirements
------------

- [numpy](https://github.com/numpy/numpy) >= 1.8
- [scipy](https://github.com/scipy/scipy) >= 0.14
- [python-future](https://pypi.python.org/pypi/future/)

Optional
--------

- [odlpp](https://github.com/odlgroup/odlpp) for GPU support (optional).
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

Main developers
---------------

- Jonas Adler (jonas<ätt>kth<dot>se)
- Holger Kohr (kohr<ätt>kth<dot>se)
