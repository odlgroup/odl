ODL
===

Operator Discretization Library (ODL) is a python library for inverse problems developed at KTH, Royal Institute of Technology.

The main intent of ODL is to enable mathematicians and applied scientists to use different reconstruction methods on real-world problems without having to implement all necessary parts from the bottom up.
ODL provides some of the most heavily used building blocks for reconstruction algorithms out of the box, which enables users to focus on real scientific issues.

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

```bash
$ git clone <repo>
```

where `<repo>` is the clone link from the right navigation bar on the [ODL GitHub page](https://gits-15.sys.kth.se/LCR/ODL).

For installation in a local user folder, go to the ODL folder and run

```bash
$ pip install --user .
```

in the root folder of the package. For system-wide installation, run (as root)

```bash
/# pip install .
```

If you intend to make changes to the code, you should add the `--editable` option to the `pip` command.
This way, a link to your working directory is installed in your Python installation site rather than a copy of the code, which makes local changes take immediate effect.

After installing, you should check that everything was installed properly by running

```
$ python run_tests.py
```

##### CUDA

If you also wish to use the (optional) CUDA extensions you need to run

```bash
$ git submodule update --init --recursive
$ cd odlpp
```

From here follow the instructions in [odlpp](https://gits-15.sys.kth.se/LCR/ODLpp) and install it. You then need to re-install ODL.

Requirements
------------

- [numpy](https://github.com/numpy/numpy) >= 1.8
- [scipy](https://github.com/scipy/scipy) >= 0.14
- [python-future](https://pypi.python.org/pypi/future/) as Python 2/3 compatibility layer.
- [odlpp](https://gits-15.sys.kth.se/LCR/ODLpp) (not yet available) for GPU support (optional).

#### For unittests (`run_tests.py`)

- [pytest](https://pypi.python.org/pypi/pytest) >= 2.7.0
- [coverage](https://pypi.python.org/pypi/coverage/) >= 4.0.0

Code guidelines
--------------------
The code is written in python 2/3 through the *future* library. It is intended to work on
all major platforms (GNU/Linux / Mac / Windows).

Current status (2015-09-16) is

| Platform     | Python | Works | CUDA  |
|--------------|--------|-------|-------|
| Windows 7    | 2.7    | ✔     | ✔     |
| Ubuntu 14.04 | 2.7    | ✔     | ✔     |
| Fedora 22    | 2.7    | ✔     | x (1) |
| Fedora 22    | 3.4    | ✔     | x (1) |
| Mac OSX      | ??     | ??    | ??    |

(1) The GCC 5.x compiler is not compatible with current CUDA (7.5)

### Formating
The code is supposed to be formated according to the python style guide [PEP8](https://www.python.org/dev/peps/pep-0008/). A useful tool to enforce this is [autopep8](https://pypi.python.org/pypi/autopep8/).

License
-------

GPL Version 3. See LICENSE file.

If you would like to get the code under a different license, please contact the
developers.

Main developers
---------------

- Jonas Adler (jonas-<ätt>-kth-<dot>-se)
- Holger Kohr (kohr-<ätt>-kth-<dot>-se)
