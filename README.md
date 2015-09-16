ODL
===

Operator Discretization Library (ODL), is a python library for inverse problems developed at KTH, Royal Institute of Technology.

The library is intended to allow researchers to write prototypes suitable for real life data by exploiting existing hardware accelerated forward and back projection codes in a uniform manner using the language of mathematical analysis.

The library has two parts. The analysis part allows users to define mathematical objects such as linear operators and vector spaces as well as methods on these such as the conjugate gradient method. The tomography part allows the definition of acquisition geometries in continuum and the discretization of these.

The initial focus of the tomography library is on X-ray based tomography problems such as CT, CBCT, SPECT and more.

Installation
------------

Run (as root)
```
python setup.py install
```
in the root folder of the package. After installing, you should check that everything installed properly by running:

```
python run_tests.py
```

##### Cuda

If you also wish to use the (optional) `cuda` extensions you need to do:

```bash
git submodule update --init --recursive
cd odlpp
```

From here follow the instructions in [odlpp](https://gits-14.sys.kth.se/LCR/ODLpp) and install it. You then need to re-install ODL.

Requirements
------------

- [numpy](https://github.com/numpy/numpy) for numerics.
- [python-future](https://pypi.python.org/pypi/future/) as Python 2/3 compatibility layer.
- [odlpp](https://gits-14.sys.kth.se/LCR/ODLpp) for GPU support (optional).

Code guidelines
--------------------
The code is written in python 2/3 through the future library. It is intended to work on all major platforms (unix/max/windows)

Current status (2015-09-16) is

| Platform     | Python | Works |
|--------------|--------|-------|
| Windows 7    | 2.7    | ✔     |
| Ubuntu 14.04 | 2.7    | ✔     |
| Fedora ??    | ??     | ✔     |
| Mac OSX      | ??     | ??     |

### Formating
The code is supposed to be formated according to the python style guide [PEP8](https://www.python.org/dev/peps/pep-0008/). A useful tool to enforce this is [autopep8](https://pypi.python.org/pypi/autopep8/).

Projects that use ODL
---------------------

Several research projects use ODL and provide bindings, these include

- [GPUMCI](https://gits-14.sys.kth.se/jonasadl/GPUMCI) Monte carlo library for simulation of medical imaging systems used at Elekta.
- [SimRec2D](https://gits-14.sys.kth.se/LCR/SimRec2D) 2D/3D forward and back projectors for (CB)CT

License
-------

GPL Version 3. See LICENSE file.
