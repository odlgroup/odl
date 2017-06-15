[![PyPI version](https://badge.fury.io/py/odl.svg)](https://badge.fury.io/py/odl)
[![Build Status](https://travis-ci.org/odlgroup/odl.svg?branch=master)](https://travis-ci.org/odlgroup/odl?branch=master)
[![license](https://img.shields.io/badge/license-MPL--2.0-orange.svg)](https://opensource.org/licenses/MPL-2.0)
[![DOI](https://zenodo.org/badge/45596393.svg)](https://zenodo.org/badge/latestdoi/45596393)

ODL
===

Operator Discretization Library (ODL) is a Python library for fast prototyping focusing on (but not restricted to) inverse problems. ODL is being developed at [KTH Royal Institute of Technology, Stockholm](https://www.kth.se/en/sci/institutioner/math) and [Centrum Wiskunde & Informatica (CWI), Amsterdam](https://www.cwi.nl).

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
- A versatile and pluggable library of optimization routines for smooth and non-smooth problems, such as CGLS, BFGS, Chambolle-Pock and Douglas-Rachford splitting.
- Support for tomographic imaging with a unified geometry representation and bindings to external libraries for efficient computation of projections and back-projections.
- Standardized tests to validate implementations against expected behavior of the corresponding mathematical object, e.g. if a user-defined norm satisfies `norm(x + y) <= norm(x) + norm(y)` for a number of input vectors `x` and `y`.

Installation
============
Installing ODL should be as easy as

    conda install -c odlgroup odl

or

    pip install odl

For more detailed instructions, check out the [Installation guide](https://odlgroup.github.io/odl/getting_started/installing.html).

The code is compatible with Python 2 and 3 through the `future` library. It is intended to work on all major platforms (GNU/Linux / Mac / Windows).

Resources
=========
- [ODL Documentation](https://odlgroup.github.io/odl/)
- [Installation guide](https://odlgroup.github.io/odl/getting_started/installing.html)
- [Getting Started](https://odlgroup.github.io/odl/getting_started/getting_started.html)
- [Code Examples](examples)
- [API reference](https://odlgroup.github.io/odl/odl.html)

Applications
============
ODL has been used in a wide range of research and development with several scientific articles and conference contributions as well as other open source projects using ODL as a back-end. Many of these release their code publicly and can serve as examples of how to use ODL. This is a short list of such projects. If you want to add your project to the list, contact the maintainers or file a pull request.

##### Articles


| Description      |  Code  |
|------------------|--------|
| *Indirect Image Registration with Large Diffeomorphic Deformations*. [Arxiv](https://arxiv.org/abs/1706.04048) | [GitHub](https://github.com/chongchenmath/odl_lddmm) |
| *High-level algorithm prototyping: an example extending the TVR-DART algorithm*. DGCI, 2017 | [GitHub](https://github.com/aringh/TVR-DART) |
| *GPUMCI, a ﬂexible platform for x-ray imaging on the GPU*. Fully3D, 2017 |  |
| *Spectral CT reconstruction with anti-correlated noise model and joint prior*. Fully3D, 2017 | [GitHub](https://github.com/adler-j/spectral_ct_examples) |
| *Solving ill-posed inverse problems using iterative deep neural networks*. [Arxiv](https://arxiv.org/abs/1704.04058) | [GitHub](https://github.com/adler-j/learned_gradient_tomography) |
| *Total variation regularization with variable Lebesgue prior*. [Arxiv](https://arxiv.org/abs/1702.08807) | [GitHub](https://github.com/kohr-h/variable_lp_paper) |
| *Generalized Sinkhorn iterations for regularizing inverse problems using optimal mass transport*. [Arxiv](https://arxiv.org/abs/1612.02273) |   |
| *A modified fuzzy C means algorithm for shading correction in craniofacial CBCT images*. CMBEBIH, 2017 | [GitHub](https://github.com/adler-j/mfcm_article) |
| *Shape Based Image Reconstruction Using Linearized Deformations*. Inverse Problems, 2017. [DOI](http://iopscience.iop.org/article/10.1088/1361-6420/aa55af) | [GitHub](https://github.com/chongchenmath/odl_ld) |


##### Other projects

| Description      |  Code  |
|------------------|--------|
| Multigrid CT reconstruction | [GitHub](https://github.com/kohr-h/odl-multigrid) |
| Inverse problems over Lie groups | [GitHub](https://github.com/adler-j/lie_grp_diffeo) |
| Bindings for the [EMRecon](http://www.uni-muenster.de/Sfbmobil/en/veroeffentlichungen/software/emrecon/index.html) package for PET |  [GitHub](https://github.com/odlgroup/odlemrecon) |
| ADF-STEM reconstruction using nuclear norm regularization | [GitHub](https://github.com/adler-j/odl-stem-examples) |


License
=======
Mozilla Public License version 2.0 or later. See the [LICENSE](LICENSE) file.

ODL developers
--------------
To contact the developers either [open an issue on the issue tracker](https://github.com/odlgroup/odl/issues/) or send an email to odl@math.kth.se

##### Main developers
- Jonas Adler ([@adler-j](https://github.com/adler-j))
- Holger Kohr ([@kohr-h](https://github.com/kohr-h))

##### Contributors
- Ozan Öktem ([@ozanoktem](https://github.com/ozanoktem))
- Axel Ringh ([@aringh](https://github.com/aringh))
- Julian Moosmann ([@moosmann](https://github.com/moosmann))
- Kati Niinimäki ([@niinimaki](https://github.com/niinimaki))
- Matthias J. Ehrhardt ([@MatthiasJE](https://github.com/matthiasje))
- Willem Jan Palenstijn ([@wjp](https://github.com/wjp))
- Chong Chen ([@chongchenmath](https://github.com/chongchenmath))

Funding
-------
Development of ODL is financially supported by the Swedish Foundation for Strategic Research as part of the project "Low complexity image reconstruction in medical imaging".

Some development time has also been financed by [Elekta](https://www.elekta.com/).
