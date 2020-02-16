[![PyPI version](https://badge.fury.io/py/odl.svg)](https://badge.fury.io/py/odl)
[![Build Status](https://travis-ci.org/odlgroup/odl.svg?branch=master)](https://travis-ci.org/odlgroup/odl?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/odlgroup/odl/badge.svg)](https://coveralls.io/github/odlgroup/odl)
[![license](https://img.shields.io/badge/license-MPL--2.0-orange.svg)](https://opensource.org/licenses/MPL-2.0)
[![DOI](https://zenodo.org/badge/45596393.svg)](https://zenodo.org/badge/latestdoi/45596393)

ODL
===
*Operator Discretization Library* (ODL) is a Python library that enables research in inverse problems on realistic or real data. The framework allows to encapsulate a physical model into an `Operator` that can be used like a mathematical object in, e.g., optimization methods. Furthermore, ODL makes it easy to experiment with reconstruction methods and optimization algorithms for variational regularization, all without sacrificing performance.

For more details and an introduction into the inner workings of ODL, please refer to the [documentation](https://odlgroup.github.io/odl/).

Highlights
==========
- A versatile and efficient library of optimization routines for smooth and non-smooth problems, such as CGLS, BFGS, PDHG and Douglas-Rachford splitting.
- Support for tomographic imaging with a unified geometry representation and bindings to external libraries for efficient computation of projections and back-projections.
- And much more, including support for deep learning libraries, figures of merits, phantom generation, data handling, etc.

Installation
============
Installing ODL should be as easy as

    conda install -c odlgroup odl

or

    pip install odl

For more detailed instructions, check out the [Installation guide](https://odlgroup.github.io/odl/getting_started/installing.html).

ODL is compatible with Python 2/3 and all major platforms (GNU/Linux / Mac / Windows).

Resources
=========
- [ODL Documentation](https://odlgroup.github.io/odl/)
- [Installation guide](https://odlgroup.github.io/odl/getting_started/installing.html)
- [Getting Started](https://odlgroup.github.io/odl/getting_started/getting_started.html)
- [Code Examples](examples)
- [API reference](https://odlgroup.github.io/odl/odl.html)
- [ODL Course Material](https://github.com/odlgroup/odlworkshop)

Applications
============
This is an incomplete list of articles and projects using ODL. If you want to add your project to the list, contact the maintainers or file a pull request.

| Article      |  Code  |
|------------------|--------|
| *Learning to solve inverse problems using Wasserstein loss*. NIPS OMT Workshop 2017. [arXiv](https://arxiv.org/abs/1710.10898) | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/adler-j/wasserstein_inverse_problems) |
| *Faster PET Reconstruction with a Stochastic Primal-Dual Hybrid Gradient Method*. [Article](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10394/103941O/Faster-PET-reconstruction-with-a-stochastic-primal-dual-hybrid-gradient/10.1117/12.2272946.full?SSO=1) |  |
| *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling and Imaging Applications*. [arXiv](https://arxiv.org/abs/1706.04957) | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/mehrhardt/spdhg) |
| *Learned Primal-Dual Reconstruction*. [arXiv](https://arxiv.org/abs/1707.06474), [blog](https://adler-j.github.io/2017/07/21/Learning-to-reconstruct.html) | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/adler-j/learned_primal_dual) |
| *Indirect Image Registration with Large Diffeomorphic Deformations*. [arXiv](https://arxiv.org/abs/1706.04048) | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/chongchenmath/odl_lddmm) |
| *High-level algorithm prototyping: an example extending the TVR-DART algorithm*. DGCI, 2017. [DOI](https://doi.org/10.1007/978-3-319-66272-5_10) | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/aringh/TVR-DART) |
| *GPUMCI, a ﬂexible platform for x-ray imaging on the GPU*. Fully3D, 2017 |  |
| *Spectral CT reconstruction with anti-correlated noise model and joint prior*. Fully3D, 2017 | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/adler-j/spectral_ct_examples) |
| *Solving ill-posed inverse problems using iterative deep neural networks*. Inverse Problems, 2017 [arXiv](https://arxiv.org/abs/1704.04058), [DOI](https://doi.org/10.1088/1361-6420/aa9581) | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/adler-j/learned_gradient_tomography) |
| *Total variation regularization with variable Lebesgue prior*. [arXiv](https://arxiv.org/abs/1702.08807) | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/kohr-h/variable_lp_paper) |
| *Generalized Sinkhorn iterations for regularizing inverse problems using optimal mass transport*. SIAM Journal on Imaging Sciences, 2017. [arXiv](https://arxiv.org/abs/1612.02273), [DOI](https://doi.org/10.1137/17M111208X) | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/aringh/Generalized-Sinkhorn-and-tomography) |
| *A modified fuzzy C means algorithm for shading correction in craniofacial CBCT images*. CMBEBIH, 2017 | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/adler-j/mfcm_article) |
| *The MAX IV imaging concept*. [Article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5133273/) | |
| *Shape Based Image Reconstruction Using Linearized Deformations*. Inverse Problems, 2017. [DOI](http://iopscience.iop.org/article/10.1088/1361-6420/aa55af) | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/chongchenmath/odl_ld) |

| Project      |  Code  |
|------------------|--------|
| Multigrid CT reconstruction | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/kohr-h/odl-multigrid) |
| Inverse problems over Lie groups | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/adler-j/lie_grp_diffeo) |
| Bindings for the [EMRecon](http://www.uni-muenster.de/Sfbmobil/en/veroeffentlichungen/software/emrecon/index.html) package for PET |  [<img src="https://github.com/favicon.ico" width="24">](https://github.com/odlgroup/odlemrecon) |
| ADF-STEM reconstruction using nuclear norm regularization | [<img src="https://github.com/favicon.ico" width="24">](https://github.com/adler-j/odl-stem-examples) |


License
-------
Mozilla Public License version 2.0 or later. See the [LICENSE](LICENSE) file.

ODL developers
--------------
Development of ODL started in 2014 as part of the project "Low complexity image reconstruction in medical imaging” by Ozan Öktem ([@ozanoktem](https://github.com/ozanoktem)), Jonas Adler ([@adler-j](https://github.com/adler-j)) and Holger Kohr ([@kohr-h](https://github.com/kohr-h)). Several others have made significant contributions, see the [contributors](CONTRIBUTORS.md) list.

To contact the developers either open an issue on the issue tracker or send an email to odl@math.kth.se.

Funding
-------
ODL has primarily been developed at [KTH Royal Institute of Technology, Stockholm](https://www.kth.se/en/sci/institutioner/math) and [Centrum Wiskunde & Informatica (CWI), Amsterdam](https://www.cwi.nl).
It is financially supported by the Swedish Foundation for Strategic Research as part of the project "Low complexity image reconstruction in medical imaging". 

Some development time has also been financed by [Elekta](https://www.elekta.com/).
