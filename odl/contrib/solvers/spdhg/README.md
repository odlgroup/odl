# SPDHG

This package contains an ODL compatible implementation of the Stochastic Primal-Dual Hybrid Gradient algorithm proposed and analyzed in [arxiv](https://arxiv.org/abs/1706.04957). It is useful to solve problems of the form
	min_x sum_{i=1}^n f_i(A_i x) + g(x)
where f_i and g are closed, proper and convex functionals and A_i are linear operators.

It has been successfully used for PET image reconstruction (from the Siemens Biograph mMR) [SPIE](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10394/103941O/Faster-PET-reconstruction-with-a-stochastic-primal-dual-hybrid-gradient/10.1117/12.2272946.full). Watch [this](https://www.youtube.com/watch?v=iZc2eFqS2l4) if you want to have an introduction to SPDHG by [Peter Richtárik](http://www.maths.ed.ac.uk/~prichtar/).

Original contribution by: [@mehrhardt](https://github.com/mehrhardt)

## Content

* `spdhg` in [stochastic_primal_dual_hybrid_gradient.py](stochastic_primal_dual_hybrid_gradient.py) is an implementation of SPDHG with constant parameters.
* `pa_spdhg` in [stochastic_primal_dual_hybrid_gradient.py](stochastic_primal_dual_hybrid_gradient.py) implements primal accelerated SPDHG if g is strongly convex.
* `da_spdhg` in [stochastic_primal_dual_hybrid_gradient.py](stochastic_primal_dual_hybrid_gradient.py) implements dual accelerated SPDHG if the f_i have Lipschitz continuous gradients.

## Example usage

The [examples](examples) folder contains examples on how to use the above functionality. The PET examples are based on [ASTRA](https://www.astra-toolbox.com/) to compute the line integrals in the forward operator.

* [get_started.py](examples/get_started.py) shows the usage of SPDHG as simple as possible.

More involved examples include:

* [PET_1k.py](examples/PET_1k.py) shows an example for PET reconstruction (from simulated data) with total variation regularization where the algorithm is proven to converge with rate O(1/k) in the partial primal-dual gap.
* [ROF_1k2_primal.py](examples/ROF_1k2_primal.py) shows primal acceleration for ROF denoising (Gaussian noise; squared L2-norm + total variation) where the algorithm is proven to converge with rate O(1/k^2) in squared norm to the primal solution.
* [deblurring_1k2_dual.py](examples/deblurring_1k2_dual.py) shows an example of dual acceleration where the dual iterates converge in squared norm with rate O(1/k^2) to the dual solution. The problem at hand is deblurring with total variation regularization and a Poisson noise model.
* [PET_linear_rate.py](examples/PET_linear_rate.py) shows linear convergence of SPDHG for PET reconstruction with a strongly convex total variation-type regularization.
