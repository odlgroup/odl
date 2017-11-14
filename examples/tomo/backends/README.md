# Backend-specific examples

Examples in this directory serve to test specific aspects of particular back-ends for tomography, e.g., performance tests.

## List of examples

Example | Purpose | Complexity
------- | ------- | ----------
[`astra_performance_cpu_parallel_2d_cg.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/backends/astra_performance_cpu_parallel_2d_cg.py) | Speed test of conjugate gradient least-squares (CGLS) reconstruction in 2D parallel beam geometry on the CPU, comparing the native ASTRA implementation with ODL's version using ASTRA as back-end | middle
[`astra_performance_cuda_cone_3d_cg.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/backends/astra_performance_cuda_cone_3d_cg.py) | Speed test of conjugate gradient least-squares (CGLS) reconstruction in 3D circular cone beam geometry using CUDA, comparing the native ASTRA implementation with ODL's version using ASTRA as back-end | middle
[`astra_performance_cuda_parallel_2d_cg.py`](https://github.com/odlgroup/odl/blob/master/examples/tomo/backends/astra_performance_cuda_parallel_2d_cg.py) | Speed test of conjugate gradient least-squares (CGLS) reconstruction in 2D parallel beam geometry using CUDA, comparing the native ASTRA implementation with ODL's version using ASTRA as back-end | middle
