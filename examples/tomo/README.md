# Tomography examples

These examples demonstrate the capability of ODL to perform tomographic projections, back-projections and filtered back-projection reconstruction in the various supported geometries. They also serve as copy-and-paste templates for the basic setup of more complex applications in tomography.

For examples on how to use the tomography operators in inverse problems, see the [examples/solvers](../solvers) folder.

Note that several of these examples are best/only runnable with astra, which can be easily installed via:

    conda install -c astra-toolbox astra-toolbox
    
If using astra is for some reason not possible, the 2d parallel beam examples can be run with scikit-image, installed by:

    pip install scikit-image

## Basic usage examples

### 2D

Example | Purpose | Complexity
------- | ------- | ----------
[`ray_trafo_parallel_2d.py`](ray_trafo_parallel_2d.py) | Projection and back-projection in 2D parallel beam geometry | middle
[`ray_trafo_parallel_2d_complex.py`](ray_trafo_parallel_2d_complex.py) | Projection and back-projection in 2D parallel beam geometry **with complex-valued data** | middle
[`ray_trafo_cone_2d.py`](ray_trafo_cone_2d.py) | Projection and back-projection in 2D cone (=fan) beam geometry | middle

### 3D

Example | Purpose | Complexity
------- | ------- | ----------
[`ray_trafo_parallel_3d.py`](ray_trafo_parallel_3d.py) | Projection and back-projection in 3D parallel beam single-axis geometry | middle
[`ray_trafo_cone_3d.py`](ray_trafo_cone_3d.py) | Projection and back-projection in 3D circular cone beam geometry | middle
[`ray_trafo_helical_cone_3d.py`](ray_trafo_helical_cone_3d.py) | Projection and back-projection in 3D helical cone beam geometry | middle
[`anisotropic_voxels.py`](anisotropic_voxels.py) | Projection in 3D parallel beam single-axis geometry **with non-cube voxels** | middle


## Reconstruction examples

### 2D

Example | Purpose | Complexity
------- | ------- | ----------
[`filtered_backprojection_parallel_2d.py`](filtered_backprojection_parallel_2d.py) | FBP reconstruction in 2D parallel beam geometry | middle
[`filtered_backprojection_parallel_2d_complex.py`](filtered_backprojection_parallel_2d_complex.py) | FBP reconstruction in 2D parallel beam geometry **with complex-valued data** | middle
[`filtered_backprojection_cone_2d.py`](filtered_backprojection_cone_2d.py) | (Inexact) FBP reconstruction in 2D fan beam geometry | middle
[`filtered_backprojection_cone_2d_short_scan.py`](filtered_backprojection_cone_2d_short_scan.py) | (Inexact) FBP reconstruction in 2D fan beam geometry **with short scan (less than 360 degrees)** | middle

### 3D

Example | Purpose | Complexity
------- | ------- | ----------
[`filtered_backprojection_parallel_3d.py`](filtered_backprojection_parallel_3d.py) | FBP reconstruction in 3D parallel beam single-axis geometry | middle
[`filtered_backprojection_cone_3d.py`](filtered_backprojection_cone_3d.py) | (Inexact) FBP reconstruction in 3D circular cone beam geometry | middle
[`filtered_backprojection_cone_3d_short_scan.py`](filtered_backprojection_cone_3d_short_scan.py) | (Inexact) FBP reconstruction in 3D circular cone beam geometry **with short scan (less than 360 degrees)** | middle
[`filtered_backprojection_helical_3d.py`](filtered_backprojection_helical_3d.py) | (Inexact) FBP reconstruction in 3D helical cone beam geometry | middle

