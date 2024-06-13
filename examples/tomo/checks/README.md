# "Visual unit test" examples

The examples in this directory serve to visually check the correctness of implementations in the ODL tomography package, e.g., correct implementation of geometric transformations, axis conventions, etc.
They are primarily intended for developers who change back-end code or introduce a new geometry.

## List of examples

Example | Purpose | Complexity
------- | ------- | ----------
[`check_axes_cone2d_bp.py`](check_axes_cone2d_bp.py) | Check axis conventions and shifts for back-projection in 2D cone beam (= fan beam) geometry | middle
[`check_axes_cone2d_fp.py`](check_axes_cone2d_fp.py) | Check axis conventions and shifts for forward projection in 2D cone beam (= fan beam) geometry | middle
[`check_axes_cone2d_vec_fp.py`](check_axes_cone2d_vec_fp.py) | Check axis conventions and shifts for forward projection in 2D cone beam "vec" geometry | middle
[`check_axes_cone3d_bp.py`](check_axes_cone3d_bp.py) | Check axis conventions and shifts for back-projection in 3D circular cone beam geometry | high
[`check_axes_cone3d_fp.py`](check_axes_cone3d_fp.py) | Check axis conventions and shifts for forward projection in 3D circular cone beam geometry | high
[`check_axes_parallel2d_bp.py`](check_axes_parallel2d_bp.py) | Check axis conventions and shifts for back-projection in 2D parallel beam geometry | middle
[`check_axes_parallel2d_fp.py`](check_axes_parallel2d_fp.py) | Check axis conventions and shifts for forward projection in 2D parallel beam geometry | middle
[`check_axes_parallel2d_vec_bp.py`](check_axes_parallel2d_vec_bp.py) | Check axis conventions and shifts for back-projection in 2D parallel beam "vec" geometry | middle
[`check_axes_parallel2d_vec_fp.py`](check_axes_parallel2d_vec_fp.py) | Check axis conventions and shifts for forward projection in 2D parallel beam "vec" geometry | middle
[`check_axes_parallel3d_bp.py`](check_axes_parallel3d_bp.py) | Check axis conventions and shifts for back-projection in 3D parallel beam geometry | high
[`check_axes_parallel3d_fp.py`](check_axes_parallel3d_fp.py) | Check axis conventions and shifts for forward projection in 3D parallel beam geometry | high
