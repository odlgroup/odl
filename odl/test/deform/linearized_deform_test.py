# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for linearized deformation operators."""

from __future__ import division

import numpy as np
import pytest

import odl
from odl.deform import LinDeformFixedDisp, LinDeformFixedTempl
from odl.space.entry_points import tensor_space_impl
from odl.util.testutils import simple_fixture

# --- pytest fixtures --- #


dtype = simple_fixture('dtype', ['float', 'complex'])
interp = simple_fixture('interp', ['linear', 'nearest'])
ndim = simple_fixture('ndim', [1, 2, 3])


@pytest.fixture
def space(request, ndim, dtype, odl_tspace_impl):
    """Provide a space for unit tests."""
    impl = odl_tspace_impl
    supported_dtypes = tensor_space_impl(impl).available_dtypes()
    if np.dtype(dtype) not in supported_dtypes:
        pytest.skip('dtype not available for this backend')

    return odl.uniform_discr(
        [-1] * ndim, [1] * ndim, [20] * ndim, impl=impl, dtype=dtype
    )


# --- Helper functions --- #


SIGMA = 0.3  # width of the gaussian
EPS = 0.25   # scale of the displacement field


def error_bound(interp):
    """Error bound varies with interpolation (larger for "worse")."""
    if interp == 'linear':
        return 0.1
    elif interp == 'nearest':
        return 0.2


def prod(x):
    """Product of a sequence."""
    prod = 1
    for xi in x:
        prod = prod * xi
    return prod


def template_function(x):
    """Gaussian function with std SIGMA."""
    return np.exp(-sum(xi ** 2 for xi in x) / SIGMA ** 2)


def template_grad_factory(n):
    """Gradient of the gaussian."""
    def template_grad_i(i):
        # Indirection for lambda capture
        return lambda x: -2 * x[i] / SIGMA ** 2 * template_function(x)
    return [template_grad_i(i) for i in range(n)]


def disp_field_factory(n):
    """Displacement field.

    In 1d: (x)
    In 2d: (xy, y)
    In 3d: (xyz, y, z)
    etc...
    """
    def coordinate_projection_i(i):
        # Indirection for lambda capture
        return lambda x: EPS * x[i]

    lst = [lambda x: EPS * prod(x)]
    lst.extend(coordinate_projection_i(i) for i in range(1, n))
    return lst


def exp_div_inv_disp(x):
    """Exponential of the divergence of the displacement field.

    In 1d: exp(- EPS)
    In 2d: exp(- EPS * (y + 1))
    In 2d: exp(- EPS * (yz + 2))
    """
    return np.exp(- EPS * (prod(x[1:]) + (len(x) - 1)))


def displaced_points(x):
    """Displaced coordinate points."""
    disp = [dsp(x) for dsp in disp_field_factory(len(x))]
    return [xi + di for xi, di in zip(x, disp)]


def deformed_template(x):
    """Deformed template."""
    return template_function(displaced_points(x))


def vector_field_factory(n):
    """Vector field for the gradient.

    In 1d: (x)
    In 2d: (x, y)
    In 3d: (x, y, z)
    etc...
    """
    def vector_field_i(i):
        return lambda x: x[i]
    return [vector_field_i(i) for i in range(n)]


def template_deformed_grad_factory(n):
    """Deformed gradient."""
    templ_grad = template_grad_factory(n)

    def template_deformed_gradi(i):
        # Indirection for lambda capture
        return lambda x: templ_grad[i](displaced_points(x))

    return [template_deformed_gradi(i) for i in range(n)]


def fixed_templ_deriv(x):
    """Derivative taken in disp_field and evaluated in vector_field."""
    dg = [tdgf(x) for tdgf in template_deformed_grad_factory(len(x))]
    v = [vff(x) for vff in vector_field_factory(len(x))]
    return sum(dgi * vi for dgi, vi in zip(dg, v))


def inv_deformed_template(x):
    """Analytic inverse deformation of the template function."""
    disp = [dsp(x) for dsp in disp_field_factory(len(x))]
    disp_x = [xi - di for xi, di in zip(x, disp)]
    return template_function(disp_x)


# --- LinDeformFixedTempl --- #


def test_fixed_templ_init():
    """Test init and props of linearized deformation with fixed template."""
    space = odl.uniform_discr(0, 1, 5)
    template = space.element(template_function)

    # Valid input
    op = LinDeformFixedTempl(space, template)
    assert repr(op) != ''
    op = LinDeformFixedTempl(space, template, interp='nearest')
    assert repr(op) != ''
    op = LinDeformFixedTempl(space, template_function)
    assert repr(op) != ''


def test_fixed_templ_call(space, interp):
    """Test call of linearized deformation with fixed template."""
    if space.dtype.kind == 'c':
        pytest.xfail('wrongly using complex displacement field')

    # Define the analytic template as the hat function and its gradient
    template = space.element(template_function)
    deform_op = LinDeformFixedTempl(space, template, interp=interp)

    # Calculate result and exact result
    true_deformed_templ = space.element(deformed_template)
    deformed_templ = deform_op(disp_field_factory(space.ndim))

    # Verify that the result is within error limits
    error = space.norm(true_deformed_templ - deformed_templ)
    rel_err = error / space.norm(deformed_templ)
    assert rel_err < error_bound(interp)


def test_fixed_templ_deriv(space, interp):
    """Test derivative of linearized deformation with fixed template."""
    if not space.is_real:
        pytest.skip('derivative not implemented for complex dtypes')

    # Set up template and displacement field
    template = space.element(template_function)
    disp_field = disp_field_factory(space.ndim)
    vector_field = vector_field_factory(space.ndim)
    fixed_templ_op = LinDeformFixedTempl(space, template, interp=interp)

    # Calculate result
    fixed_templ_op_deriv = fixed_templ_op.derivative(disp_field)
    fixed_templ_deriv_comp = fixed_templ_op_deriv(vector_field)

    # Calculate the analytic result
    fixed_templ_deriv_exact = space.element(fixed_templ_deriv)

    # Verify that the result is within error limits
    error = space.norm(fixed_templ_deriv_exact - fixed_templ_deriv_comp)
    rel_err = error / space.norm(fixed_templ_deriv_comp)
    assert rel_err < error_bound(interp)


# --- LinDeformFixedDisp --- #


def test_fixed_disp_init():
    """Test init and props of lin. deformation with fixed displacement."""
    space = odl.uniform_discr(0, 1, 5)
    disp_field = space.tangent_bundle.element(disp_field_factory(space.ndim))

    # Valid input
    op = LinDeformFixedDisp(space, disp_field)
    assert repr(op) != ''
    op = LinDeformFixedDisp(space, disp_field, interp='nearest')
    assert repr(op) != ''
    # Okay in 1D
    op = LinDeformFixedDisp(space, disp_field[0], interp='nearest')

    # Non-valid input
    with pytest.raises(TypeError):  # templ_space not a power space
        bad_pspace = odl.ProductSpace(space, odl.rn(3))
        LinDeformFixedDisp(bad_pspace, disp_field)
    with pytest.raises(TypeError):  # templ_space not based on DiscreteLp
        bad_pspace = odl.ProductSpace(odl.rn(2), 1)
        LinDeformFixedDisp(bad_pspace, disp_field)
    with pytest.raises(ValueError):  # vector field spaces don't match
        bad_space = odl.uniform_discr(0, 1, 10)
        LinDeformFixedDisp(bad_space, disp_field)


def test_fixed_disp_call(space, interp):
    """Test call of lin. deformation with fixed displacement."""
    if space.dtype.kind == 'c':
        pytest.xfail('wrongly using complex displacement field')

    template = space.element(template_function)
    disp_field = space.real_space.tangent_bundle.element(
        disp_field_factory(space.ndim)
    )

    # Calculate result and exact result
    deform_op = LinDeformFixedDisp(space, disp_field, interp)
    deformed_templ = deform_op(template)
    true_deformed_templ = space.element(deformed_template)

    # Verify that the result is within error limits
    error = space.norm(true_deformed_templ - deformed_templ)
    rel_err = error / space.norm(deformed_templ)
    assert rel_err < error_bound(interp)


def test_fixed_disp_inv(space, interp):
    """Test inverse of lin. deformation with fixed displacement."""
    if space.dtype.kind == 'c':
        pytest.xfail('wrongly using complex displacement field')

    # Set up template and displacement field
    template = space.element(template_function)
    disp_field = space.real_space.tangent_bundle.element(
        disp_field_factory(space.ndim))

    # Verify that the inverse is in fact a (left and right) inverse
    deform_op = LinDeformFixedDisp(space, disp_field, interp)

    result_op_inv = deform_op(deform_op.inverse(template))
    error = space.norm(result_op_inv - template)
    rel_err = error / space.norm(template)
    assert rel_err < 2 * error_bound(interp)  # need a bit more tolerance

    result_inv_op = deform_op.inverse(deform_op(template))
    error = space.norm(result_inv_op - template)
    rel_err = error / space.norm(template)
    assert rel_err < 2 * error_bound(interp)  # need a bit more tolerance


def test_fixed_disp_adj(space, interp):
    """Test adjoint of lin. deformation with fixed displacement."""
    if space.dtype.kind == 'c':
        pytest.xfail('wrongly using complex displacement field')

    # Set up template and displacement field
    template = space.element(template_function)
    disp_field = space.real_space.tangent_bundle.element(
        disp_field_factory(space.ndim))

    # Calculate result
    deform_op = LinDeformFixedDisp(space, disp_field, interp)
    deformed_templ_adj = deform_op.adjoint(template)

    # Calculate the analytic result
    true_deformed_templ_adj = space.element(inv_deformed_template)
    exp_div = space.element(exp_div_inv_disp)
    true_deformed_templ_adj *= exp_div

    # Verify that the result is within error limits
    error = space.norm(deformed_templ_adj - true_deformed_templ_adj)
    rel_err = error / space.norm(true_deformed_templ_adj)
    assert rel_err < error_bound(interp)

    # Verify the adjoint definition <Ax, x> = <x, A^* x>
    deformed_templ = deform_op(template)
    inner1 = space.inner(deformed_templ, template)
    inner2 = space.inner(template, deformed_templ_adj)
    assert inner1 == pytest.approx(inner2, abs=.1)


if __name__ == '__main__':
    odl.util.test_file(__file__)
