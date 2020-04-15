# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Convenience functions for operators."""

from __future__ import absolute_import, division, print_function

__all__ = ('as_proximal_lang_operator',)


def as_proximal_lang_operator(op, norm_bound=None):
    """Wrap ``op`` as a ``proximal.BlackBox``.

    This is intended to be used with the `ProxImaL language solvers.
    <https://github.com/comp-imaging/proximal>`_

    For documentation on the proximal language (ProxImaL) see [Hei+2016].

    Parameters
    ----------
    op : `Operator`
        Linear operator to be wrapped. Its domain and range must implement
        ``shape``, and elements in these need to implement ``asarray``.
    norm_bound : float, optional
        An upper bound on the spectral norm of the operator. Note that this is
        the norm as defined by ProxImaL, and hence use the unweighted spaces.

    Returns
    -------
    ``proximal.BlackBox`` : proximal_lang_operator
        The wrapped operator.

    Notes
    -----
    If the data representation of ``op``'s domain and range is of type
    `NumpyTensorSpace` this incurs no significant overhead. If the data
    space is implemented with CUDA or some other non-local representation,
    the overhead is significant.

    References
    ----------
    [Hei+2016] Heide, F et al. *ProxImaL: Efficient Image Optimization using
    Proximal Algorithms*. ACM Transactions on Graphics (TOG), 2016.
    """
    import proximal

    def forward(inp, out):
        out[:] = op(inp).asarray()

    def adjoint(inp, out):
        out[:] = op.adjoint(inp).asarray()

    return proximal.LinOpFactory(
        input_shape=op.domain.shape,
        output_shape=op.range.shape,
        forward=forward,
        adjoint=adjoint,
        norm_bound=norm_bound,
    )


if __name__ == '__main__':
    from odl.util.testutils import run_doctests

    run_doctests()
