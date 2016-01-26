# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""The split bregman method for L1 regularized problems.

The method is proposed in `[1]_` and used to solve TV style regularized
problems.

.. [1]: ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External

# Internal
from odl.operator.operator import Operator
from odl.solvers.iterative import conjugate_gradient as cgn
from odl.util import pspace_squared_sum

__all__ = ('l2_gradient', 'split_bregman_solver',)


class l2_gradient(Operator):

    """ Gradient of ``||op(x) - rhs||_2^2``"""

    def __init__(self, op, rhs):
        """Initialize an instance.

        Parameters
        ----------
        H_grad : linear `Operator`
            Forward operator
        rhs : ``op.range`` element
            Right hand side of the inverse problem.

        """
        self.op = op
        self.rhs = rhs

        assert rhs in op.domain
        assert op.is_linear

        super().__init__(op.domain, op.domain)

    def _call(self, x, out=None):
        return self.op.adjoint(self.op(x) - self.rhs, out=out)

    def derivative(self, x):
        return self.op.adjoint * self.op


def split_bregman_solver(H_grad, Phi, x, lam,
                         niter=1, inner_niter=1,
                         isotropic=False, inner_solver=None,
                         partial=None):
    """ Reconstruct with split Bregman.

    Solves the L1 regularized problem

        ``min_x H(x) + mu ||Phi(x)||_1``

    By applying the bregman distance:

        ...

    Parameters
    ----------
    H_grad : `Operator`
        Gradient of data discreptancy operator H
    Phi : `Operator`
        Sparsifying transformation
    x : ``op.domain`` element
        Initial guess and output parameter
    lam : positive `float`
        Penalty function weights parameter "lambda"
    niter : positive `int`
        Number of outer iterations
    inner_niter : positive `int`
        Number of inner iterations
    isotropic : `bool`
        Applicable in case where ``Phi`` is a function``X -> Y^n``,
        then optimized for the case where the ``sum(Phi(x)**2)`` is sparse.
    inner_solver : `callable`
        Callable with signature ``inner_solver(op, x, rhs)``, used to solve
        the inner problem. Default: `conjugate_gradient` with 1 iteration.
    """

    assert H_grad.domain == Phi.domain
    assert x in H_grad.domain

    lam = float(lam)

    # If no solver is given, create a new.
    if inner_solver is None:
        inner_solver = lambda op, x, rhs: cgn(op, x, rhs, 1)
    else:
        assert callable(inner_solver)

    b = Phi.range.zero()
    d = Phi.range.zero()

    for i in range(niter):
        for n in range(inner_niter):
            # Solve tomography part using the given solver
            inner_op = H_grad + lam * (Phi.adjoint * Phi)
            inner_rhs = lam * Phi.adjoint(d-b)
            inner_solver(inner_op, x, inner_rhs)

            # Solve for d using soft threshholding
            s = Phi(x) + b
            if isotropic:
                sn = pspace_squared_sum(s)
                sn.ufunc.add(0.0001, out=sn)  # avoid 0/0 issues
                sn = sn.ufunc.add(-1.0 / lam).ufunc.maximum(0.0) / sn
                for j in range(len(d)):
                    d[j].multiply(sn, s[j])
            else:
                # d = sign(Phi(x)+b) * max(|Phi(x)+b|-la^-1,0)
                d = s.ufunc.sign() * (s.ufunc.absolute().
                                      ufunc.add(-1.0 / lam).
                                      ufunc.maximum(0.0))

        # Update lagrangian estimates
        b.lincomb(1, s, -1, d)

        if partial:
            partial(x)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
