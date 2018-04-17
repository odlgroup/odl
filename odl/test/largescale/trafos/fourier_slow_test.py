# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Large-scale tests for the Fourier transform.

This test suite is intended for performance monitoring, to capture
speed regressions.
"""

from __future__ import division

import numpy as np
import pytest

import odl
from odl.util.testutils import never_skip, simple_fixture

skip_if_no_pyfftw = pytest.mark.skipif("not odl.trafos.PYFFTW_AVAILABLE",
                                       reason='pyfftw not available')
pytestmark = odl.util.skip_if_no_largescale


# --- pytest fixtures --- #


# bug in pytest (ignores pytestmark) forces us to do this this
impl_params = [never_skip('numpy'), skip_if_no_pyfftw('pyfftw')]
largescale = " or not pytest.config.getoption('--largescale')"
impl = simple_fixture('impl',
                      [pytest.mark.skipif(p.args[0] + largescale, p.args[1])
                       for p in impl_params])

domain = simple_fixture(
    name='domain',
    params=[odl.uniform_discr(-2, 2, 10 ** 5),
            odl.uniform_discr([-2, -2, -2], [2, 2, 2], [200, 200, 200]),
            odl.uniform_discr(-2, 2, 10 ** 5, dtype='complex'),
            odl.uniform_discr([-2, -2, -2], [2, 2, 2], [200, 200, 200],
                              dtype='complex')])


# --- FourierTransform tests --- #


def test_dft_forward(domain, impl):

    halfcomplex = np.issubdtype(domain.dtype, np.floating)
    dft = odl.trafos.DiscreteFourierTransform(domain, halfcomplex=halfcomplex)
    one = domain.one()
    out = dft.range.element()

    dft(one, out=out)
    assert out[(0,) * out.ndim] == domain.size


def test_fourier_trafo_forward_complex(domain, impl):

    if domain.field == odl.RealNumbers():
        return

    ft = odl.trafos.FourierTransform(domain, impl=impl)

    def charfun_ball(x):
        sum_sq = sum(xi ** 2 for xi in x)
        return np.where(sum_sq < 1, 1, 0)

    def charfun_freq_ball(x):
        rad = np.max(ft.range.domain.extent / 4)
        sum_sq = sum(xi ** 2 for xi in x)
        return np.where(sum_sq < rad ** 2, 1, 0)

    ball_dom = ft.domain.element(charfun_ball)
    ball_ran = ft.range.element(charfun_freq_ball)

    ball_dom_ft = ft(ball_dom)
    ball_ran_ift = ft.adjoint(ball_ran)
    assert almost_equal(ball_dom.inner(ball_ran_ift),
                        ball_ran.inner(ball_dom_ft), places=1)


if __name__ == '__main__':
    odl.util.test_file(__file__, ['--largescale'])
