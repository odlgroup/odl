# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import pytest
try:
    import pywt
except ImportError:
    pass

import odl
from odl.trafos.backends.pywt_bindings import (
    PYWT_AVAILABLE, PAD_MODES_ODL2PYWT, pywt_wavelet, pywt_pad_mode)
from odl.util.testutils import (simple_fixture)

pytestmark = pytest.mark.skipif(not PYWT_AVAILABLE,
                                reason='`pywt` backend not available')


# --- pytest fixtures --- #
wavelet = simple_fixture('wavelet_name', ['db1', 'sym2'])
odl_mode = simple_fixture('odl_mode', ['constant', 'order0', 'order1'])


def test_pywt_wavelet(wavelet):
    # pywt_wavelet takes either string or pywt.Wavelet object as input
    wavelet = pywt_wavelet(wavelet)
    assert isinstance(wavelet, pywt.Wavelet)

    wavelet2 = pywt_wavelet(wavelet)
    assert isinstance(wavelet2, pywt.Wavelet)
    assert wavelet2 is wavelet


def test_pywt_pad_mode(odl_mode):
    pywt_mode = pywt_pad_mode(odl_mode)
    assert pywt_mode in list(PAD_MODES_ODL2PYWT.values())


def test_pywt_pad_errors():
    with pytest.raises(ValueError):
        pywt_pad_mode('constant', pad_const=1.)
    with pytest.raises(ValueError):
        pywt_pad_mode('invalid mode')


if __name__ == '__main__':
    odl.util.test_file(__file__)
