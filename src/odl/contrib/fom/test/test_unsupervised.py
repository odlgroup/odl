# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for unsupervised FoMs."""

import numpy as np
import pytest
import odl
import odl.contrib.fom


def noise_element(space):
    return space.element(np.random.randn(*space.shape))

def test_estimate_noise_std_constant_1d():
    """Verify ``estimate_noise_std(0) == 0``"""
    img = odl.rn(10).zero()
    result = odl.contrib.fom.estimate_noise_std(img)
    assert pytest.approx(result) == 0.0


def test_estimate_noise_std_normal_1d():
    """Verify ``estimate_noise_std(N(0, 1)) == 1`` in 1d."""
    img = noise_element(odl.rn(1000))
    result = odl.contrib.fom.estimate_noise_std(img)
    expected = np.std(img.asarray())
    assert pytest.approx(result, abs=0.2) == expected


def test_estimate_noise_std_normal_2d():
    """Verify ``estimate_noise_std(N(0, 1)) == 1`` in 2d."""
    img = noise_element(odl.rn((100, 100)))
    result = odl.contrib.fom.estimate_noise_std(img)
    expected = np.std(img.asarray())
    assert pytest.approx(result, abs=0.2) == expected


def test_estimate_noise_std_normal_4d():
    """Verify ``estimate_noise_std(N(0, 1)) == 1`` in 4d."""
    img = noise_element(odl.rn((10, 10, 10, 10)))
    result = odl.contrib.fom.estimate_noise_std(img)
    expected = np.std(img.asarray())
    assert pytest.approx(result, abs=0.2) == expected


def test_estimate_noise_std_normal_large_1d():
    """Verify ``estimate_noise_std(N(0, 1)) == 1`` with low error."""
    img = noise_element(odl.rn(100000))
    result = odl.contrib.fom.estimate_noise_std(img)
    expected = np.std(img.asarray())
    assert pytest.approx(result, abs=0.01) == expected


def test_estimate_noise_std_normal_2d_pointwise():
    """Verify ``estimate_noise_std(N(0, 1)) == 1`` in 2d."""
    img = noise_element(odl.rn((100, 100)))
    result = odl.contrib.fom.estimate_noise_std(img, average=False)
    result_mean = np.mean(result.asarray())
    expected = np.std(img.asarray())
    assert result.shape == img.shape
    assert result.dtype == result.dtype
    assert pytest.approx(result_mean, abs=0.25) == expected


if __name__ == '__main__':
    odl.core.util.test_file(__file__)
