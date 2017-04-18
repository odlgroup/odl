# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import pytest
import odl
from odl.util.testutils import simple_fixture, noise_element


# --- pytest fixtures --- #


hilbert_spaces = [odl.rn(3), odl.cn(3), odl.uniform_discr(0, 1, 3)]
normed_spaces = [odl.rn(3, exponent=1)] + hilbert_spaces
metric_spaces = normed_spaces
linear_spaces = metric_spaces

hilbert_space = simple_fixture('space', hilbert_spaces)
normed_space = simple_fixture('space', normed_spaces)
metric_space = simple_fixture('space', metric_spaces)
linear_space = simple_fixture('space', linear_spaces)


# --- LinearSpace tests --- #


def test_hash(linear_space):
    """Verify that hashing spaces works but elements doesnt."""
    hsh = hash(linear_space)

    # Check that the trivial hash algorithm is not used
    assert hsh != id(linear_space)

    x = noise_element(linear_space)
    with pytest.raises(TypeError):
        hash(x)


def test_equality(metric_space):
    """Verify that equality testing works."""
    x = noise_element(metric_space)
    y = noise_element(metric_space)

    assert x == x
    assert y == y
    assert x != y


def test_comparsion(linear_space):
    """Verify that spaces and elements in spaces cannot be compared."""
    with pytest.raises(TypeError):
        linear_space <= linear_space
    with pytest.raises(TypeError):
        linear_space < linear_space
    with pytest.raises(TypeError):
        linear_space >= linear_space
    with pytest.raises(TypeError):
        linear_space > linear_space

    x = noise_element(linear_space)
    y = noise_element(linear_space)

    with pytest.raises(TypeError):
        x <= y
    with pytest.raises(TypeError):
        x < y
    with pytest.raises(TypeError):
        x >= y
    with pytest.raises(TypeError):
        x > y


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
