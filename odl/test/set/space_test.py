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
from odl.util.testutils import simple_fixture


# --- pytest fixtures --- #


spaces = [
    odl.rn(3), odl.cn(3), odl.rn(3, exponent=1), odl.uniform_discr(0, 1, 3)
]
space = simple_fixture('space', spaces)


# --- LinearSpace tests --- #


def test_hash(space):
    """Verify that hashing of spaces works."""
    hsh = hash(space)

    # Check that the trivial hash algorithm is not used
    assert hsh != id(space)


def test_comparsion_raises(space):
    """Verify that spaces cannot be compared."""
    with pytest.raises(TypeError):
        space <= space
    with pytest.raises(TypeError):
        space < space
    with pytest.raises(TypeError):
        space >= space
    with pytest.raises(TypeError):
        space > space


if __name__ == '__main__':
    odl.util.test_file(__file__)
