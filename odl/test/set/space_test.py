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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External module imports
import pytest
import odl
from odl.util.testutils import simple_fixture, noise_element

hilbert_spaces =  [odl.rn(3), odl.cn(3), odl.uniform_discr(0, 1, 3)]
normed_spaces = [odl.rn(3, exponent=1)] + hilbert_spaces
metric_spaces = normed_spaces
linear_spaces = metric_spaces

hilbert_space = simple_fixture('space', hilbert_spaces)
normed_space = simple_fixture('space', normed_spaces)
metric_space = simple_fixture('space', metric_spaces)
linear_space = simple_fixture('space', linear_spaces)


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
    """Verify that elements in spaces cannot be compared."""
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
