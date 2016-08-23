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

# ODL imports
from odl.set.sets import (EmptySet, UniversalSet, Strings, ComplexNumbers,
                          RealNumbers, Integers)


def test_empty_set():
    X = EmptySet()
    Z = Integers()

    # __contains
    assert None in X
    assert 1 not in X

    # Contains_set
    assert X.contains_set(X)
    assert not X.contains_set(Z)

    # __eq__
    assert X == X
    assert X != Z

    # element
    assert X.element() is None


def test_universal_set():
    X = UniversalSet()
    Z = Integers()

    # __contains
    assert None in X
    assert 1 in X

    # Contains_set
    assert X.contains_set(X)
    assert X.contains_set(Z)
    assert not X.contains_set(1)

    # __eq__
    assert X == X
    assert X != Z

    # element
    assert X.element() is None
    assert X.element(1) == 1


def test_strings():
    S1 = Strings(1)
    S6 = Strings(6)
    Z = Integers()

    with pytest.raises(ValueError):  # Raises in the conversion to int
        Strings('fail')

    with pytest.raises(ValueError):
        Strings(-2)

    # __contains
    assert 'c' in S1
    assert 'string' not in S1
    assert 'c' not in S6
    assert 'string' in S6

    assert 1 not in S1
    assert 1 not in S6

    # __eq__
    assert S1 == S1
    assert S1 != S6
    assert S6 != S1
    assert S6 == S6
    assert S1 != Z

    # element
    assert S1.element() == ' '
    assert S6.element() == ' ' * 6
    assert S1.element('') == ' '
    assert S6.element('') == ' ' * 6
    assert S1.element('a') == 'a'
    assert S1.element('abcdefg') == 'a'
    assert S6.element('a') == 'a     '
    assert S6.element('abcdefg') == 'abcdef'


def test_complex():
    C = ComplexNumbers()
    R = RealNumbers()
    Z = Integers()

    # __contains__
    assert -1 in C
    assert 1 in C
    assert 0 in C
    assert -1.0 in C
    assert 1.0 in C
    assert 0.0 in C
    assert 2j in C
    assert 2 + 2j in C

    assert 'a' not in C

    # contains_set
    assert C.contains_set(C)
    assert C.contains_set(R)
    assert C.contains_set(Z)

    # __eq__
    assert C == C
    assert C != R
    assert C != Z

    # element
    assert C.element() == complex(0.0, 0.0)
    assert C.element(1) == complex(1.0, 0.0)
    assert C.element(1 + 2j) == complex(1.0, 2.0)


def test_real():
    C = ComplexNumbers()
    R = RealNumbers()
    Z = Integers()

    # __contains__
    assert -1 in R
    assert 1 in R
    assert 0 in R
    assert -1.0 in R
    assert 1.0 in R
    assert 0.0 in R

    assert 2j not in R
    assert 2 + 2j not in R
    assert 'a' not in R

    # contains_set
    assert not R.contains_set(C)
    assert R.contains_set(R)
    assert C.contains_set(Z)

    # __eq__
    assert R != C
    assert R == R
    assert R != Z

    # element
    assert C.element() == float(0.0)
    assert C.element(1) == float(1.0)


def test_integers():
    C = ComplexNumbers()
    R = RealNumbers()
    Z = Integers()

    # __contains__
    assert -1 in Z
    assert 1 in Z
    assert 0 in Z

    assert -1.0 not in Z
    assert 1.0 not in Z
    assert 0.0 not in Z
    assert 2j not in Z
    assert 2 + 2j not in Z
    assert 'a' not in Z

    # contains_set
    assert not Z.contains_set(C)
    assert not Z.contains_set(R)
    assert Z.contains_set(Z)

    # __eq__
    assert Z != C
    assert Z != R
    assert Z == Z

    # element
    assert Z.element() == int(0)
    assert Z.element(1) == int(1)
    assert Z.element(1.5) == int(1.5)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
