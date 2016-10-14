# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify odl
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

"""Functions for generating standardized examples in spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from itertools import product

__all__ = ('samples',)


def samples(*sets):
    """Generate samples from the given sets using their ``examples`` method.

    Parameters
    ----------
    *sets : `Set` instance(s)

    Returns
    -------
    samples : `generator`
        Generator that yields tuples of examples from the sets.

    Examples
    --------
    >>> R, C = odl.RealNumbers(), odl.ComplexNumbers()
    >>> for [name_x, x], [name_y, y] in samples(R, C): pass  # use examples
    """
    if len(sets) == 1:
        for example in sets[0].examples:
            yield example
    else:
        generators = [set_.examples for set_ in sets]
        for examples in product(*generators):
            yield examples


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
