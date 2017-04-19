# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

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
