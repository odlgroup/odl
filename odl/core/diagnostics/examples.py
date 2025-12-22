# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Functions for generating standardized examples in spaces."""

from itertools import product

__all__ = ('samples',)


def samples(*sets):
    """Generate samples from the given sets using their ``examples`` method.

    Parameters
    ----------
    set1, ..., setN : `Set` instance
        Set(s) from which to generate the samples.

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
        yield from sets[0].examples
    else:
        generators = [set_.examples for set_ in sets]
        yield from product(*generators)


if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests
    run_doctests()
