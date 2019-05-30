# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for creating phantoms."""

from __future__ import print_function, division, absolute_import
import numpy as np


__all__ = ('cylinders_from_ellipses',)


def cylinders_from_ellipses(ellipses):
    """Create 3d cylinders from ellipses."""
    ellipses = np.asarray(ellipses)
    ellipsoids = np.zeros((ellipses.shape[0], 10))
    ellipsoids[:, [0, 1, 2, 4, 5, 7]] = ellipses
    ellipsoids[:, 3] = 100000.0

    return ellipsoids


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
