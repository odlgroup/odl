# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Linear Algebra functions expected by the python array API.
Note: This is not obvious that we should actually support it.
"""

__all__ = ('vecdot',)

def vecdot(x1, x2, axis=-1, out = None):
    """Computes the (vector) dot product of two arrays."""
    raise NotImplementedError("WIP")