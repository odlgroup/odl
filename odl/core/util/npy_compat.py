# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Numpy functionality that is not uniform across the supported versions.
"""

import numpy as np

# This is supposed to be used as the `copy=AVOID_UNNECESSARY_COPY` argument to
# the `np.array` constructor when a copy is not desired but may not be avoidable
# (for example because a conversion to a different `dtype` is performed).
# ODL uses this in many places, since it gives usually good performance but
# also still offers flexibility.
# In NumPy-1, the `copy` argument could only be `True` or `False`, the latter
# being merely a request which NumPy-1 ignored if necessary.
# By contrast, NumPy-2 makes `False` binding: if a copy cannot be avoided,
# an error is raised. For the old weak request, NumPy-2 offers `copy=None`
# as the value, which is thus what ODL shall use forward-facing.
# NumPy-1 does however not understand this, which is why the following definition
# is needed for compatibility with both.
AVOID_UNNECESSARY_COPY = None 

__all__ = ("AVOID_UNNECESSARY_COPY",)

if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests
    run_doctests()
