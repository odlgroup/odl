# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Basic vector spaces and utilities."""

from __future__ import absolute_import

__all__ = ()

from . import base_tensors
from . import entry_points
from . import weighting

from .npy_tensors import *
__all__ += npy_tensors.__all__

from .pspace import *
__all__ += pspace.__all__

from .space_utils import *
__all__ += space_utils.__all__
