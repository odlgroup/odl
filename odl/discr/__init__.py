# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Discretization-related functionality like grids and discrete spaces."""

from __future__ import absolute_import

__all__ = ()

from .grid import *
__all__ += grid.__all__

from .partition import *
__all__ += partition.__all__

from .lp_discr import *
__all__ += lp_discr.__all__

from .discr_ops import *
__all__ += discr_ops.__all__

from .diff_ops import *
__all__ += diff_ops.__all__

from . import discr_utils
