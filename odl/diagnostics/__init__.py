# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Automated diagnostic checks."""

from __future__ import absolute_import

from .examples import *
from .operator import *
from .space import *

__all__ = ()
__all__ += examples.__all__
__all__ += operator.__all__
__all__ += space.__all__
