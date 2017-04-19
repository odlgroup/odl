# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Automated tests for ODL."""

from __future__ import absolute_import

__all__ = ()

from .examples import *
__all__ += examples.__all__

from .space import *
__all__ += space.__all__

from .operator import *
__all__ += operator.__all__
