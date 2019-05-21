# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Library of operators.

This submodule is a place for operators that either do not fit in any other
places or require "advanced" features of ODL that would make them hard
to put into any other submodule due to circular dependencies.
"""

from __future__ import absolute_import

__all__ = ('convolution', 'ufunc_ops')

from .convolution import *
__all__ += convolution.__all__

from .ufunc_ops import *
__all__ += ufunc_ops.__all__
