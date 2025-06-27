# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Python Array API support."""

from __future__ import absolute_import

from .element_wise import *
from .reductions import *
from .linalg import *
from .utils import *
from .comparisons import *

__all__ = ()
__all__ += element_wise.__all__
__all__ += reductions.__all__
__all__ += linalg.__all__
__all__ += utils.__all__
__all__ += comparisons.__all__
