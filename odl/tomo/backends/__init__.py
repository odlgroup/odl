# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Back-ends for other libraries."""

from __future__ import absolute_import

__all__ = ()

from . import stir_bindings

from .astra_setup import *
__all__ += astra_setup.__all__

from .astra_cpu import *
__all__ += astra_cpu.__all__

from .astra_cuda import *
__all__ += astra_cuda.__all__

from .skimage_radon import *
__all__ += skimage_radon.__all__
