# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

__all__ = ('skip_if_no_astra', 'skip_if_no_astra_cuda', 'skip_if_no_skimage')

try:
    import pytest
except ImportError:
    # Use the identity decorator (default of OptionalArgDecorator)
    from odl.util import OptionalArgDecorator as ident
    skip_if_no_astra = skip_if_no_astra_cuda = skip_if_no_skimage = ident

else:
    skip_if_no_astra = pytest.mark.skipif('not odl.tomo.ASTRA_AVAILABLE',
                                          reason='ASTRA not available')
    skip_if_no_astra_cuda = pytest.mark.skipif(
        'not odl.tomo.ASTRA_CUDA_AVAILABLE', reason='ASTRA CUDA not available')
    skip_if_no_skimage = pytest.mark.skipif(
        'not odl.tomo.SKIMAGE_AVAILABLE', reason='skimage not available')
