# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Testing utilities for the ``tomo`` subpackage."""

from __future__ import absolute_import, division, print_function


__all__ = (
    'skip_if_no_astra',
    'skip_if_no_astra_cuda',
    'skip_if_no_skimage',
)

try:
    import pytest

except ImportError:

    def identity(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        else:
            return identity

    skip_if_no_astra = skip_if_no_astra_cuda = skip_if_no_skimage = identity

else:
    skip_if_no_astra = pytest.mark.skipif(
        'not odl.tomo.ASTRA_AVAILABLE',
        reason='ASTRA not available',
    )
    skip_if_no_astra_cuda = pytest.mark.skipif(
        'not odl.tomo.ASTRA_CUDA_AVAILABLE',
        reason='ASTRA CUDA not available',
    )
    skip_if_no_skimage = pytest.mark.skipif(
        'not odl.tomo.SKIMAGE_AVAILABLE',
        reason='skimage not available',
    )
