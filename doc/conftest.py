# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test configuration file."""

from __future__ import print_function, division, absolute_import

import os
import sys

from odl.trafos.backends import PYFFTW_AVAILABLE, PYWT_AVAILABLE
from odl.tomo.backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE, SKIMAGE_AVAILABLE)

# We don't bother excluding stuff depending on available backends. We just
# require them all (except stir)
if not all([PYFFTW_AVAILABLE, PYWT_AVAILABLE,
            ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE, SKIMAGE_AVAILABLE]):
    print('to be able to test the docs, you need to install all backends, '
          'currently: '
          '`astra`, `pyfftw`, `pywt`, `skimage`; '
          'please consult the installation documentation for details',
          file=sys.stderr)
    sys.exit(-1)

here = os.path.normpath(os.path.dirname(__file__))
collect_ignore = [os.path.join(here, 'sphinxext')]
collect_ignore = [os.path.normcase(ignored) for ignored in collect_ignore]


def pytest_ignore_collect(path, config):
    normalized = os.path.normcase(str(path))
    return any(normalized.startswith(ignored) for ignored in collect_ignore)
