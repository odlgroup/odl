# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tomography-related operators and geometries."""

from __future__ import absolute_import

from .analytic import *
from .backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE, SKIMAGE_AVAILABLE,
    cone_2d_geom_to_astra_vecs, cone_3d_geom_to_astra_vecs,
    parallel_2d_geom_to_astra_vecs, parallel_3d_geom_to_astra_vecs,
    vecs_astra_to_odl_coords, vecs_odl_to_astra_coords)
from .geometry import *
from .operators import *
from .util import *

__all__ = ()
__all__ += analytic.__all__
__all__ += geometry.__all__
__all__ += operators.__all__
__all__ += util.source_detector_shifts.__all__
__all__ += (
    'ASTRA_AVAILABLE',
    'ASTRA_CUDA_AVAILABLE',
    'SKIMAGE_AVAILABLE',
    'vecs_astra_to_odl_coords',
    'vecs_odl_to_astra_coords',
    'parallel_2d_geom_to_astra_vecs',
    'parallel_3d_geom_to_astra_vecs',
    'cone_2d_geom_to_astra_vecs',
    'cone_3d_geom_to_astra_vecs'
)
