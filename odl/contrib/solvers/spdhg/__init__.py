# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Contributed code for the stochastic PDHG.

[1] A. Chambolle, M. J. Ehrhardt, P. Richtárik and C.-B. Schönlieb (2017).
Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling and
Imaging Applications. http://arxiv.org/abs/1706.04957

[2] M. J. Ehrhardt, P. J. Markiewicz, P. Richtárik, J. Schott, A. Chambolle
and C.-B. Schönlieb (2017). Faster PET Reconstruction with a Stochastic
Primal-Dual Hybrid Gradient Method. Wavelets and Sparsity XVII, 58
http://doi.org/10.1117/12.2272946.
"""

from __future__ import absolute_import

__all__ = ()

from .misc import *
__all__ += misc.__all__

from .stochastic_primal_dual_hybrid_gradient import *
__all__ += stochastic_primal_dual_hybrid_gradient.__all__
