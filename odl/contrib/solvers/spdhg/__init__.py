# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Contributed code for the stochastic PDHG.

[CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
*Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).

[E+2017] M. J. Ehrhardt, P. J. Markiewicz, P. Richtarik, J. Schott,
A. Chambolle and C.-B. Schoenlieb, *Faster PET reconstruction with a
stochastic primal-dual hybrid gradient method*. Wavelets and Sparsity XVII,
58 (2017) http://doi.org/10.1117/12.2272946.
"""

from __future__ import absolute_import

__all__ = ()

from .misc import *
__all__ += misc.__all__

from .stochastic_primal_dual_hybrid_gradient import *
__all__ += stochastic_primal_dual_hybrid_gradient.__all__
