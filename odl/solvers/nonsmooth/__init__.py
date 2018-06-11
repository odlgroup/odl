# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Solvers for non-smooth optimization problems."""

from __future__ import absolute_import

__all__ = ()

from .proximal_operators import *
__all__ += proximal_operators.__all__

from .admm import *
__all__ += admm.__all__

from .primal_dual_hybrid_gradient import *
__all__ += primal_dual_hybrid_gradient.__all__

from .douglas_rachford import *
__all__ += douglas_rachford.__all__

from .forward_backward import *
__all__ += forward_backward.__all__

from .proximal_gradient_solvers import *
__all__ += proximal_gradient_solvers.__all__

from .alternating_dual_updates import *
__all__ += alternating_dual_updates.__all__

from .difference_convex import *
__all__ += difference_convex.__all__
