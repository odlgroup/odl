# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Solvers for non-smooth optimization problems."""

from .admm import *
from .alternating_dual_updates import *
from .difference_convex import *
from .douglas_rachford import *
from .forward_backward import *
from .primal_dual_hybrid_gradient import *
from .proximal_gradient_solvers import *
from .proximal_operators import *

__all__ = ()
__all__ += admm.__all__
__all__ += alternating_dual_updates.__all__
__all__ += difference_convex.__all__
__all__ += douglas_rachford.__all__
__all__ += forward_backward.__all__
__all__ += primal_dual_hybrid_gradient.__all__
__all__ += proximal_gradient_solvers.__all__
__all__ += proximal_operators.__all__
