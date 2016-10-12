# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Solvers for non-smooth optimization problems."""

from __future__ import absolute_import

__all__ = ()

from .proximal_operators import *
__all__ += proximal_operators.__all__

from .chambolle_pock import *
__all__ += chambolle_pock.__all__

from .douglas_rachford import *
__all__ += douglas_rachford.__all__

from .forward_backward import *
__all__ += forward_backward.__all__
