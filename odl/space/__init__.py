# Copyright 2014, 2015 The ODL development group
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

"""Concrete vector spaces.

Spaces of n-tuples (modules `ntuples`, `cu_ntuples`)
====================================================

NumPy implementation (module `ntuples`)
---------------------------------------

+----------------------+-----------------------------------------------+
|Name                  |Description                                    |
+======================+===============================================+
|`NTuplesBase`         |**Abstract** base class for sets of n-tuples of|
|                      |various types                                  |
+----------------------+-----------------------------------------------+
|`Ntuples`             |Set of n-tuples of any NumPy supported type    |
+----------------------+-----------------------------------------------+
|`FnBase`              |**Abstract** base class for spaces of n-tuples |
|                      |over the real or complex numbers               |
+----------------------+-----------------------------------------------+
|`Fn`                  |Space of n-tuples over the real or complex     |
|                      |numbers allowing any adequate scalar data type |
+----------------------+-----------------------------------------------+
|`Cn`                  |Space of n-tuples of complex numbers           |
+----------------------+-----------------------------------------------+
|`Rn`                  |Space of n-tuples of real numbers              |
+----------------------+-----------------------------------------------+

CUDA implementation (module `cu_ntuples`)
-----------------------------------------

Requires the compiled extension `odlpp`

+----------------------+-----------------------------------------------+
|Name                  |Description                                    |
+======================+===============================================+
|`CudaNtuples`         |Set of n-tuples of any type supported by the   |
|                      |`odlpp` backend                                |
+----------------------+-----------------------------------------------+
|`CudaFn`              |Space of n-tuples over the real or complex     |
|                      |numbers allowing any adequate scalar data type |
+----------------------+-----------------------------------------------+
|(`CudaCn`)            |Space of n-tuples of complex numbers (TODO)    |
+----------------------+-----------------------------------------------+
|`CudaRn`              |Space of n-tuples of real numbers              |
+----------------------+-----------------------------------------------+

Function spaces (module `fspace`)
=================================

+----------------------+-----------------------------------------------+
|Name                  |Description                                    |
+======================+===============================================+
|`L2`                  |Square-integrable functions taking real or     |
|                      |complex values                                 |
+----------------------+-----------------------------------------------+
"""

from __future__ import absolute_import

__all__ = ()

from . import ntuples
from .ntuples import *
__all__ += ntuples.__all__

from . import default
from .default import *
__all__ += default.__all__

from . import fspace
from .fspace import *
__all__ += fspace.__all__

from . import inner
from .inner import *
__all__ += inner.__all__

try:
    from . import cu_ntuples
    from .cu_ntuples import *
    __all__ += cu_ntuples.__all__
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

__all__ += ('CUDA_AVAILABLE',)
