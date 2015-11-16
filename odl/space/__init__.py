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

Spaces of n-tuples (modules :mod:`odl.space.ntuples`, :mod:`~odl.space.cu_ntuples`)
===================================================================================

NumPy implementation (module :mod:`~odl.space.ntuples`)
-------------------------------------------------------

+----------------------+-----------------------------------------------+
|Name                  |Description                                    |
+======================+===============================================+
|:class:`Ntuples`      |Set of n-tuples of any NumPy supported type    |
+----------------------+-----------------------------------------------+
|:class:`FnBase`       |**Abstract** base class for spaces of n-tuples |
|                      |over the real or `complex` numbers             |
+----------------------+-----------------------------------------------+
|:class:`Fn`           |Space of n-tuples over the real or `complex`   |
|                      |numbers allowing any adequate scalar data type |
+----------------------+-----------------------------------------------+
|:class:`Cn`           |Space of n-tuples of `complex` numbers         |
+----------------------+-----------------------------------------------+
|:class:`Rn`           |Space of n-tuples of real numbers              |
+----------------------+-----------------------------------------------+

CUDA implementation (module :mod:`~odl.space.cu_ntuples`)
---------------------------------------------------------

Requires the compiled extension ``odlpp`` #TODO link

+----------------------+-----------------------------------------------+
|Name                  |Description                                    |
+======================+===============================================+
|:class:`CudaNtuples`  |Set of n-tuples of any type supported by the   |
|                      |``odlpp`` backend                              |
+----------------------+-----------------------------------------------+
|:class:`CudaFn`       |Space of n-tuples over the real or `complex`   |
|                      |numbers allowing any adequate scalar data type |
+----------------------+-----------------------------------------------+
|(``CudaCn``)          |Space of n-tuples of `complex` numbers (TODO)  |
+----------------------+-----------------------------------------------+
|:class:`CudaRn`       |Space of n-tuples of real numbers              |
+----------------------+-----------------------------------------------+

Function spaces (module :mod:`~odl.space.fspace`)
=================================================

+----------------------+-----------------------------------------------+
|Name                  |Description                                    |
+======================+===============================================+
|:class:`L2`           |Square-integrable functions taking real or     |
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

try:
    from . import cu_ntuples
    from .cu_ntuples import *
    __all__ += cu_ntuples.__all__
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DTYPES = []

__all__ += ('CUDA_AVAILABLE', 'CUDA_DTYPES')
