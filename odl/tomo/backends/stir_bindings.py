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


"""Backend for STIR: Software for Tomographic Reconstruction
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.set.sets import UniversalSet
from odl.operator.operator import Operator

try:
    import stir
    STIR_AVAILABLE = True
except ImportError:
    STIR_AVAILABLE = False

__all__ = ('StirProjectorFromFile', 'STIR_AVAILABLE')


class StirProjectorFromFile(Operator):
    def __init__(self, data_template, projection_template):
        # Read template of the projection
        proj_data_in = stir.ProjData.read_from_file(projection_template)
        self.proj_data_info = proj_data_in.get_proj_data_info()
        self.proj_data = stir.ProjDataInMemory(proj_data_in.get_exam_info(),
                                               proj_data_in.get_proj_data_info())

        # Read template data for the volume
        self.volume = stir.FloatVoxelsOnCartesianGrid.read_from_file(data_template)

        # Create forward projection by matrix
        proj_matrix = stir.ProjMatrixByBinUsingRayTracing()
        proj_matrix.set_up(self.proj_data_info, self.volume)
        self.projector = stir.ForwardProjectorByBinUsingProjMatrixByBin(proj_matrix)
        self.projector.set_up(self.proj_data_info, self.volume)

        super().__init__(UniversalSet(), UniversalSet())

    def _call(self, volume):
        # Set volume data
        asarr = volume.asarray()
        shape = self.volume.shape()
        assert shape == volume.shape

        for x in range(self.volume.get_min_indices()[1],
                       self.volume.get_max_indices()[1]+1):
            for y in range(self.volume.get_min_indices()[2],
                           self.volume.get_max_indices()[2]+1):
                for z in range(self.volume.get_min_indices()[3],
                               self.volume.get_max_indices()[3]+1):
                    self.volume[(x, y, z)] = asarr[x, y, z]

        # project
        self.projector.forward_project(self.proj_data, self.volume)

        return self.proj_data


