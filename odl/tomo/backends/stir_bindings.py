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

from odl.operator.operator import Operator

try:
    import stir
    STIR_AVAILABLE = True
except ImportError:
    STIR_AVAILABLE = False

__all__ = ('StirProjectorFromFile', 'STIR_AVAILABLE')


class StirProjectorFromFile(Operator):

    """ A forward projector using STIR input files """

    def __init__(self, dom, ran, data_template, projection_template):
        # Read template of the projection
        proj_data_in = stir.ProjData.read_from_file(projection_template)
        self.proj_data_info = proj_data_in.get_proj_data_info()
        self.proj_data = stir.ProjDataInMemory(proj_data_in.get_exam_info(),
                                               proj_data_in.get_proj_data_info())

        # Read template data for the volume
        self.volume = stir.FloatVoxelsOnCartesianGrid.read_from_file(data_template)

        # Create forward projection by matrix
        self.proj_matrix = stir.ProjMatrixByBinUsingRayTracing()
        self.proj_matrix.set_up(self.proj_data_info, self.volume)
        self.projector = stir.ForwardProjectorByBinUsingProjMatrixByBin(self.proj_matrix)
        self.projector.set_up(self.proj_data_info, self.volume)

        assert dom.shape == self.volume.shape()
        assert ran.shape == self.proj_data.to_array().shape()

        super().__init__(dom, ran, True)

    def _call(self, volume):
        # Set volume data
        self.volume.fill(volume.asarray().flat)

        # project
        self.projector.forward_project(self.proj_data, self.volume)

        # make odl
        arr = stir.stirextra.to_numpy(self.proj_data)
        proj = self.range.element(arr)

        return proj

    @property
    def adjoint(self):
        """The back-projector associated with this array."""
        return StirProjectorFromFileAdjoint(self.range, self.domain,
                                            self.proj_matrix,
                                            self.volume, self.proj_data)


class StirProjectorFromFileAdjoint(Operator):

    """The adjoint of a `StirProjectorFromFile`.

    Use the `StirProjectorFromFile.adjoint` method to create new objects.
    """

    def __init__(self, dom, ran, proj_matrix, volume, projections):
        super().__init__(dom, ran, True)
        self.volume = volume
        self.proj_data = projections

        self.back_proj = stir.BackProjectorByBinUsingProjMatrixByBin(proj_matrix)
        self.back_proj.set_up(self.proj_data.get_proj_data_info(), self.volume)

    def _call(self, projections):
        # Set projection data
        self.proj_data.fill(projections.asarray().flat)

        # back-project
        self.back_proj.back_project(self.volume, self.proj_data)

        # make odl
        arr = stir.stirextra.to_numpy(self.volume)
        vol = self.range.element(arr)

        return vol
