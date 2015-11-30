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

Back and forward projectors for PET.

See the STIR `webpage
<http://stir.sourceforge.net/>`_ for more information.
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

    """ A forward projector using STIR input files.

    Uses "ForwardProjectorByBinUsingProjMatrixByBin" as a projector.
    """

    def __init__(self, dom, ran, volume_file, projection_file):
        """ Initialize a new projector.

        Parameters
        ----------
        dom : `DiscreteLp`
            Volume of the projection. Needs to have the same shape as given
            by the ini file ``data_template``.
        ran : `DiscreteLp`
            Projection space. Needs to have the same shape as given by the
            geometry in ``projection_template`` when converted to an array
            using the projection_data when cast using ``to_array``.
        volume_file : `str`, STIR input file path
            An interfile of type '.hv' giving a description of the volume
            geometry.
        projection_file : `str`, STIR input file path
            An interfile of type '.hs' giving a description of the projection
            (volume) geometry.

        """
        # Read template of the projection
        proj_data_in = stir.ProjData.read_from_file(projection_file)
        self.proj_data_info = proj_data_in.get_proj_data_info()
        self.proj_data = stir.ProjDataInMemory(proj_data_in.get_exam_info(),
                                               proj_data_in.get_proj_data_info())

        # Read template data for the volume
        self.volume = stir.FloatVoxelsOnCartesianGrid.read_from_file(volume_file)

        # Create forward projection by matrix
        self.proj_matrix = stir.ProjMatrixByBinUsingRayTracing()
        self.proj_matrix.set_up(self.proj_data_info, self.volume)
        self.projector = stir.ForwardProjectorByBinUsingProjMatrixByBin(self.proj_matrix)
        self.projector.set_up(self.proj_data_info, self.volume)

        # Check data sizes
        assert dom.shape == self.volume.shape()
        assert ran.shape == self.proj_data.to_array().shape()

        super().__init__(dom, ran, True)

        self._adjoint = StirProjectorFromFileAdjoint(self.range, self.domain,
                                                     self.proj_matrix,
                                                     self.volume, self.proj_data)

    def _call(self, volume):
        """Forward project a volume."""
        # Set volume data
        self.volume.fill(volume.asarray().flat)

        # project
        old_verbosity = stir.Verbosity.get()
        stir.Verbosity.set(0)
        self.projector.forward_project(self.proj_data, self.volume)
        stir.Verbosity.set(old_verbosity)

        # make odl data
        arr = stir.stirextra.to_numpy(self.proj_data)

        return self.range.element(arr)

    @property
    def adjoint(self):
        """The back-projector associated with this array."""
        return self._adjoint


class StirProjectorFromFileAdjoint(Operator):

    """The adjoint of a `StirProjectorFromFile`.

    Use the `StirProjectorFromFile.adjoint` method to create new objects.
    """

    def __init__(self, dom, ran, proj_matrix, volume, projections):
        """Initialize a new back-projector."""
        super().__init__(dom, ran, True)
        self.volume = volume
        self.proj_data = projections

        self.back_proj = stir.BackProjectorByBinUsingProjMatrixByBin(proj_matrix)
        self.back_proj.set_up(self.proj_data.get_proj_data_info(), self.volume)

    def _call(self, projections):
        """Back project."""
        # Set projection data
        self.proj_data.fill(projections.asarray().flat)

        # back-project
        old_verbosity = stir.Verbosity.get()
        stir.Verbosity.set(0)
        self.back_proj.back_project(self.volume, self.proj_data)
        stir.Verbosity.set(old_verbosity)

        # make odl data
        arr = stir.stirextra.to_numpy(self.volume)

        return self.range.element(arr)
