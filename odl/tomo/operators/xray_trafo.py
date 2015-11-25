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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from future.builtins import str, super

# External
import numpy as np
from odl import (CUDA_AVAILABLE, DiscreteLp, FunctionSpace, Operator)
if CUDA_AVAILABLE:
    from odl import CudaNtuples
else:
    CudaNtuples = type(None)

# Internal
from odl.tomo.geometry.geometry import Geometry
from odl.tomo.backends import ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE
if ASTRA_AVAILABLE:
    from odl.tomo.backends.astra_cpu import (
        astra_cpu_forward_projector_call, astra_cpu_backward_projector_call)
else:
    astra_cpu_forward_projection_call = None
    astra_cpu_backward_projector_call = None
if ASTRA_CUDA_AVAILABLE:
    from odl.tomo.backends.astra_cuda import (
        astra_gpu_forward_projector_call, astra_gpu_backward_projector_call)
else:
    astra_gpu_forward_projector_call = None
    astra_gpu_backward_projector_call = None

_SUPPORTED_BACKENDS = ('astra',)

__all__ = ('DiscreteXrayTransform',)


class DiscreteXrayTransform(Operator):

    """The discrete X-ray transform between :math:`L^p` spaces."""

    def __init__(self, discr_dom, geometry, backend='astra', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        discr_dom : `odl.DiscreteLp`
            Discretization of a two-dimensional space, the domain of
            the discretized operator
        geometry : `Geometry`
            The geometry of the transform, contains information about
            the operator range. It needs to have a sampling grid for
            motion and detector parameters.
        backend : {'astra'}
            Implementation backend for the transform. Supported backends:
            'astra': ASTRA toolbox, CPU or CUDA
        kwargs : {'range_interpolation'}
            'range_interpolation' : {'nearest', 'linear', 'cubic'}
                Interpolation type for the discretization of the
                operator range.
                Default: 'nearest'
        """
        if not isinstance(discr_dom, DiscreteLp):
            raise TypeError('discretized domain {!r} is not a `DiscreteLp`'
                            ' instance.'.format(discr_dom))

        if not isinstance(geometry, Geometry):
            raise TypeError('geometry {!r} is not a `Geometry` instance.'
                            ''.format(geometry))

        if not (geometry.has_motion_sampling and geometry.has_det_sampling):
            raise ValueError('geometry {} does not have sampling grids for '
                             'both motion and detector.'.format(geometry))

        backend = str(backend).lower()
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError('backend {!r} not supported.'
                             ''.format(backend))

        if backend == 'astra':
            if not ASTRA_AVAILABLE:
                raise ValueError('ASTRA backend not available.')
            if (isinstance(discr_dom.dspace, CudaNtuples) and
                    not ASTRA_CUDA_AVAILABLE):
                raise ValueError('ASTRA CUDA backend not available.')
            if discr_dom.dspace.dtype not in (np.float32, np.complex64):
                raise ValueError('ASTRA support is limited to `float32` '
                                 'for real and `complex64` for complex '
                                 'data.')
            if not np.allclose(discr_dom.grid.stride[1:],
                               discr_dom.grid.stride[:-1]):
                raise ValueError('ASTRA does not support different pixel/voxel'
                                 ' sizes per axis (got {}).'
                                 ''.format(discr_dom.grid.stride))

        self._geometry = geometry

        # Create discretized space (operator range). Use same data space
        # type as the domain.
        # TODO: maybe use a ProductSpace structure
        ran_uspace = FunctionSpace(geometry.params)
        ran_dspace = discr_dom.dspace_type(geometry.grid.ntotal,
                                           dtype=discr_dom.dspace.dtype)

        ran_interp = kwargs.pop('range_interpolation', 'nearest')
        discr_ran = DiscreteLp(ran_uspace, geometry.grid, ran_dspace,
                               interp=ran_interp, order=geometry.grid.order)
        super().__init__(discr_dom, discr_ran, linear=True)

        if backend == 'astra' and isinstance(discr_dom.dspace, CudaNtuples):
            self._backend = 'astra_cuda'
        else:
            self._backend = 'astra_cpu'

    @property
    def backend(self):
        """Computational backend for this operator."""
        return self._backend

    @property
    def geometry(self):
        """Geometry of this operator."""
        return self._geometry

    def _call(self, inp):
        """Call the transform on an input, producing a new vector."""
        back, impl = self.backend.split('_')
        if back == 'astra':
            if impl == 'cpu':
                return astra_cpu_forward_projector_call(inp, self.geometry,
                                                        self.range)
            elif impl == 'cuda':
                return astra_gpu_forward_projector_call(inp, self.geometry,
                                                        self.range)
            else:
                raise ValueError('unkown implementation {}.'.format(impl))
        else:  # Should never happen
            raise RuntimeError('backend support information is inconsistent.')
