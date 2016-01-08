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

"""X-ray transforms."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from future.builtins import str, super

# External
import numpy as np

# Internal
from odl.discr.lp_discr import DiscreteLp
from odl.space import FunctionSpace, Ntuples, CudaNtuples
from odl.operator.operator import Operator
from odl.tomo.geometry.geometry import Geometry
from odl.tomo.backends.astra_setup import ASTRA_AVAILABLE
from odl.tomo.backends.astra_cuda import ASTRA_CUDA_AVAILABLE
from odl.tomo.backends.astra_cpu import (
    astra_cpu_forward_projector_call, astra_cpu_backward_projector_call)
from odl.tomo.backends.astra_cuda import (
    astra_cuda_forward_projector_call, astra_cuda_backward_projector_call)

_SUPPORTED_BACKENDS = ('astra', 'astra_cpu', 'astra_cuda')

__all__ = ('DiscreteXrayTransform', 'DiscreteXrayTransformAdjoint')


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
        backend : {'astra', 'astra_cuda', 'astra_cpu'}, optional
            Implementation backend for the transform. Supported backends:
            'astra': ASTRA toolbox, uses CPU or CUDA depending on the
            underlying data space of ``discr_dom``
            'astra_cpu': ASTRA toolbox using CPU, only 2D
            'astra_cuda': ASTRA toolbox, using CUDA, 2D or 3D
            Default: 'astra'
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
            if isinstance(discr_dom.dspace, CudaNtuples):
                self._backend = 'astra_cuda'
            elif isinstance(discr_dom.dspace, Ntuples):
                self._backend = 'astra_cpu'
            else:
                raise TypeError('discr_dom.dspace {} must be a CudaNtuples '
                                'or a Ntuples'.format(discr_dom.dspace))
        else:
            self._backend = backend

        if self.backend.startswith('astra'):
            if not ASTRA_AVAILABLE:
                raise ValueError('ASTRA backend not available.')
            if not ASTRA_CUDA_AVAILABLE and self.backend == 'astra_cuda':
                raise ValueError('ASTRA CUDA backend not available.')
            if discr_dom.dspace.dtype not in (np.float32, np.complex64):
                raise ValueError('ASTRA support is limited to `float32` for '
                                 'real and `complex64` for complex data.')
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
        # CHECKME: Is this the right weight?
        ran_dspace = discr_dom.dspace_type(geometry.grid.ntotal,
                                           weight=geometry.grid.cell_volume,
                                           dtype=discr_dom.dspace.dtype)

        ran_interp = kwargs.pop('range_interpolation', 'nearest')
        discr_ran = DiscreteLp(ran_uspace, geometry.grid, ran_dspace,
                               interp=ran_interp, order=geometry.grid.order)
        super().__init__(discr_dom, discr_ran, linear=True)

        self._adjoint = DiscreteXrayTransformAdjoint(self)

    @property
    def backend(self):
        """Computational backend for this operator."""
        return self._backend

    @property
    def geometry(self):
        """Geometry of this operator."""
        return self._geometry

    def _call(self, inp):
        """Call the transform on an input, producing a new vector.

        Parameters
        ----------
        inp : `DiscreteLpVector`
           Element in the domain of the operator to be forward projected

        Returns
        -------
        out : `DiscreteLpVector`
            Returns an element in the projection space
        """
        back, impl = self.backend.split('_')
        if back == 'astra':
            if impl == 'cpu':
                return astra_cpu_forward_projector_call(
                    inp, self.geometry, self.range)
            elif impl == 'cuda':
                return astra_cuda_forward_projector_call(
                    inp, self.geometry, self.range)
            else:
                raise ValueError('unknown implementation {}.'.format(impl))
        else:  # Should never happen
            raise RuntimeError('backend support information is inconsistent.')

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return self._adjoint


class DiscreteXrayTransformAdjoint(Operator):

    """The adjoint of the discrete X-ray transform."""

    def __init__(self, forward):
        """Initialize a new instance.

        Parameters
        ----------
        forward : `DiscreteXrayTransform`
            An instance of the discrete X-ray transform
        """
        self.forward = forward
        super().__init__(forward.range, forward.domain, forward.is_linear)
        self._backend = forward.backend

    def _call(self, inp):
        """Call the adjoint transform on an input, producing a new vector.

        Parameters
        ----------
        inp : `DiscreteLpVector`
            Element in the domain of the operator to be back-projected

        Returns
        -------
        out : `DiscreteLpVector`
            Returns an element in the reconstruction space
        """
        back, impl = self.backend.split('_')
        if back == 'astra':
            if impl == 'cpu':
                return astra_cpu_backward_projector_call(
                    inp, self.forward.geometry, self.range)
            elif impl == 'cuda':
                return astra_cuda_backward_projector_call(
                    inp, self.forward.geometry, self.range)
            else:
                raise ValueError('unknown implementation {}.'.format(impl))
        else:  # Should never happen
            raise RuntimeError('backend support information is inconsistent.')

    @property
    def backend(self):
        """Computational backend for this operator."""
        return self._backend

    @property
    def adjoint(self):
        """Return the adjoint operator. """
        return self.forward
