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
from odl.space import FunctionSpace
from odl.operator.operator import Operator
from odl.tomo.geometry.geometry import Geometry
from odl.tomo.backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE,
    astra_cpu_forward_projector, astra_cpu_back_projector,
    astra_cuda_forward_projector, astra_cuda_back_projector)


_SUPPORTED_IMPL = ('astra_cpu', 'astra_cuda')

__all__ = ('XrayTransform', 'XrayBackProjection')


# TODO: DivergentBeamTransform

class XrayTransform(Operator):

    """The discrete X-ray transform between `L^p` spaces."""

    def __init__(self, discr_domain, geometry, impl='astra_cpu', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        discr_domain : `odl.DiscreteLp`
            Discretized space, the domain of the forward projector
        geometry : `Geometry`
            Geometry of the transform, containing information about
            the operator range
        impl : {'astra_cpu', 'astra_cuda'}, optional
            Implementation back-end for the transform. Supported back-ends:
            'astra_cpu': ASTRA toolbox using CPU, only 2D
            'astra_cuda': ASTRA toolbox, using CUDA, 2D or 3D
        interp : {'nearest', 'linear'}
            Interpolation type for the discretization of the operator
            range.
            Default: 'nearest'
        """
        if not isinstance(discr_domain, DiscreteLp):
            raise TypeError('discretized domain {!r} is not a `DiscreteLp`'
                            ' instance.'.format(discr_domain))

        if not isinstance(geometry, Geometry):
            raise TypeError('geometry {!r} is not a `Geometry` instance.'
                            ''.format(geometry))

        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_IMPL:
            raise ValueError('implementation {!r} not supported.'
                             ''.format(impl_in))

        # TODO: sanity checks between impl and discretization impl
        if impl.startswith('astra'):
            if not ASTRA_AVAILABLE:
                raise ValueError('ASTRA back-end not available.')
            if impl == 'astra_cuda' and not ASTRA_CUDA_AVAILABLE:
                raise ValueError('ASTRA CUDA back-end not available.')
            if discr_domain.dspace.dtype not in (np.float32, np.complex64):
                raise ValueError('ASTRA support is limited to `float32` for '
                                 'real and `complex64` for complex data.')
            if not np.allclose(discr_domain.partition.cell_sides[1:],
                               discr_domain.partition.cell_sides[:-1]):
                raise ValueError('ASTRA does not support different voxel '
                                 'sizes per axis, got {}.'
                                 ''.format(discr_domain.partition.cell_sides))

        # TODO: sanity checks between domain and geometry (ndim, ...)
        self._geometry = geometry
        self._impl = impl
        self.kwargs = kwargs

        # Create a discretized space (operator range) with the same data-space
        # type as the domain.
        # TODO: use a ProductSpace structure or find a way to treat different
        # dimensions differently in DiscreteLp (i.e. in partitions).
        range_uspace = FunctionSpace(geometry.params)

        # Approximate cell volume
        # TODO: angles and detector must be handled separately. While the
        # detector should be uniformly discretized, the angles do not have
        # to and often are not.
        extent = float(geometry.partition.extent().prod())
        size = float(geometry.partition.size)
        weight = extent / size

        range_dspace = discr_domain.dspace_type(
            geometry.partition.size, weight=weight,
            dtype=discr_domain.dspace.dtype)

        range_interp = kwargs.get('interp', 'nearest')
        discr_range = DiscreteLp(
            range_uspace, geometry.partition, range_dspace,
            interp=range_interp, order=discr_domain.order)

        super().__init__(discr_domain, discr_range, linear=True)

    @property
    def impl(self):
        """Implementation back-end for evaluation of this operator."""
        return self._impl

    @property
    def geometry(self):
        """Geometry of this operator."""
        return self._geometry

    def _call(self, x, out=None):
        """Apply the operator to ``x`` and store the result in ``out``.

        Parameters
        ----------
        x : `DiscreteLpVector`
           Element in the domain of the operator to be forward projected
        out : `DiscreteLpVector`, optional
            Vector in the projection space to which the result is written.
            If `None` creates an element in the range of the operator.

        Returns
        -------
        out : `DiscreteLpVector`
            Returns an element in the projection space
        """
        backend, data_impl = self.impl.split('_')
        if backend == 'astra':
            if data_impl == 'cpu':
                return astra_cpu_forward_projector(x, self.geometry,
                                                   self.range, out)
            elif data_impl == 'cuda':
                return astra_cuda_forward_projector(x, self.geometry,
                                                    self.range, out)
            else:
                # Should never happen
                raise RuntimeError('implementation info is inconsistent.')
        else:  # Should never happen
            raise RuntimeError('implementation info is inconsistent.')

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return XrayBackProjection(self.domain, self.geometry, self.impl,
                                  **self.kwargs)


class XrayBackProjection(Operator):
    """The adjoint of the discrete X-ray transform between `L^p` spaces."""

    def __init__(self, discr_range, geometry, impl='astra_cpu', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        discr_range : `DiscreteLp`
            Reconstruction space, the range of the back-projector
        geometry : `Geometry`
            The geometry of the transform, contains information about
            the operator domain
        impl : {'astra_cpu', 'astra_cuda'}, optional
            Implementation back-end for the transform. Supported back-ends:
            'astra_cpu': ASTRA toolbox using CPU, only 2D
            'astra_cuda': ASTRA toolbox, using CUDA, 2D or 3D
        interp : {'nearest', 'linear'}
            Interpolation type for the discretization of the operator range.
            Default: 'nearest'
        """
        if not isinstance(discr_range, DiscreteLp):
            raise TypeError('discretized range {!r} is not a `DiscreteLp`'
                            ' instance.'.format(discr_range))

        if not isinstance(geometry, Geometry):
            raise TypeError('geometry {!r} is not a `Geometry` instance.'
                            ''.format(geometry))

        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_IMPL:
            raise ValueError("implementation '{}' not supported."
                             ''.format(impl_in))

        if impl.startswith('astra'):
            if not ASTRA_AVAILABLE:
                raise ValueError('ASTRA backend not available.')
            if impl == 'astra_cuda' and not ASTRA_CUDA_AVAILABLE:
                raise ValueError('ASTRA CUDA backend not available.')
            if discr_range.dspace.dtype not in (np.float32, np.complex64):
                raise ValueError('ASTRA support is limited to `float32` for '
                                 'real and `complex64` for complex data.')
            if not np.allclose(discr_range.partition.cell_sides[1:],
                               discr_range.partition.cell_sides[:-1]):
                raise ValueError('ASTRA does not support different voxel '
                                 'sizes per axis, got {}.'
                                 ''.format(discr_range.partition.cell_sides))

        self._geometry = geometry
        self._impl = impl
        self.kwargs = kwargs

        # Create a discretized space (operator domain) with the same data-space
        # type as the range.
        domain_uspace = FunctionSpace(geometry.params)

        # Approximate cell volume
        extent = float(geometry.partition.extent().prod())
        size = float(geometry.partition.size)
        weight = extent / size

        domain_dspace = discr_range.dspace_type(
            geometry.partition.size, weight=weight,
            dtype=discr_range.dspace.dtype)

        domain_interp = kwargs.get('interp', 'nearest')
        disc_domain = DiscreteLp(
            domain_uspace, geometry.partition, domain_dspace,
            interp=domain_interp, order=discr_range.order)
        super().__init__(disc_domain, discr_range, linear=True)

    @property
    def impl(self):
        """Implementation back-end for evaluation of this operator."""
        return self._impl

    @property
    def geometry(self):
        """Geometry of this operator."""
        return self._geometry

    def _call(self, x, out=None):
        """Apply the operator to ``x`` and store the result in ``out``.

        Parameters
        ----------
        x : `DiscreteLpVector`
           Element in the domain of the operator which is back-projected
        out : `DiscreteLpVector`, optional
            Element in the reconstruction space to which the result is
            written. If `None` an element in the range of the operator is
            created.

        Returns
        -------
        out : `DiscreteLpVector`
            Returns an element in the projection space
        """
        backend, data_impl = self.impl.split('_')
        if backend == 'astra':
            if data_impl == 'cpu':
                return astra_cpu_back_projector(x, self.geometry,
                                                self.range, out)
            elif data_impl == 'cuda':
                return astra_cuda_back_projector(x, self.geometry,
                                                 self.range, out)
            else:
                # Should never happen
                raise RuntimeError('implementation info is inconsistent.')
        else:  # Should never happen
            raise RuntimeError('implementation info is inconsistent.')

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return XrayTransform(self.range, self.geometry, self.backend,
                             **self.kwargs)
