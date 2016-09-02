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

"""Ray transforms."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str, super

import numpy as np

from odl.discr import DiscreteLp
from odl.operator import Operator
from odl.space import FunctionSpace
from odl.tomo.geometry import Geometry, Parallel2dGeometry
from odl.tomo.backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE,
    astra_cpu_forward_projector, astra_cpu_back_projector,
    astra_cuda_forward_projector, astra_cuda_back_projector,
    scikit_radon_forward, scikit_radon_back_projector)

_SUPPORTED_IMPL = ('astra_cpu', 'astra_cuda', 'scikit')


__all__ = ('RayTransform', 'RayBackProjection')


# TODO: DivergentBeamTransform?

class RayTransform(Operator):

    """Discrete Ray transform between L^p spaces."""

    def __init__(self, discr_domain, geometry, impl='astra_cpu', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        discr_domain : `DiscreteLp`
            Discretized space, the domain of the forward projector
        geometry : `Geometry`
            Geometry of the transform, containing information about
            the operator range
        impl : {'astra_cpu', 'astra_cuda', 'scikit'}, optional
            Implementation back-end for the transform. Supported back-ends:
            'astra_cpu': ASTRA toolbox using CPU, only 2D
            'astra_cuda': ASTRA toolbox, using CUDA, 2D or 3D
            'scikit': scikit-image, only 2D parallel with square domain
        interp : {'nearest', 'linear'}
            Interpolation type for the discretization of the operator
            range.
            Default: 'nearest'

        Other Parameters
        ----------------
        discr_range : `DiscreteLp`
            Discretized space, the range of the forward projector.
            Default: Infered from parameters.

        Notes
        -----
        The ASTRA backend is faster if data is given with ``dtype`` 'float32'
        and storage order 'C'. Otherwise copies will be needed.
        """
        if not isinstance(discr_domain, DiscreteLp):
            raise TypeError('`discr_domain` {!r} is not a `DiscreteLp`'
                            ' instance'.format(discr_domain))

        if not isinstance(geometry, Geometry):
            raise TypeError('`geometry` {!r} is not a `Geometry` instance'
                            ''.format(geometry))

        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_IMPL:
            raise ValueError('`impl` {!r} not supported'
                             ''.format(impl_in))

        # TODO: sanity checks between impl and discretization impl
        if impl.startswith('astra'):
            # TODO: these should be moved somewhere else
            if not ASTRA_AVAILABLE:
                raise ValueError("'astra' back-end not available")
            if impl == 'astra_cuda' and not ASTRA_CUDA_AVAILABLE:
                raise ValueError("'astra_cuda' back-end not available")
            if not np.allclose(discr_domain.partition.cell_sides[1:],
                               discr_domain.partition.cell_sides[:-1]):
                raise ValueError('ASTRA does not support different voxel '
                                 'sizes per axis, got {}'
                                 ''.format(discr_domain.partition.cell_sides))
            if geometry.ndim > 2 and impl.endswith('cpu'):
                raise ValueError('`impl` {}, only works for 2d geometries'
                                 ' got {}-d'.format(impl_in, geometry))
        elif impl == 'scikit':
            if not isinstance(geometry, Parallel2dGeometry):
                raise TypeError("'scikit' backend only supports 2d parallel "
                                'geometries')

            mid_pt = discr_domain.domain.mid_pt
            if not all(mid_pt == [0, 0]):
                raise ValueError('`discr_domain.domain` needs to be '
                                 'centered on [0, 0], got {}'.format(mid_pt))

            shape = discr_domain.shape
            if shape[0] != shape[1]:
                raise ValueError('`discr_domain.shape` needs to be square '
                                 'got {}'.format(shape))

            extent = discr_domain.domain.extent()
            if extent[0] != extent[1]:
                raise ValueError('`discr_domain.extent` needs to be square '
                                 'got {}'.format(extent))

        # TODO: sanity checks between domain and geometry (ndim, ...)
        self.__geometry = geometry
        self.__impl = impl
        self.kwargs = kwargs

        discr_range = kwargs.pop('discr_range', None)
        if discr_range is None:
            dtype = discr_domain.dspace.dtype

            # Create a discretized space (operator range) with the same
            # data-space type as the domain.
            # TODO: use a ProductSpace structure or find a way to treat
            # different dimensions differently in DiscreteLp
            # (i.e. in partitions).
            range_uspace = FunctionSpace(geometry.params,
                                         out_dtype=dtype)

            # Approximate cell volume
            # TODO: angles and detector must be handled separately. While the
            # detector should be uniformly discretized, the angles do not have
            # to and often are not.
            extent = float(geometry.partition.extent().prod())
            size = float(geometry.partition.size)
            weight = extent / size

            range_dspace = discr_domain.dspace_type(geometry.partition.size,
                                                    weight=weight, dtype=dtype)

            range_interp = kwargs.get('interp', 'nearest')
            discr_range = DiscreteLp(
                range_uspace, geometry.partition, range_dspace,
                interp=range_interp, order=discr_domain.order)

        super().__init__(discr_domain, discr_range, linear=True)

    @property
    def impl(self):
        """Implementation back-end for evaluation of this operator."""
        return self.__impl

    @property
    def geometry(self):
        """Geometry of this operator."""
        return self.__geometry

    def _call(self, x, out=None):
        """Forward project ``x`` and store the result in ``out`` if given."""
        if self.impl.startswith('astra'):
            backend, data_impl = self.impl.split('_')
            if data_impl == 'cpu':
                return astra_cpu_forward_projector(x, self.geometry,
                                                   self.range, out)
            elif data_impl == 'cuda':
                return astra_cuda_forward_projector(x, self.geometry,
                                                    self.range, out)
            else:
                # Should never happen
                raise RuntimeError('implementation info is inconsistent')
        elif self.impl == 'scikit':
            return scikit_radon_forward(x, self.geometry, self.range, out)
        else:  # Should never happen
            raise RuntimeError('implementation info is inconsistent')

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `RayBackProjection`
        """
        kwargs = self.kwargs.copy()
        kwargs['discr_domain'] = self.range
        return RayBackProjection(self.domain, self.geometry, self.impl,
                                 **kwargs)


class RayBackProjection(Operator):
    """Adjoint of the discrete Ray transform between L^p spaces."""

    def __init__(self, discr_range, geometry, impl='astra_cpu', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        discr_range : `DiscreteLp`
            Reconstruction space, the range of the back-projector
        geometry : `Geometry`
            The geometry of the transform, contains information about
            the operator domain
        impl : {'astra_cpu', 'astra_cuda', 'scikit'}, optional
            Implementation back-end for the transform. Supported back-ends:
            'astra_cpu': ASTRA toolbox using CPU, only 2D
            'astra_cuda': ASTRA toolbox, using CUDA, 2D or 3D
            'scikit': scikit-image, only 2D parallel with square domain
        interp : {'nearest', 'linear'}
            Interpolation type for the discretization of the operator range.
            Default: 'nearest'

        Other Parameters
        ----------------
        discr_domain : `DiscreteLp`
            Discretized space, the range of the forward projector.
            Default: Infered from parameters.
        """
        if not isinstance(discr_range, DiscreteLp):
            raise TypeError('`discr_range` {!r} is not a `DiscreteLp`'
                            ' instance'.format(discr_range))

        if not isinstance(geometry, Geometry):
            raise TypeError('`geometry` {!r} is not a `Geometry` instance'
                            ''.format(geometry))

        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_IMPL:
            raise ValueError("`impl` '{}' not supported"
                             ''.format(impl_in))

        if impl.startswith('astra'):
            if not ASTRA_AVAILABLE:
                raise ValueError("'astra' backend not available")
            if impl == 'astra_cuda' and not ASTRA_CUDA_AVAILABLE:
                raise ValueError("'astra_cuda' backend not available")
            if not np.allclose(discr_range.partition.cell_sides[1:],
                               discr_range.partition.cell_sides[:-1]):
                raise ValueError('ASTRA does not support different voxel '
                                 'sizes per axis, got {}'
                                 ''.format(discr_range.partition.cell_sides))

        self.__geometry = geometry
        self.__impl = impl
        self.kwargs = kwargs

        discr_domain = kwargs.pop('discr_domain', None)
        if discr_domain is None:
            dtype = discr_range.dspace.dtype

            # Create a discretized space (operator domain) with the same
            # data-space type as the range.
            domain_uspace = FunctionSpace(geometry.params, out_dtype=dtype)

            # Approximate cell volume
            extent = float(geometry.partition.extent().prod())
            size = float(geometry.partition.size)
            weight = extent / size

            domain_dspace = discr_range.dspace_type(geometry.partition.size,
                                                    weight=weight, dtype=dtype)

            domain_interp = kwargs.get('interp', 'nearest')
            discr_domain = DiscreteLp(
                domain_uspace, geometry.partition, domain_dspace,
                interp=domain_interp, order=discr_range.order)
        super().__init__(discr_domain, discr_range, linear=True)

    @property
    def impl(self):
        """Implementation back-end for evaluation of this operator."""
        return self.__impl

    @property
    def geometry(self):
        """Geometry of this operator."""
        return self.__geometry

    def _call(self, x, out=None):
        """Back-project ``x`` and store the result in ``out`` if given."""

        if self.impl.startswith('astra'):
            backend, data_impl = self.impl.split('_')
            if data_impl == 'cpu':
                return astra_cpu_back_projector(x, self.geometry,
                                                self.range, out)
            elif data_impl == 'cuda':
                return astra_cuda_back_projector(x, self.geometry,
                                                 self.range, out)
            else:
                # Should never happen
                raise RuntimeError('implementation info is inconsistent')
        elif self.impl == 'scikit':
            return scikit_radon_back_projector(x, self.geometry,
                                               self.range, out)
        else:  # Should never happen
            raise RuntimeError('implementation info is inconsistent')

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `RayTransform`
        """
        kwargs = self.kwargs.copy()
        kwargs['discr_domain'] = self.domain
        return RayTransform(self.range, self.geometry, impl=self.impl,
                            **kwargs)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
