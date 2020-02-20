# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ray transforms."""

from __future__ import absolute_import, division, print_function

import numpy as np

from odl.discr import DiscretizedSpace
from odl.operator import Operator
from odl.space.weighting import ConstWeighting
from odl.tomo.backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE, SKIMAGE_AVAILABLE)
from odl.tomo.backends.astra_cpu import AstraCpu
from odl.tomo.backends.astra_cuda import AstraCuda
from odl.tomo.backends.skimage_radon import Skimage
from odl.tomo.geometry import Geometry

# Backends that are implemented in ODL and can be specified via `impl`
_SUPPORTED_IMPL = {
    'astra_cpu': AstraCpu,
    'astra_cuda': AstraCuda,
    'skimage': Skimage}

# Backends that are installed, ordered by preference
_AVAILABLE_IMPLS = []
if ASTRA_CUDA_AVAILABLE:
    _AVAILABLE_IMPLS.append('astra_cuda')
if ASTRA_AVAILABLE:
    _AVAILABLE_IMPLS.append('astra_cpu')
if SKIMAGE_AVAILABLE:
    _AVAILABLE_IMPLS.append('skimage')

__all__ = ('RayTransform',)


class RayTransform(Operator):
    """Linear X-Ray (Radon) transform operator between L^p spaces."""

    def __init__(self, reco_space, geometry, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        reco_space : `DiscretizedSpace`
            Discretized reconstruction space, the domain of the forward
            operator or the range of the adjoint (back-projection).
        geometry : `Geometry`
            Geometry of the transform that contains information about
            the data structure.

        Other Parameters
        ----------------
        impl : {`None`, 'astra_cuda', 'astra_cpu', 'skimage'}, optional
            Implementation back-end for the transform. Supported back-ends:

            - ``'astra_cuda'``: ASTRA toolbox, using CUDA, 2D or 3D
            - ``'astra_cpu'``: ASTRA toolbox using CPU, only 2D
            - ``'skimage'``: scikit-image, only 2D parallel with square
              reconstruction space.

            For the default ``None``, the fastest available back-end is
            used.

        proj_space : `DiscretizedSpace`, optional
            Discretized projection (sinogram) space, the range of the forward
            operator or the domain of the adjoint (back-projection).
            Default: Inferred from parameters.
        use_cache : bool, optional
            If ``True``, data is cached. This gives a significant speed-up
            at the expense of a notable memory overhead, both on the GPU
            and on the CPU, since a full volume and a projection dataset
            are stored. That may be prohibitive in 3D.
            Default: True
        kwargs
            Further keyword arguments passed to the projector backend.

        Notes
        -----
        The ASTRA backend is faster if data are given with
        ``dtype='float32'`` and storage order 'C'. Otherwise copies will be
        needed.
        """
        if not isinstance(reco_space, DiscretizedSpace):
            raise TypeError(
                '`reco_space` must be a `DiscretizedSpace` instance, got '
                '{!r}'.format(reco_space))

        if not isinstance(geometry, Geometry):
            raise TypeError('`geometry` must be a `Geometry` instance, got '
                            '{!r}'.format(geometry))

        # Generate or check projection space
        proj_space = kwargs.pop('proj_space', None)
        if proj_space is None:
            dtype = reco_space.dtype

            if not reco_space.is_weighted:
                weighting = None
            elif (isinstance(reco_space.weighting, ConstWeighting)
                  and np.isclose(reco_space.weighting.const,
                                 reco_space.cell_volume)):
                # Approximate cell volume
                # TODO: find a way to treat angles and detector differently
                # regarding weighting. While the detector should be uniformly
                # discretized, the angles do not have to and often are not.
                # The needed partition property is available since
                # commit a551190d, but weighting is not adapted yet.
                # See also issue #286
                extent = float(geometry.partition.extent.prod())
                size = float(geometry.partition.size)
                weighting = extent / size
            else:
                raise NotImplementedError('unknown weighting of domain')

            proj_tspace = reco_space.tspace_type(geometry.partition.shape,
                                                 weighting=weighting,
                                                 dtype=dtype)

            if geometry.motion_partition.ndim == 0:
                angle_labels = []
            elif geometry.motion_partition.ndim == 1:
                angle_labels = ['$\\varphi$']
            elif geometry.motion_partition.ndim == 2:
                # TODO: check order
                angle_labels = ['$\\vartheta$', '$\\varphi$']
            elif geometry.motion_partition.ndim == 3:
                # TODO: check order
                angle_labels = ['$\\vartheta$', '$\\varphi$', '$\\psi$']
            else:
                angle_labels = None

            if geometry.det_partition.ndim == 1:
                det_labels = ['$s$']
            elif geometry.det_partition.ndim == 2:
                det_labels = ['$u$', '$v$']
            else:
                det_labels = None

            if angle_labels is None or det_labels is None:
                # Fallback for unknown configuration
                axis_labels = None
            else:
                axis_labels = angle_labels + det_labels

            proj_space = DiscretizedSpace(
                geometry.partition,
                proj_tspace,
                axis_labels=axis_labels
            )

        else:
            # proj_space was given, checking some stuff
            if not isinstance(proj_space, DiscretizedSpace):
                raise TypeError(
                    '`proj_space` must be a `DiscretizedSpace` instance, '
                    'got {!r}'.format(proj_space)
                )
            if proj_space.shape != geometry.partition.shape:
                raise ValueError(
                    '`proj_space.shape` not equal to `geometry.shape`: '
                    '{} != {}'
                    ''.format(proj_space.shape, geometry.partition.shape)
                )
            if proj_space.dtype != reco_space.dtype:
                raise ValueError(
                    '`proj_space.dtype` not equal to `reco_space.dtype`: '
                    '{} != {}'.format(proj_space.dtype, reco_space.dtype)
                )

        if reco_space.ndim != geometry.ndim:
            raise ValueError(
                '`reco_space.ndim` not equal to `geometry.ndim`: '
                '{} != {}'.format(reco_space.ndim, geometry.ndim)
            )

        # Cache for input/output arrays of transforms
        self.use_cache = kwargs.pop('use_cache', True)

        # Check `impl`
        impl = kwargs.pop('impl', None)
        impl_type, self.__cached_impl = self._check_impl(impl)
        self._impl_type = impl_type
        self.__impl = impl.lower() \
            if isinstance(impl, str) else impl_type.__name__

        self._geometry = geometry
        self.__vol_space = vol_space
        self.__proj_space = proj_space

        # Reserve name for cached properties (used for efficiency reasons)
        self._adjoint = None

        # Extra kwargs that can be reused for adjoint etc. These must
        # be retrieved with `get` instead of `pop` above.
        self._extra_kwargs = kwargs

        # Finally, initialize the Operator structure
        super(RayTransform, self).__init__(
            domain=reco_space, range=proj_space, linear=True
        )

    def _check_impl(self, impl):
        cached_impl = None

        if impl is None:  # User didn't specify a backend
            if not _AVAILABLE_IMPLS:
                raise RuntimeError('no ray transform back-end available; '
                                   'this requires 3rd party packages, please '
                                   'check the install docs')

            # Select fastest available, _AVAILABLE_IMPLS is sorted by speed
            impl_type = _SUPPORTED_IMPL[_AVAILABLE_IMPLS[0]]

        else:
            # User did specify `impl`
            if isinstance(impl, str):
                if impl.lower() not in _SUPPORTED_IMPL:
                    raise ValueError('`impl` {!r} not understood'.format(impl))

                if impl.lower() not in _AVAILABLE_IMPLS:
                    raise ValueError(
                        '{!r} back-end not available'.format(impl))

                impl_type = _SUPPORTED_IMPL[impl.lower()]
            elif isinstance(impl, type) or isinstance(impl, object):
                # User gave the type and leaves instantiation to us
                forward = getattr(impl, "call_forward", None)
                backward = getattr(impl, "call_backward", None)

                if not callable(forward) and not callable(backward):
                    raise TypeError('Type {!r} must be have a `call_forward` '
                                    'or `call_backward`.'.format(impl))

                if isinstance(impl, type):
                    impl_type = impl
                else:
                    # User gave an object for `impl`, meaning to set the
                    # backend cache to an already initiated object
                    impl_type = type(impl)
                    cached_impl = impl
            else:
                raise TypeError(
                    '`impl` {!r} should be a `str`, or an object or type '
                    'having a `call_forward()` and/or `call_backward()`. '
                    ''.format(type(impl))
                )

        return impl_type, cached_impl

    @property
    def impl(self):
        """Implementation name string or type."""

        return self.__impl

    def create_impl(self, use_cache=True):
        """Fetches or instantiates implementation backend for evaluation."""

        # Use impl creation (__cached_impl) when `use_cache` is True
        if not use_cache or self.__cached_impl is None:
            # Lazily (re)instantiate the backend
            self.__cached_impl = self._impl_type(
                self.geometry,
                reco_space=self.__reco_space.real_space,
                proj_space=self.__proj_space.real_space)

        return self.__cached_impl

    def _call(self, x, out, **kwargs):
        """Real-space forward projection for the current set-up."""

        return self.create_impl(self.use_cache) \
            .call_forward(x, out, **kwargs)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `RayBackProjection`
        """
        if self._adjoint is None:
            # bring `self` into scope to prevent shadowing in inline class
            ray_trafo = self

            class RayBackProjection(Operator):
                """Adjoint of the discrete Ray transform between L^p spaces."""

                def _call(self, x, out, **kwargs):
                    """Back-projection for the current set-up."""
                    return ray_trafo.create_impl(ray_trafo.use_cache) \
                        .call_backward(x, out, **kwargs)

                @property
                def geometry(self):
                    return ray_trafo.geometry

                @property
                def adjoint(self):
                    return ray_trafo

            kwargs = self._extra_kwargs.copy()
            kwargs['domain'] = self.range
            self._adjoint = RayBackProjection(range=self.domain, linear=True,
                                              **kwargs)

        return self._adjoint


if __name__ == '__main__':
    from odl.util.testutils import run_doctests

    run_doctests()
