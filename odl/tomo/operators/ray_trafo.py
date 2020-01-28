# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ray transforms."""

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np

from odl.discr import DiscretizedSpace
from odl.operator import Operator
from odl.space.weighting import ConstWeighting
from odl.tomo.backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE, SKIMAGE_AVAILABLE,
    RayTransformImplBase,
    SkimageRayTransformImpl, AstraCudaRayTransformImpl,
    AstraCpuRayTransformImpl)
from odl.tomo.geometry import Geometry

# Backends that are implemented in ODL and can be specified via `impl`
_SUPPORTED_IMPL = {
    'astra_cpu': AstraCpuRayTransformImpl,
    'astra_cuda': AstraCudaRayTransformImpl,
    'skimage': SkimageRayTransformImpl}

# Backends that are installed, ordered by preference
_AVAILABLE_IMPLS = []
if ASTRA_CUDA_AVAILABLE:
    _AVAILABLE_IMPLS.append('astra_cuda')
if ASTRA_AVAILABLE:
    _AVAILABLE_IMPLS.append('astra_cpu')
if SKIMAGE_AVAILABLE:
    _AVAILABLE_IMPLS.append('skimage')

__all__ = ('RayTransform', 'RayBackProjection')


class RayTransformBase(Operator):
    """Base class for ray transforms containing common attributes."""

    def __init__(self, reco_space, geometry, variant, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        reco_space : `DiscretizedSpace`
            Discretized reconstruction space, the domain of the forward
            operator or the range of the adjoint (back-projection).
        geometry : `Geometry`
            Geometry of the transform that contains information about
            the data structure.
        variant : {'forward', 'backward'}
            Variant of the transform, i.e., whether the ray transform
            or its back-projection should be created.

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
        variant, variant_in = str(variant).lower(), variant
        if variant not in ('forward', 'backward'):
            raise ValueError('`variant` {!r} not understood'
                             ''.format(variant_in))

        if variant == 'forward':
            reco_name = 'domain'
            proj_name = 'range'
        else:
            reco_name = 'range'
            proj_name = 'domain'

        if not isinstance(reco_space, DiscretizedSpace):
            raise TypeError('`{}` must be a `DiscretizedSpace` instance, got '
                            '{!r}'.format(reco_name, reco_space))

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
                raise TypeError('`{}` must be a `DiscretizedSpace` instance, '
                                'got {!r}'.format(proj_name, proj_space))
            if proj_space.shape != geometry.partition.shape:
                raise ValueError('`{}.shape` not equal to `geometry.shape`: '
                                 '{} != {}'.format(proj_name, proj_space.shape,
                                                   geometry.partition.shape))
            if proj_space.dtype != reco_space.dtype:
                raise ValueError('`{}.dtype` not equal to `{}.dtype`: '
                                 '{} != {}'.format(proj_name, reco_name,
                                                   proj_space.dtype,
                                                   reco_space.dtype))

        if reco_space.ndim != geometry.ndim:
            raise ValueError('`{}.ndim` not equal to `geometry.ndim`: '
                             '{} != {}'.format(reco_name, reco_space.ndim,
                                               geometry.ndim))

        # Cache for input/output arrays of transforms
        self.use_cache = kwargs.pop('use_cache', True)

        impl = kwargs.pop('impl', None)
        self.__cached_impl = None

        if impl is None:  # User didn't specify a backend
            if not _AVAILABLE_IMPLS:
                raise RuntimeError('no ray transform back-end available; '
                                   'this requires 3rd party packages, please '
                                   'check the install docs')

            # Select fastest available, _AVAILABLE_IMPLS is sorted by speed
            impl_type = _SUPPORTED_IMPL[_AVAILABLE_IMPLS[0]]

            # Warn if implementation is not fast enough
            if not impl_type.can_handle_size(reco_space.size):
                if impl_type == AstraCpuRayTransformImpl:
                    warnings.warn(
                        "The best available backend ('astra_cpu') may be too "
                        "slow for volumes of this size. Consider using "
                        "'astra_cuda' if your machine has an Nvidia GPU. "
                        "This warning can be disabled by explicitly setting "
                        "`impl='astra_cpu'`.",
                        RuntimeWarning)
                elif impl_type == SkimageRayTransformImpl:
                    warnings.warn(
                        "The best available backend ('skimage') may be too "
                        "slow for volumes of this size. Consider using ASTRA. "
                        "This warning can be disabled by explicitly setting "
                        "`impl='skimage'`.",
                        RuntimeWarning)
                else:
                    warnings.warn(
                        "The `impl` backend indicates that it might be too "
                        "slow for volumes of the input size.",
                        RuntimeWarning)

        else:
            # User did specify `impl`
            if isinstance(impl, str):
                if impl.lower() not in _SUPPORTED_IMPL:
                    raise ValueError('`impl` {!r} not understood'.format(impl))

                if impl.lower() not in _AVAILABLE_IMPLS:
                    raise ValueError(
                        '{!r} back-end not available'.format(impl))

                impl_type = _SUPPORTED_IMPL[impl.lower()]
            elif isinstance(impl, type):
                # User gave the type and leaves instantiation to us
                if not issubclass(impl, RayTransformImplBase):
                    raise TypeError('Type {!r} must be a subclass of '
                                    '`RayTransformImplBase`.'.format(impl))

                impl_type = impl
            elif isinstance(impl, RayTransformImplBase):
                # User gave an object for `impl`, meaning to set the
                # backend cache to an already initiated object
                impl_type = type(impl)
                self.__cached_impl = impl
            else:
                raise TypeError(
                    'Given `impl` should be a `str`, or the type or subclass '
                    'of `RayTransformImplBase`, '
                    'now it is a {!r}.'.format(type(impl)))

        # Sanity checks
        geometry_support = impl_type.supports_geometry(geometry)
        if not geometry_support:
            raise geometry_support

        reco_space_support = impl_type.supports_reco_space(reco_space,
                                                           reco_name)
        if not reco_space_support:
            raise reco_space_support

        self.__geometry = geometry
        self._impl_type = impl_type
        self.__impl = impl.lower() if isinstance(impl, str) else impl_type
        self.__reco_space = reco_space
        self.__proj_space = proj_space

        # Reserve name for cached properties (used for efficiency reasons)
        self._adjoint = None

        # Extra kwargs that can be reused for adjoint etc. These must
        # be retrieved with `get` instead of `pop` above.
        self._extra_kwargs = kwargs

        # Finally, initialize the Operator structure
        if variant == 'forward':
            super(RayTransformBase, self).__init__(
                domain=reco_space, range=proj_space, linear=True)
        elif variant == 'backward':
            super(RayTransformBase, self).__init__(
                domain=proj_space, range=reco_space, linear=True)

    @property
    def impl(self):
        """Implementation name string or type."""

        return self.__impl

    def create_impl(self, from_cache=True):
        """Fetches or instantiates implementation backend for evaluation."""

        # Skip impl creation (__cached_impl) when `clear_cache` is True
        if not from_cache or self.__cached_impl is None:
            # Lazily (re)instantiate the backend
            self.__cached_impl = self._impl_type(
                self.geometry,
                reco_space=self.__reco_space.real_space,
                proj_space=self.__proj_space.real_space)

        return self.__cached_impl

    @property
    def geometry(self):
        """Geometry of this operator."""
        return self.__geometry

    def _call(self, x, out=None):
        """Return ``self(x[, out])``."""
        if self.domain.is_real:
            return self._call_real(x, out, **self._extra_kwargs)

        elif self.domain.is_complex:
            result_parts = [
                self._call_real(
                    x.real, getattr(out, 'real', None), **self._extra_kwargs
                ),
                self._call_real(
                    x.imag, getattr(out, 'imag', None), **self._extra_kwargs
                ),
            ]

            if out is None:
                out = self.range.element()
                out.real = result_parts[0]
                out.imag = result_parts[1]

            return out

        else:
            raise RuntimeError('bad domain {!r}'.format(self.domain))


class RayTransform(RayTransformBase):
    """Discrete Ray transform between L^p spaces."""

    def __init__(self, domain, geometry, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscretizedSpace`
            Discretized reconstruction space, the domain of the forward
            projector.
        geometry : `Geometry`
            Geometry of the transform, containing information about
            the operator range (projection/sinogram space).

        Other Parameters
        ----------------
        impl : {`None`, 'astra_cuda', 'astra_cpu', 'skimage'}, optional
            Implementation back-end for the transform. Supported back-ends:

            - ``'astra_cuda'``: ASTRA toolbox, using CUDA, 2D or 3D
            - ``'astra_cpu'``: ASTRA toolbox using CPU, only 2D
            - ``'skimage'``: scikit-image, only 2D parallel with square
              reconstruction space.

            For the default ``None``, the fastest available back-end is
            used, tried in the above order.
        range : `DiscretizedSpace`, optional
            Discretized projection (sinogram) space, the range of the
            forward projector.
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

        See Also
        --------
        astra_cpu_forward_projector
        AstraCudaProjectorImpl
        skimage_radon_forward_projector
        """
        range = kwargs.pop('range', None)
        super(RayTransform, self).__init__(
            reco_space=domain, proj_space=range, geometry=geometry,
            variant='forward', **kwargs)

    def _call_real(self, x_real, out_real, **kwargs):
        """Real-space forward projection for the current set-up."""

        return self.create_impl(self.use_cache) \
            .call_forward(x_real, out_real, **kwargs)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `RayBackProjection`
        """
        if self._adjoint is None:
            kwargs = self._extra_kwargs.copy()
            kwargs['domain'] = self.range

            # initiate adjoint with cached implementation if `use_cache`
            self._adjoint = RayBackProjection(
                self.domain, self.geometry,
                impl=(self.impl
                      if not self.use_cache
                      else self.create_impl(self.use_cache)),
                use_cache=self.use_cache,
                **kwargs)

        return self._adjoint


class RayBackProjection(RayTransformBase):
    """Adjoint of the discrete Ray transform between L^p spaces."""

    def __init__(self, range, geometry, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        range : `DiscretizedSpace`
            Discretized reconstruction space, the range of the
            backprojection operator.
        geometry : `Geometry`
            Geometry of the transform, containing information about
            the operator domain (projection/sinogram space).

        Other Parameters
        ----------------
        impl : {`None`, 'astra_cuda', 'astra_cpu', 'skimage'}, optional
            Implementation back-end for the transform. Supported back-ends:

            - ``'astra_cuda'``: ASTRA toolbox, using CUDA, 2D or 3D
            - ``'astra_cpu'``: ASTRA toolbox using CPU, only 2D
            - ``'skimage'``: scikit-image, only 2D parallel with square
              reconstruction space.

            For the default ``None``, the fastest available back-end is
            used, tried in the above order.

        domain : `DiscretizedSpace`, optional
            Discretized projection (sinogram) space, the domain of the
            backprojection operator.
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

        See Also
        --------
        astra_cpu_back_projector
        AstraCudaBackProjectorImpl
        skimage_radon_back_projector
        """
        domain = kwargs.pop('domain', None)
        super(RayBackProjection, self).__init__(
            reco_space=range, proj_space=domain, geometry=geometry,
            variant='backward', **kwargs)

    def _call_real(self, x_real, out_real, **kwargs):
        """Real-space back-projection for the current set-up."""

        return self.create_impl(self.use_cache) \
            .call_backward(x_real, out_real, **kwargs)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `RayTransform`
        """
        if self._adjoint is None:
            kwargs = self._extra_kwargs.copy()
            kwargs['range'] = self.domain

            # initiate adjoint with cached implementation if `use_cache`
            self._adjoint = RayTransform(
                self.range, self.geometry,
                impl=(self.impl
                      if not self.use_cache
                      else self.create_impl(self.use_cache)),
                use_cache=self.use_cache,
                **kwargs)

            return self._adjoint


if __name__ == '__main__':
    from odl.util.testutils import run_doctests

    run_doctests()
