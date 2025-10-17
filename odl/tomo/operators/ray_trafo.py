# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ray transforms."""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np

from odl.core.discr import DiscretizedSpace
from odl.operator import Operator
from odl.core.space.weightings.weighting import ConstWeighting
from odl.tomo.backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE, SKIMAGE_AVAILABLE)
from odl.tomo.backends.astra_cpu import AstraCpuImpl
from odl.tomo.backends.astra_cuda import AstraCudaImpl
from odl.tomo.backends.skimage_radon import SkImageImpl
from odl.tomo.geometry import Geometry
from odl.core.util import is_string

# RAY_TRAFO_IMPLS are used by `RayTransform` when no `impl` is given.
# The last inserted implementation has highest priority.
RAY_TRAFO_IMPLS = OrderedDict()
if SKIMAGE_AVAILABLE:
    RAY_TRAFO_IMPLS['skimage'] = SkImageImpl
if ASTRA_AVAILABLE:
    RAY_TRAFO_IMPLS['astra_cpu'] = AstraCpuImpl
if ASTRA_CUDA_AVAILABLE:
    RAY_TRAFO_IMPLS['astra_cuda'] = AstraCudaImpl

__all__ = ('RayTransform',)


class RayTransform(Operator):
    """Linear X-Ray (Radon) transform operator between L^p spaces."""

    def __init__(self, vol_space, geometry, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        vol_space : `DiscretizedSpace`
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
        if not isinstance(vol_space, DiscretizedSpace):
            raise TypeError(
                '`vol_space` must be a `DiscretizedSpace` instance, got '
                '{!r}'.format(vol_space))

        if not isinstance(geometry, Geometry):
            raise TypeError(
                '`geometry` must be a `Geometry` instance, got {!r}'
                ''.format(geometry)
            )

        # Generate or check projection space
        proj_space = kwargs.pop('proj_space', None)
        if proj_space is None:
            dtype = vol_space.dtype

            if not vol_space.is_weighted:
                weighting = None
            elif (
                isinstance(vol_space.weighting, ConstWeighting)
                and np.isclose(
                    vol_space.weighting.const, vol_space.cell_volume
                )
            ):
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

            proj_tspace = vol_space.tspace_type(
                geometry.partition.shape,
                weighting=weighting,
                dtype=dtype,
                device=vol_space.device
            )

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
            if proj_space.dtype != vol_space.dtype:
                raise ValueError(
                    '`proj_space.dtype` not equal to `vol_space.dtype`: '
                    '{} != {}'.format(proj_space.dtype, vol_space.dtype)
                )

        if vol_space.ndim != geometry.ndim:
            raise ValueError(
                '`vol_space.ndim` not equal to `geometry.ndim`: '
                '{} != {}'.format(vol_space.ndim, geometry.ndim)
            )

        # Cache for input/output arrays of transforms
        self.use_cache = kwargs.pop('use_cache', True)

        # Check `impl`
        impl = kwargs.pop('impl', None)
        impl_type, self.__cached_impl = self._initialize_impl(impl)
        self._impl_type = impl_type
        if is_string(impl):
            self.__impl = impl.lower()
        else:
            self.__impl = impl_type.__name__

        self._geometry = geometry

        # Reserve name for cached properties (used for efficiency reasons)
        self._adjoint = None

        # Extra kwargs that can be reused for adjoint etc. These must
        # be retrieved with `get` instead of `pop` above.
        self._extra_kwargs = kwargs

        # Finally, initialize the Operator structure
        super(RayTransform, self).__init__(
            domain=vol_space, range=proj_space, linear=True
        )

    @staticmethod
    def _initialize_impl(impl):
        """Internal method to verify the validity of the `impl` kwarg."""
        impl_instance = None

        if impl is None:  # User didn't specify a backend
            if not RAY_TRAFO_IMPLS:
                raise RuntimeError(
                    'No `RayTransform` back-end available; this requires '
                    '3rd party packages, please check the install docs.'
                )

            # Select fastest available
            impl_type = next(reversed(RAY_TRAFO_IMPLS.values()))

        else:
            # User did specify `impl`
            if is_string(impl):
                if impl.lower() not in RAY_TRAFO_IMPLS.keys():
                    raise ValueError(
                        'The {!r} `impl` is not found. This `impl` is either '
                        'not supported, it may be misspelled, or external '
                        'packages required are not available. Consult '
                        '`RAY_TRAFO_IMPLS` to find the run-time available '
                        'implementations.'.format(impl)
                    )

                impl_type = RAY_TRAFO_IMPLS[impl.lower()]
            elif isinstance(impl, type) or isinstance(impl, object):
                # User gave the type and leaves instantiation to us
                forward = getattr(impl, "call_forward", None)
                backward = getattr(impl, "call_backward", None)

                if not callable(forward) and not callable(backward):
                    raise TypeError(
                        'Type {!r} must have a `call_forward()` '
                        'and/or `call_backward()`.'.format(impl)
                    )

                if isinstance(impl, type):
                    impl_type = impl
                else:
                    # User gave an object for `impl`, meaning to set the
                    # backend cache to an already initiated object
                    impl_type = type(impl)
                    impl_instance = impl
            else:
                raise TypeError(
                    '`impl` {!r} should be a string, or an object or type '
                    'having a `call_forward()` and/or `call_backward()`. '
                    ''.format(type(impl))
                )

        return impl_type, impl_instance

    @property
    def impl(self):
        """Implementation name string.

        If a custom ``impl`` was provided this method returns a ``str``
        of the type."""
        return self.__impl

    def get_impl(self, use_cache=True):
        """Fetches or instantiates implementation backend for evaluation.

        Parameters
        ----------
        bool : use_cache
            If ``True`` returns the cached implementation backend, if it
            was generated in a previous call (or given with ``__init__``).
            If ``False`` a new instance of the backend will be generated,
            freeing up GPU memory and RAM used by the backend.
        """

        # Use impl creation (__cached_impl) when `use_cache` is True
        if not use_cache or self.__cached_impl is None:
            # Lazily (re)instantiate the backend
            self.__cached_impl = self._impl_type(
                self.geometry,
                vol_space=self.domain,
                proj_space=self.range)

        return self.__cached_impl

    def _call(self, x, out=None, **kwargs):
        """Forward projection.

        Parameters
        ----------
        x : DiscretizedSpaceElement
            A volume. Must be an element of `RayTransform.domain`.
        out : `RayTransform.range` element, optional
            Element to which the result of the operator evaluation is written.
        **kwargs
            Extra keyword arguments, passed on to the implementation
            backend.

        Returns
        -------
        DiscretizedSpaceElement
            Result of the transform, an element of the range.
        """
        return self.get_impl(self.use_cache).call_forward(x, out, **kwargs)

    @property
    def geometry(self):
        return self._geometry

    @property
    def adjoint(self):
        """Adjoint of this operator.

        The adjoint of the `RayTransform` is the linear `RayBackProjection`
        operator. It uses the same geometry and shares the implementation
        backend whenever `RayTransform.use_cache` is `True`.

        Returns
        -------
        adjoint : `RayBackProjection`
        """
        if self._adjoint is None:
            # bring `self` into scope to prevent shadowing in inline class
            ray_trafo = self

            class RayBackProjection(Operator):
                """Adjoint of the discrete Ray transform between L^p spaces."""

                def _call(self, x, out=None, **kwargs):
                    """Backprojection.

                    Parameters
                    ----------
                    x : DiscretizedSpaceElement
                        A sinogram. Must be an element of
                        `RayTransform.range` (domain of `RayBackProjection`).
                    out : `RayBackProjection.domain` element, optional
                        A volume to which the result of this evaluation is
                        written.
                    **kwargs
                        Extra keyword arguments, passed on to the
                        implementation backend.

                    Returns
                    -------
                    DiscretizedSpaceElement
                        Result of the transform in the domain
                        of `RayProjection`.
                    """
                    return ray_trafo.get_impl(
                        ray_trafo.use_cache
                    ).call_backward(x, out, **kwargs)

                @property
                def geometry(self):
                    return ray_trafo.geometry

                @property
                def adjoint(self):
                    return ray_trafo

            kwargs = self._extra_kwargs.copy()
            kwargs['domain'] = self.range
            self._adjoint = RayBackProjection(
                range=self.domain, linear=True, **kwargs
            )

        return self._adjoint


if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests

    run_doctests()
