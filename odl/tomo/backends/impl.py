# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ray transforms."""
from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABC
from odl.discr import DiscreteLp
from odl.tomo.geometry import Geometry

__all__ = ('RayTransformImplBase',)


class RayTransformImplBase(ABC):
    """Base for a RayTransform implementation (a backend)"""

    def __init__(self, geometry, reco_space, proj_space):
        """Initialize a new instance.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup.
        reco_space : `DiscreteLp`
            Reconstruction space, the space of the images to be forward
            projected.
        proj_space : `DiscreteLp`
            Projection space, the space of the result.
        """
        if not isinstance(geometry, Geometry):
            raise TypeError('`geometry` must be a `Geometry` instance, got '
                            '{!r}'.format(geometry))

        if not isinstance(reco_space, DiscreteLp):
            raise TypeError('`reco_space` must be a `DiscreteLP` instance, got '
                            '{!r}'.format(reco_space))

        if not isinstance(proj_space, DiscreteLp):
            raise TypeError('`proj_space` must be a `DiscreteLP` instance, got '
                            '{!r}'.format(proj_space))

        self.geometry = geometry
        self.reco_space = reco_space
        self.proj_space = proj_space

    @staticmethod
    def can_handle_size(size):
        """A very general way to check if the implementation is capable
        handling reconstruction volumes of a given size."""
        return True  # by default no assumptions are made

    @classmethod
    def supports_geometry(cls, geom):
        """Check if the implementation can handle this geometry."""
        return True

    @classmethod
    def supports_reco_space(cls, reco_name, reco_space):
        """Check if the implementation can handle the reconstruction space."""
        return True

    @abstractmethod
    def call_forward(self, x_real, out_real, **kwargs):
        raise NotImplementedError('Needs to be implemented by the subclass.')

    @abstractmethod
    def call_backward(self, x_real, out_real, **kwargs):
        raise NotImplementedError('Needs to be implemented by the subclass.')
