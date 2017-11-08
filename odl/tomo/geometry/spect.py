# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Single-photon emission computed tomography (SPECT) geometry."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.tomo.geometry.parallel import Parallel3dAxisGeometry
from odl.tomo.util.utility import transform_system
from odl.util import signature_string, indent, array_str

__all__ = ('ParallelHoleCollimatorGeometry', )


class ParallelHoleCollimatorGeometry(Parallel3dAxisGeometry):

    """Geometry for SPECT Parallel hole collimator.

    For details, check `the online docs
    <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_.
    """

    def __init__(self, apart, dpart, det_radius, axis=(0, 0, 1), **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval.
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter rectangle.
        det_radius : positive float
            Radius of the circular detector orbit.
        axis : `array-like`, shape ``(3,)``, optional
            Vector defining the fixed rotation axis of this geometry.

        Other Parameters
        ----------------
        orig_to_det_init : `array-like`, shape ``(3,)``, optional
            Vector pointing towards the initial position of the detector
            reference point. The default depends on ``axis``, see Notes.
            The zero vector is not allowed.
        det_axes_init : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation. The default
            depends on ``axis``, see Notes.
        translation : `array-like`, shape ``(3,)``, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`,
            and also shifts the axis of rotation.
            Default: ``(0, 0, 0)``
        check_bounds : bool, optional
            If ``True``, methods perform sanity checks on provided input
            parameters.
            Default: ``True``

        Notes
        -----
        In the default configuration, the rotation axis is ``(0, 0, 1)``,
        the vector towards the initial detector reference point is
        ``(0, 1, 0)``, and the default detector axes are
        ``[(1, 0, 0), (0, 0, 1)]``.
        If a different ``axis`` is provided, the new default initial
        position and the new default axes are the computed by rotating
        the original ones by a matrix that transforms ``(0, 0, 1)`` to the
        new (normalized) ``axis``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 0, 1), axis)
            orig_to_det_init = init_rot.dot((0, 1, 0))
            det_axes_init[0] = init_rot.dot((1, 0, 0))
            det_axes_init[1] = init_rot.dot((0, 0, 1))
        """
        self.__det_radius = float(det_radius)
        if self.det_radius <= 0:
            raise ValueError('`det_radius` must be positive, got {}'
                             ''.format(det_radius))

        orig_to_det_init = kwargs.pop('orig_to_det_init', None)

        if orig_to_det_init is not None:
            orig_to_det_init = np.asarray(orig_to_det_init, dtype=float)
            orig_to_det_norm = np.linalg.norm(orig_to_det_init)
            if orig_to_det_norm == 0:
                raise ValueError('`orig_to_det_init` cannot be zero')
            else:
                det_pos_init = (orig_to_det_init / orig_to_det_norm *
                                self.det_radius)
            kwargs['det_pos_init'] = det_pos_init
        self._orig_to_det_init_arg = orig_to_det_init

        super(ParallelHoleCollimatorGeometry, self).__init__(
            apart, dpart, axis, **kwargs)

    @classmethod
    def frommatrix(cls, apart, dpart, det_radius, init_matrix, **kwargs):
        """Create a `ParallelHoleCollimatorGeometry` using a matrix.

        This alternative constructor uses a matrix to rotate and
        translate the default configuration. It is most useful when
        the transformation to be applied is already given as a matrix.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the parameter interval.
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter set.
        det_radius : positive float
            Radius of the circular detector orbit.
        init_matrix : `array_like`, shape ``(3, 3)`` or ``(3, 4)``, optional
            Transformation matrix whose left ``(3, 3)`` block is multiplied
            with the default ``det_pos_init`` and ``det_axes_init`` to
            determine the new vectors. If present, the fourth column acts
            as a translation after the initial transformation.
            The resulting ``det_axes_init`` will be normalized.
        kwargs :
            Further keyword arguments passed to the class constructor.

        Returns
        -------
        geometry : `ParallelHoleCollimatorGeometry`
            The resulting geometry.
        """
        # Get transformation and translation parts from `init_matrix`
        init_matrix = np.asarray(init_matrix, dtype=float)
        if init_matrix.shape not in ((3, 3), (3, 4)):
            raise ValueError('`matrix` must have shape (3, 3) or (3, 4), '
                             'got array with shape {}'
                             ''.format(init_matrix.shape))
        trafo_matrix = init_matrix[:, :3]
        translation = init_matrix[:, 3:].squeeze()

        # Transform the default vectors
        default_axis = cls._default_config['axis']
        # Normalized version, just in case
        default_orig_to_det_init = (
            np.array(cls._default_config['det_pos_init'], dtype=float) /
            np.linalg.norm(cls._default_config['det_pos_init']))
        default_det_axes_init = cls._default_config['det_axes_init']
        vecs_to_transform = ((default_orig_to_det_init,) +
                             default_det_axes_init)
        transformed_vecs = transform_system(
            default_axis, None, vecs_to_transform, matrix=trafo_matrix)

        # Use the standard constructor with these vectors
        axis, orig_to_det, det_axis_0, det_axis_1 = transformed_vecs
        if translation.size != 0:
            kwargs['translation'] = translation

        return cls(apart, dpart, det_radius, axis,
                   orig_to_det_init=orig_to_det,
                   det_axes_init=[det_axis_0, det_axis_1],
                   **kwargs)

    @property
    def det_radius(self):
        """Radius of the detector orbit."""
        return self.__det_radius

    @property
    def orig_to_det_init(self):
        """Unit vector from rotation center to initial detector position."""
        return self.det_pos_init / np.linalg.norm(self.det_pos_init)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = [('det_radius', self.det_radius, -1)]

        if not np.allclose(self.axis, self._default_config['axis']):
            optargs.append(['axis', array_str(self.axis), ''])

        if self._orig_to_det_init_arg is not None:
            optargs.append(['orig_to_det_init',
                            array_str(self._orig_to_det_init_arg),
                            ''])

        if self._det_axes_init_arg is not None:
            optargs.append(
                ['det_axes_init',
                 tuple(array_str(a) for a in self._det_axes_init_arg),
                 None])

        if not np.array_equal(self.translation, (0, 0, 0)):
            optargs.append(['translation', array_str(self.translation), ''])

        sig_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(sig_str))
