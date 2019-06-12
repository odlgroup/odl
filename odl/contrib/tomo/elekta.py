# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tomography helpers for Elekta systems."""

import numpy as np
import odl


__all__ = ('elekta_icon_geometry',
           'elekta_icon_space',
           'elekta_icon_fbp',
           'elekta_xvi_geometry',
           'elekta_xvi_space',
           'elekta_xvi_fbp')


def elekta_icon_geometry(sad=780.0, sdd=1000.0,
                         piercing_point=(390.0, 0.0),
                         angles=None, num_angles=None,
                         detector_shape=(780, 720)):
    """Tomographic geometry of the Elekta Icon CBCT system.

    See the [whitepaper]_ for specific descriptions of each parameter.

    All measurments are given in millimeters unless otherwise stated.

    Parameters
    ----------
    sad : float, optional
        Source to Axis distance.
    sdd : float, optional
        Source to Detector distance.
    piercing_point : sequence of foat, optional
        Position in the detector (in pixel coordinates) that a beam from the
        source, passing through the axis of rotation perpendicularly, hits.
    angles : array-like, optional
        List of angles given in radians that the projection images were taken
        at. Exclusive with num_angles.
        Default: np.linspace(1.2, 5.0, 332)
    num_angles : int, optional
        Number of angles. Exclusive with angles.
        Default: 332
    detector_shape : sequence of int, optional
        Shape of the detector (in pixels). Useful if a sub-sampled system
        should be studied.

    Returns
    -------
    elekta_icon_geometry : `ConeBeamGeometry`

    Examples
    --------
    Create default geometry:

    >>> from odl.contrib import tomo
    >>> geometry = tomo.elekta_icon_geometry()

    Use a smaller detector (improves efficiency):

    >>> small_geometry = tomo.elekta_icon_geometry(detector_shape=[100, 100])

    See Also
    --------
    elekta_icon_space : Default reconstruction space for the Elekta Icon CBCT.
    elekta_icon_fbp: Default reconstruction method for the Elekta Icon CBCT.

    References
    ----------
    .. [whitepaper] *Design and performance characteristics of a Cone Beam
       CT system for Leksell Gamma Knife Icon*
    """
    sad = float(sad)
    assert sad > 0
    sdd = float(sdd)
    assert sdd > sad
    piercing_point = np.array(piercing_point, dtype=float)
    assert piercing_point.shape == (2,)

    if angles is not None and num_angles is not None:
        raise ValueError('cannot provide both `angles` and `num_angles`')
    elif angles is not None:
        angles = odl.nonuniform_partition(angles)
        assert angles.ndim == 1
    elif num_angles is not None:
        angles = odl.uniform_partition(1.2, 5.0, num_angles)
    else:
        angles = odl.uniform_partition(1.2, 5.0, 332)

    detector_shape = np.array(detector_shape, dtype=int)

    # Constant system parameters
    pixel_size = 0.368
    det_extent_mm = np.array([287.04, 264.96])

    # Compute the detector partition
    piercing_point_mm = pixel_size * piercing_point
    det_min_pt = -piercing_point_mm
    det_max_pt = det_min_pt + det_extent_mm
    detector_partition = odl.uniform_partition(min_pt=det_min_pt,
                                               max_pt=det_max_pt,
                                               shape=detector_shape)

    # Create the geometry
    geometry = odl.tomo.ConeBeamGeometry(
        angles, detector_partition,
        src_radius=sad, det_radius=sdd - sad)

    return geometry


def elekta_icon_space(shape=(448, 448, 448), **kwargs):
    """Default reconstruction space for the Elekta Icon CBCT.

    See the [whitepaper]_ for further information.

    Parameters
    ----------
    shape : sequence of int, optional
        Shape of the space, in voxels.
    kwargs :
        Keyword arguments to pass to `uniform_discr` to modify the space, e.g.
        use another backend. By default, the dtype is set to float32.

    Returns
    -------
    elekta_icon_space : `DiscreteLp`

    Examples
    --------
    Create default space:

    >>> from odl.contrib import tomo
    >>> space = tomo.elekta_icon_space()

    Create sub-sampled space:

    >>> space = tomo.elekta_icon_space(shape=(100, 100, 100))

    See Also
    --------
    elekta_icon_geometry: Geometry for the Elekta Icon CBCT.
    elekta_icon_fbp: Default reconstruction method for the Elekta Icon CBCT.

    References
    ----------
    .. [whitepaper] *Design and performance characteristics of a Cone Beam
       CT system for Leksell Gamma Knife Icon*
    """
    if 'dtype' not in kwargs:
        kwargs['dtype'] = 'float32'
    return odl.uniform_discr(min_pt=[-112.0, -112.0, 0.0],
                             max_pt=[112.0, 112.0, 224.0],
                             shape=shape,
                             **kwargs)


def elekta_icon_fbp(ray_transform,
                    padding=False, filter_type='Hann', frequency_scaling=0.6,
                    parker_weighting=True):
    """Approximation of the FDK reconstruction used in the Elekta Icon.

    Parameters
    ----------
    ray_transform : `RayTransform`
        The ray transform to be used, should have an Elekta Icon geometry.
    padding : bool, optional
        Whether the FBP filter should use padding, increases memory use
        significantly.
    filter_type : str, optional
        Type of filter to apply in the FBP filter.
    frequency_scaling : float, optional
        Frequency scaling for FBP filter.
    parker_weighting : bool, optional
        Whether Parker weighting should be applied to compensate for partial
        scan.

    Returns
    -------
    elekta_icon_fbp : `DiscreteLp`

    Examples
    --------
    Create default FBP for default geometry:

    >>> from odl.contrib import tomo
    >>> geometry = tomo.elekta_icon_geometry()
    >>> space = tomo.elekta_icon_space()
    >>> ray_transform = odl.tomo.RayTransform(space, geometry)
    >>> fbp_op = tomo.elekta_icon_fbp(ray_transform)
    """
    fbp_op = odl.tomo.fbp_op(ray_transform,
                             padding=padding,
                             filter_type=filter_type,
                             frequency_scaling=frequency_scaling)
    if parker_weighting:
        parker_weighting = odl.tomo.parker_weighting(ray_transform)
        fbp_op = fbp_op * parker_weighting

    return fbp_op


def elekta_xvi_geometry(sad=1000.0, sdd=1500.0,
                        piercing_point=(512.0, 512.0),
                        angles=None, num_angles=None,
                        detector_shape=(1024, 1024)):
    """Tomographic geometry of the Elekta XVI system.

    All measurments are given in millimeters unless otherwise stated.

    Parameters
    ----------
    sad : float, optional
        Source to Axis distance.
    sdd : float, optional
        Source to Detector distance.
    piercing_point : sequence of foat, optional
        Position in the detector (in pixel coordinates) that a beam from the
        source, passing through the axis of rotation perpendicularly, hits.
    angles : array-like, optional
        List of angles given in radians that the projection images were taken
        at. Exclusive with num_angles.
        Default: np.linspace(0, 2 * np.pi, 650, endpoint=False)
    num_angles : int, optional
        Number of angles. Exclusive with angles.
        Default: 332
    detector_shape : sequence of int, optional
        Shape of the detector (in pixels). Useful if a sub-sampled system
        should be studied.

    Returns
    -------
    elekta_xvi_geometry : `ConeBeamGeometry`

    Examples
    --------
    Create default geometry:

    >>> from odl.contrib import tomo
    >>> geometry = tomo.elekta_xvi_geometry()

    Use a smaller detector (improves efficiency):

    >>> small_geometry = tomo.elekta_xvi_geometry(detector_shape=[100, 100])

    See Also
    --------
    elekta_xvi_space : Default reconstruction space for the Elekta XVI CBCT.
    elekta_xvi_fbp: Default reconstruction method for the Elekta XVI CBCT.
    """
    sad = float(sad)
    assert sad > 0
    sdd = float(sdd)
    assert sdd > sad
    piercing_point = np.array(piercing_point, dtype=float)
    assert piercing_point.shape == (2,)

    if angles is not None and num_angles is not None:
        raise ValueError('cannot provide both `angles` and `num_angles`')
    elif angles is not None:
        angles = odl.nonuniform_partition(angles)
        assert angles.ndim == 1
    elif num_angles is not None:
        angles = odl.uniform_partition(0, 2 * np.pi, num_angles)
    else:
        angles = odl.uniform_partition(0, 2 * np.pi, 650)

    detector_shape = np.array(detector_shape, dtype=int)

    # Constant system parameters
    pixel_size = 0.4
    det_extent_mm = np.array([409.6, 409.6])

    # Compute the detector partition
    piercing_point_mm = pixel_size * piercing_point
    det_min_pt = -piercing_point_mm
    det_max_pt = det_min_pt + det_extent_mm
    detector_partition = odl.uniform_partition(min_pt=det_min_pt,
                                               max_pt=det_max_pt,
                                               shape=detector_shape)

    # Create the geometry
    geometry = odl.tomo.ConeBeamGeometry(
        angles, detector_partition,
        src_radius=sad, det_radius=sdd - sad)

    return geometry


def elekta_xvi_space(shape=(512, 512, 512), **kwargs):
    """Default reconstruction space for the Elekta XVI CBCT.

    Parameters
    ----------
    shape : sequence of int, optional
        Shape of the space, in voxels.
    kwargs :
        Keyword arguments to pass to `uniform_discr` to modify the space, e.g.
        use another backend. By default, the dtype is set to float32.

    Returns
    -------
    elekta_xvi_space : `DiscreteLp`

    Examples
    --------
    Create default space:

    >>> from odl.contrib import tomo
    >>> space = tomo.elekta_xvi_space()

    Create sub-sampled space:

    >>> space = tomo.elekta_xvi_space(shape=(100, 100, 100))

    See Also
    --------
    elekta_xvi_geometry: Geometry for the Elekta XVI CBCT.
    elekta_xvi_fbp: Default reconstruction method for the Elekta XVI CBCT.
    """
    if 'dtype' not in kwargs:
        kwargs['dtype'] = 'float32'
    return odl.uniform_discr(min_pt=[-128.0, -128, -128.0],
                             max_pt=[128.0, 128.0, 128.0],
                             shape=shape,
                             **kwargs)


def elekta_xvi_fbp(ray_transform,
                   padding=False, filter_type='Hann', frequency_scaling=0.6):
    """Approximation of the FDK reconstruction used in the Elekta XVI.

    Parameters
    ----------
    ray_transform : `RayTransform`
        The ray transform to be used, should have an Elekta XVI geometry.
    padding : bool, optional
        Whether the FBP filter should use padding, increases memory use
        significantly.
    filter_type : str, optional
        Type of filter to apply in the FBP filter.
    frequency_scaling : float, optional
        Frequency scaling for FBP filter.

    Returns
    -------
    elekta_xvi_fbp : `DiscreteLp`

    Examples
    --------
    Create default FBP for default geometry:

    >>> from odl.contrib import tomo
    >>> geometry = tomo.elekta_xvi_geometry()
    >>> space = tomo.elekta_xvi_space()
    >>> ray_transform = odl.tomo.RayTransform(space, geometry)
    >>> fbp_op = tomo.elekta_xvi_fbp(ray_transform)
    """
    fbp_op = odl.tomo.fbp_op(ray_transform,
                             padding=padding,
                             filter_type=filter_type,
                             frequency_scaling=frequency_scaling)

    return fbp_op


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
