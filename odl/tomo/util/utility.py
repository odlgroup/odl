# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function, division, absolute_import
import numpy as np

__all__ = ('euler_matrix', 'axis_rotation', 'axis_rotation_matrix',
           'rotation_matrix_from_to', 'transform_system',
           'perpendicular_vector', 'is_inside_bounds')


def euler_matrix(phi, theta=None, psi=None):
    """Rotation matrix in 2 and 3 dimensions.

    Its rows represent the canonical unit vectors as seen from the
    rotated system while the columns are the rotated unit vectors as
    seen from the canonical system.

    Parameters
    ----------
    phi : float or `array-like`
        Either 2D counter-clockwise rotation angle (in radians) or first
        Euler angle.
    theta, psi : float or `array-like`, optional
        Second and third Euler angles in radians. If both are ``None``, a
        2D rotation matrix is computed. Otherwise a 3D rotation is computed,
        where the default ``None`` is equivalent to ``0.0``.
        The rotation is performed in "ZXZ" rotation order, see the
        Wikipedia article `Euler angles`_.

    Returns
    -------
    mat : `numpy.ndarray`
        Rotation matrix corresponding to the given angles. The
        returned array has shape ``(ndim, ndim)`` if all angles represent
        single parameters, with ``ndim == 2`` for ``phi`` only and
        ``ndim == 3`` for 2 or 3 Euler angles.
        If any of the angle parameters is an array, the shape of the
        returned array is ``broadcast(phi, theta, psi).shape + (ndim, ndim)``.

    References
    ----------
    .. _Euler angles:
        https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    """
    if theta is None and psi is None:
        squeeze_out = (np.shape(phi) == ())
        ndim = 2
        phi = np.array(phi, dtype=float, copy=False, ndmin=1)
        theta = psi = 0.0
    else:
        # `None` broadcasts like a scalar
        squeeze_out = (np.broadcast(phi, theta, psi).shape == ())
        ndim = 3
        phi = np.array(phi, dtype=float, copy=False, ndmin=1)
        if theta is None:
            theta = 0.0
        if psi is None:
            psi = 0.0
        theta = np.array(theta, dtype=float, copy=False, ndmin=1)
        psi = np.array(psi, dtype=float, copy=False, ndmin=1)
        ndim = 3

    cph = np.cos(phi)
    sph = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cps = np.cos(psi)
    sps = np.sin(psi)

    if ndim == 2:
        mat = np.array([[cph, -sph],
                        [sph, cph]])
    else:
        mat = np.array([
            [cph * cps - sph * cth * sps,
             -cph * sps - sph * cth * cps,
             sph * sth],
            [sph * cps + cph * cth * sps,
             -sph * sps + cph * cth * cps,
             -cph * sth],
            [sth * sps + 0 * cph,
             sth * cps + 0 * cph,
             cth + 0 * (cph + cps)]])  # Make sure all components broadcast

    if squeeze_out:
        return mat.squeeze()
    else:
        # Move the `(ndim, ndim)` axes to the end
        extra_dims = len(np.broadcast(phi, theta, psi).shape)
        newaxes = list(range(2, 2 + extra_dims)) + [0, 1]
        return np.transpose(mat, newaxes)


def axis_rotation(axis, angle, vectors, axis_shift=(0, 0, 0)):
    """Rotate a vector or an array of vectors around an axis in 3d.

    The rotation is computed by `Rodrigues' rotation formula`_.

    Parameters
    ----------
    axis : `array-like`, shape ``(3,)``
        Rotation axis, assumed to be a unit vector.
    angle : float
        Angle of the counter-clockwise rotation.
    vectors : `array-like`, shape ``(3,)`` or ``(N, 3)``
        The vector(s) to be rotated.
    axis_shift : `array_like`, shape ``(3,)``, optional
        Shift the rotation center by this vector. Note that only shifts
        perpendicular to ``axis`` matter.

    Returns
    -------
    rot_vec : `numpy.ndarray`
        The rotated vector(s).

    References
    ----------
    .. _Rodrigues' rotation formula:
        https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula

    Examples
    --------
    Rotating around the third coordinate axis by and angle of 90 degrees:

    >>> axis = (0, 0, 1)
    >>> rot1 = axis_rotation(axis, angle=np.pi / 2, vectors=(1, 0, 0))
    >>> np.allclose(rot1, (0, 1, 0))
    True
    >>> rot2 = axis_rotation(axis, angle=np.pi / 2, vectors=(0, 1, 0))
    >>> np.allclose(rot2, (-1, 0, 0))
    True

    The rotation can be performed with shifted rotation center. A shift
    along the axis does not matter:

    >>> rot3 = axis_rotation(axis, angle=np.pi / 2, vectors=(1, 0, 0),
    ...                      axis_shift=(0, 0, 2))
    >>> np.allclose(rot3, (0, 1, 0))
    True

    The distance between the rotation center and the vector to be rotated
    determines the radius of the rotation circle:

    >>> # Rotation center in the point to be rotated, should do nothing
    >>> rot4 = axis_rotation(axis, angle=np.pi / 2, vectors=(1, 0, 0),
    ...                      axis_shift=(1, 0, 0))
    >>> np.allclose(rot4, (1, 0, 0))
    True
    >>> # Distance 2, thus rotates to (0, 2, 0) in the shifted system,
    >>> # resulting in (-1, 2, 0) from shifting back after rotating
    >>> rot5 = axis_rotation(axis, angle=np.pi / 2, vectors=(1, 0, 0),
    ...                      axis_shift=(-1, 0, 0))
    >>> np.allclose(rot5, (-1, 2, 0))
    True

    Rotation of multiple vectors can be done in bulk:

    >>> vectors = [[1, 0, 0], [0, 1, 0]]
    >>> rot = axis_rotation(axis, angle=np.pi / 2, vectors=vectors)
    >>> np.allclose(rot[0], (0, 1, 0))
    True
    >>> np.allclose(rot[1], (-1, 0, 0))
    True
    """
    rot_matrix = axis_rotation_matrix(axis, angle)
    vectors = np.asarray(vectors, dtype=float)
    if vectors.shape == (3,):
        vectors = vectors[None, :]
    elif vectors.ndim == 2 and vectors.shape[1] == 3:
        pass
    else:
        raise ValueError('`vectors` must have shape (3,) or (N, 3), got array '
                         'with shape {}'.format(vectors.shape))

    # Get `axis_shift` part that is perpendicular to `axis`
    axis_shift = np.asarray(axis_shift, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis_shift = axis_shift - axis.dot(axis_shift) * axis

    # Shift vectors with the negative of the axis shift to move the rotation
    # center to the origin. Then rotate and shift back.
    centered_vecs = vectors - axis_shift[None, :]
    # Need to transpose the vectors to make the axis of length 3 come first
    rot_vecs = rot_matrix.dot(centered_vecs.T).T
    return axis_shift[None, :] + rot_vecs


def axis_rotation_matrix(axis, angle):
    """Matrix of the rotation around an axis in 3d.

    The matrix is computed according to `Rodriguez' rotation formula`_.

    Parameters
    ----------
    axis : `array-like`, shape ``(3,)``
        Rotation axis, assumed to be a unit vector.
    angle : float or `array-like`
        Angle(s) of counter-clockwise rotation.

    Returns
    -------
    mat : `numpy.ndarray`, shape ``(3, 3)``
        The axis rotation matrix.

    References
    ----------
    .. _Rodriguez' rotation formula:
        https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
    """
    scalar_out = (np.shape(angle) == ())
    axis = np.asarray(axis)
    if axis.shape != (3,):
        raise ValueError('`axis` shape must be (3,), got {}'
                         ''.format(axis.shape))

    angle = np.array(angle, dtype=float, copy=False, ndmin=1)

    cross_mat = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
    dy_mat = np.outer(axis, axis)
    id_mat = np.eye(3)
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)

    # Add extra dimensions for broadcasting
    extra_dims = cos_ang.ndim
    mat_slc = (None,) * extra_dims + (slice(None), slice(None))
    ang_slc = (slice(None),) * extra_dims + (None, None)
    # Matrices will have shape (1, ..., 1, ndim, ndim)
    cross_mat = cross_mat[mat_slc]
    dy_mat = dy_mat[mat_slc]
    id_mat = id_mat[mat_slc]
    # Angle arrays will have shape (..., 1, 1)
    cos_ang = cos_ang[ang_slc]
    sin_ang = sin_ang[ang_slc]

    axis_mat = cos_ang * id_mat + (1. - cos_ang) * dy_mat + sin_ang * cross_mat
    if scalar_out:
        return axis_mat.squeeze()
    else:
        return axis_mat


def rotation_matrix_from_to(from_vec, to_vec):
    r"""Return a matrix that rotates ``from_vec`` to ``to_vec`` in 2d or 3d.

    Since a rotation from one vector to another in 3 dimensions has
    (at least) one degree of freedom, this function makes deliberate but
    still arbitrary choices to fix these free parameters. See Notes for
    details. For the applied formula in 3d, see `this Wikipedia page
    about Rodrigues' rotation formula
    <https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula>`_.

    Parameters
    ----------
    from_vec, to_vec : `array-like`, shape ``(2,)`` or ``(3,)``
        Vectors between which the returned matrix rotates. They should not
        be very close to zero or collinear.

    Returns
    -------
    matrix : `numpy.ndarray`, shape ``(2, 2)`` or ``(3, 3)``
        A matrix rotating ``from_vec`` to ``to_vec``. Note that the
        matrix does *not* include scaling, i.e. it is not guaranteed
        that ``matrix.dot(from_vec) == to_vec``.

    Examples
    --------
    In two dimensions, rotation is simple:

    >>> from_vec, to_vec = [1, 0], [1, 1]
    >>> mat = rotation_matrix_from_to(from_vec, to_vec)
    >>> to_vec_normalized = np.divide(to_vec, np.linalg.norm(to_vec))
    >>> np.allclose(mat.dot([1, 0]), to_vec_normalized)
    True
    >>> from_vec, to_vec = [1, 0], [-1, 1]
    >>> mat = rotation_matrix_from_to(from_vec, to_vec)
    >>> to_vec_normalized = np.divide(to_vec, np.linalg.norm(to_vec))
    >>> np.allclose(mat.dot([1, 0]), to_vec_normalized)
    True

    Rotation in 3d by less than ``pi``:

    >>> from_vec, to_vec = [1, 0, 0], [-1, 1, 0]
    >>> mat = rotation_matrix_from_to(from_vec, to_vec)
    >>> to_vec_normalized = np.divide(to_vec, np.linalg.norm(to_vec))
    >>> np.allclose(mat.dot([1, 0, 0]), to_vec_normalized)
    True

    Rotation by more than ``pi``:

    >>> from_vec, to_vec = [1, 0, 0], [-1, -1, 0]
    >>> mat = rotation_matrix_from_to(from_vec, to_vec)
    >>> to_vec_normalized = np.divide(to_vec, np.linalg.norm(to_vec))
    >>> np.allclose(mat.dot([1, 0, 0]), to_vec_normalized)
    True

    Notes
    -----
    In 3d, the matrix corresponds to a rotation around the normal vector
    :math:`\hat n = \hat u \times \hat v`, where :math:`\hat u` and
    :math:`\hat v` are the normalized versions of :math:`u`, the
    vector from which to rotate, and :math:`v`, the vector to which
    should be rotated.

    The rotation angle is determined as
    :math:`\alpha = \pm \arccos(\langle \hat u, \hat v \rangle)`.
    Its sign corresponds to the sign of
    :math:`\langle \hat b, \hat v\rangle`, where
    :math:`\hat b = \hat n \times \hat u` is the binormal vector.

    In the case that :math:`\hat u` and :math:`\hat v` are collinear,
    a perpendicular vector is chosen as :math:`\hat n = (1, 0, 0)` if
    :math:`v_1 = v_2 = 0`, else :math:`\hat n = (-v_2, v_1, v_3)`.
    The angle in this case is :math:`\alpha = 0` if
    :math:`\langle \hat u, \hat v \rangle > 0`, otherwise
    :math:`\alpha = \pi`.
    """
    from_vec, from_vec_in = (np.array(from_vec, dtype=float, copy=True),
                             from_vec)
    to_vec, to_vec_in = np.array(to_vec, dtype=float, copy=True), to_vec

    if from_vec.shape not in ((2,), (3,)):
        raise ValueError('`from_vec.shape` must be (2,) or (3,), got {}'
                         ''.format(from_vec.shape))
    if to_vec.shape not in ((2,), (3,)):
        raise ValueError('`to_vec.shape` must be (2,) or (3,), got {}'
                         ''.format(to_vec.shape))
    if from_vec.shape != to_vec.shape:
        raise ValueError('`from_vec.shape` and `to_vec.shape` not equal: '
                         '{} != {}'
                         ''.format(from_vec.shape, to_vec.shape))

    ndim = len(from_vec)

    # Normalize vectors
    from_vec_norm = np.linalg.norm(from_vec)
    if from_vec_norm < 1e-10:
        raise ValueError('`from_vec` {} too close to zero'.format(from_vec_in))
    from_vec /= from_vec_norm
    to_vec_norm = np.linalg.norm(to_vec)
    if to_vec_norm < 1e-10:
        raise ValueError('`to_vec` {} too close to zero'.format(to_vec_in))
    to_vec /= to_vec_norm

    if ndim == 2:
        dot = np.dot(from_vec, to_vec)
        from_rot = (-from_vec[1], from_vec[0])
        if dot == 0:
            angle = np.pi / 2 if np.dot(from_rot, to_vec) > 0 else -np.pi / 2
        elif np.array_equal(to_vec, -from_vec):
            angle = np.pi
        else:
            angle = (np.sign(np.dot(from_rot, to_vec)) *
                     np.arccos(np.dot(from_vec, to_vec)))
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    elif ndim == 3:
        # Determine normal
        normal = np.cross(from_vec, to_vec)
        normal_norm = np.linalg.norm(normal)

        if normal_norm < 1e-10:
            # Collinear vectors, use perpendicular vector and angle = 0 or pi
            normal = perpendicular_vector(from_vec)
            angle = 0 if np.dot(from_vec, to_vec) > 0 else np.pi
            return axis_rotation_matrix(normal, angle)
        else:
            # Usual case, determine binormal and sign of rotation angle
            normal /= normal_norm
            binormal = np.cross(normal, from_vec)
            angle = (np.sign(np.dot(binormal, to_vec)) *
                     np.arccos(np.dot(from_vec, to_vec)))
            return axis_rotation_matrix(normal, angle)

    else:
        raise RuntimeError('bad ndim')


def transform_system(principal_vec, principal_default, other_vecs,
                     matrix=None):
    """Transform vectors with either ``matrix`` or based on ``principal_vec``.

    The logic of this function is as follows:

    - If ``matrix`` is not ``None``, transform ``principal_vec`` and
      all vectors in ``other_vecs`` by ``matrix``, ignoring
      ``principal_default``.
    - If ``matrix`` is ``None``, compute the rotation matrix from
      ``principal_default`` to ``principal_vec``, not including the
      dilation. Apply that rotation to all vectors in ``other_vecs``.

    **Note:** All vectors must have the same shape and match the shape
    of ``matrix`` if given.

    Parameters
    ----------
    principal_vec : `array-like`, shape ``(ndim,)``
        Vector that defines the transformation if ``matrix`` is not
        provided.
    principal_default : `array-like`, shape ``(ndim,)``
        Default value for ``principal_vec``. The deviation from this
        determines the transformation.
        If ``matrix`` is given, this has no effect.
    other_vecs : sequence of ``None`` or `array-like`'s with shape ``(ndim,)``
        The other vectors that should be transformed. ``None`` entries
        are just appended as-is.
    matrix : `array-like`, shape ``(ndim, ndim)``, optional
        Explicit transformation matrix to be applied to the vectors.
        It is allowed to include a constant scaling but shouldn't have
        strongly varying directional scaling (bad condition).

    Returns
    -------
    transformed_vecs : tuple of `numpy.ndarray`, shape ``(ndim,)``
        The transformed vectors. The first entry is (the transformed)
        ``principal_vec``, followed by the transformed ``other_vecs``.
        Thus the length of the tuple is ``len(other_vecs) + 1``.
    """
    transformed_vecs = []
    principal_vec = np.asarray(principal_vec, dtype=float)
    ndim = principal_vec.shape[0]

    if matrix is None:
        # Separate into dilation and rotation. The dilation is only used
        # for comparison, not in the final matrix.
        principal_default = np.asarray(principal_default, dtype=float)

        pr_norm = np.linalg.norm(principal_vec)
        pr_default_norm = np.linalg.norm(principal_default)

        if pr_default_norm == 0.0 and pr_norm != 0.0:
            raise ValueError('no transformation from {} to {}'
                             ''.format(principal_default, principal_vec))
        elif pr_norm == 0.0 and pr_default_norm != 0.0:
            raise ValueError('transformation from {} to {} is singular'
                             ''.format(principal_default, principal_vec))
        elif pr_norm == 0.0 and pr_default_norm == 0.0:
            dilation = 1.0
        else:
            dilation = (np.linalg.norm(principal_vec) /
                        np.linalg.norm(principal_default))

        # Determine the rotation part
        if np.allclose(principal_vec, dilation * principal_default):
            # Dilation only
            matrix = np.eye(ndim)
        else:
            matrix = rotation_matrix_from_to(principal_default, principal_vec)

        # This one goes straight in
        transformed_vecs.append(principal_vec)

    else:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (ndim, ndim):
            raise ValueError('matrix shape must be {}, got {}'
                             ''.format((ndim, ndim), matrix.shape))

        # Check matrix condition
        svals = np.linalg.svd(matrix, compute_uv=False)
        condition = np.inf if 0.0 in svals else svals[0] / svals[-1]
        if condition > 1e6:
            raise np.linalg.LinAlgError(
                'matrix is badly conditioned: condition number is {}'
                ''.format(condition))

        transformed_vecs.append(matrix.dot(principal_vec))

    for vec in other_vecs:
        if vec is None:
            transformed_vecs.append(None)
        else:
            transformed_vecs.append(matrix.dot(vec))

    return tuple(transformed_vecs)


def is_rotation_matrix(mat, show_diff=False):
    from scipy.linalg import det, norm

    dim = mat.shape[0]
    if dim != mat.shape[1]:
        return False

    determ = det(mat)
    right_handed = (np.abs(determ - 1.) < 1E-10)
    orthonorm_diff = mat * mat.T - np.eye(dim)
    diff_norm = norm(orthonorm_diff, 2)
    orthonormal = (diff_norm < 1E-10)
    if not right_handed or not orthonormal:
        if show_diff:
            print('matrix S:\n', mat)
            print('det(S): ', determ)
            print('S*S.T - eye:\n', orthonorm_diff)
            print('2-norm of difference: ', diff_norm)
        return False
    return True


def angles_from_matrix(rot_matrix):
    if rot_matrix.shape == (2, 2):
        theta = np.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
        return theta,
    elif rot_matrix.shape == (3, 3):
        if rot_matrix[2, 2] == 1.:  # cannot use last row and column
            theta = 0.
            # upper-left block is 2d rotation for phi + psi, so one needs
            # to be fixed
            psi = 0.
            phi = np.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
            if phi < 0:
                phi += 2 * np.pi  # in [0, 2pi)
        else:
            phi = np.atan2(rot_matrix[0, 2], -rot_matrix[1, 2])
            psi = np.atan2(rot_matrix[2, 0], rot_matrix[2, 1])
            theta = np.acos(rot_matrix[2, 2])

            if phi < 0. or psi < 0.:
                phi += np.pi
                psi += np.pi
                theta = -theta

        return phi, theta, psi

    else:
        raise ValueError('shape of `rot_matrix` must be (2, 2) or (3, 3), '
                         'got {}'.format(rot_matrix.shape))


def to_lab_sys(vec_in_local_coords, local_sys):
    vec_in_local_coords = np.array(vec_in_local_coords)
    trafo_matrix = np.matrix(local_sys).T
    return np.dot(trafo_matrix, vec_in_local_coords)


def to_local_sys(vec_in_lab_coords, local_sys):
    vec_in_lab_coords = np.array(vec_in_lab_coords)
    trafo_matrix = np.matrix(local_sys)
    return np.dot(trafo_matrix, vec_in_lab_coords)


def perpendicular_vector(vec):
    """Return a vector perpendicular to ``vec``.

    Parameters
    ----------
    vec : `array-like`
        Vector(s) of arbitrary length. The axis along the vector components
        must come last.

    Returns
    -------
    perp_vec : `numpy.ndarray`
        Array of same shape as ``vec`` such that ``dot(vec, perp_vec) == 0``
        (along the last axis if there are multiple vectors).

    Examples
    --------
    Works in 2d:

    >>> perpendicular_vector([0, 1])
    array([-1.,  0.])
    >>> np.allclose(perpendicular_vector([1, 0]), [0, 1])  # would print -0
    True

    And in 3d:

    >>> perpendicular_vector([0, 1, 0])
    array([-1.,  0.,  0.])
    >>> perpendicular_vector([0, 0, 1])
    array([ 1.,  0.,  0.])
    >>> np.allclose(perpendicular_vector([1, 0, 0]), [0, 1, 0])
    True

    The function is vectorized, i.e., it can be called with multiple
    vectors at once (additional axes being added to the left):

    >>> perpendicular_vector([[0, 1, 0],
    ...                       [0, 0, 1]])  # 2 vectors
    array([[-1.,  0.,  0.],
           [ 1.,  0.,  0.]])
    >>> vecs = np.zeros((2, 3, 3))
    >>> vecs[..., 1] = 1  # (2, 3) array of vectors (0, 1, 0)
    >>> perpendicular_vector(vecs)
    array([[[-1.,  0.,  0.],
            [-1.,  0.,  0.],
            [-1.,  0.,  0.]],
    <BLANKLINE>
           [[-1.,  0.,  0.],
            [-1.,  0.,  0.],
            [-1.,  0.,  0.]]])
    """
    squeeze_out = (np.ndim(vec) == 1)
    vec = np.array(vec, dtype=float, copy=False, ndmin=2)

    if np.any(np.all(vec == 0, axis=-1)):
        raise ValueError('zero vector')

    result = np.zeros(vec.shape)
    cond = np.any(vec[..., :2] != 0, axis=-1)
    result[cond, 0] = -vec[cond, 1]
    result[cond, 1] = vec[cond, 0]
    result[~cond, 0] = 1
    result /= np.linalg.norm(result, axis=-1, keepdims=True)

    if squeeze_out:
        result = result.squeeze()

    return result


def is_inside_bounds(value, params):
    """Return ``True`` if ``value`` is contained in ``params``.

    This method supports broadcasting in the sense that for
    ``params.ndim >= 2``, if more than one value is given, the inputs
    are broadcast against each other.

    Parameters
    ----------
    value : `array-like`
        Value(s) to be checked. For several inputs, the final bool
        tells whether all inputs pass the check or not.
    params : `IntervalProd`
        Set in which the value is / the values are supposed to lie.

    Returns
    -------
    is_inside_bounds : bool
        ``True`` is all values lie in ``params``, ``False`` otherwise.

    Examples
    --------
    Check a single point:

    >>> params = odl.IntervalProd([0, 0], [1, 2])
    >>> is_inside_bounds([0, 0], params)
    True
    >>> is_inside_bounds([0, -1], params)
    False

    Using broadcasting:

    >>> pts_ax0 = np.array([0, 0, 1, 0, 1])[:, None]
    >>> pts_ax1 = np.array([2, 0, 1])[None, :]
    >>> is_inside_bounds([pts_ax0, pts_ax1], params)
    True
    >>> pts_ax1 = np.array([-2, 1])[None, :]
    >>> is_inside_bounds([pts_ax0, pts_ax1], params)
    False
    """
    if value in params:
        # Single parameter
        return True
    else:
        if params.ndim == 1:
            return params.contains_all(np.ravel(value))
        else:
            # Flesh out and flatten to check bounds
            bcast_value = np.broadcast_arrays(*value)
            stacked_value = np.vstack(bcast_value)
            flat_value = stacked_value.reshape(params.ndim, -1)
            return params.contains_all(flat_value)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
