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

# Imports for common Python 2/3 codebase
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
from math import sin, cos, atan2, acos, pi
from future import standard_library

standard_library.install_aliases()
from builtins import int

# External
import numpy as np


def euler_matrix(*angles):
    """Rotation matrix in 2 and 3 dimensions.

    Compute the Euler rotation matrix from angles given in radians.
    Its rows represent the canonical unit vectors as seen from the
    rotated system while the columns are the rotated unit vectors as
    seen from the canonical system.

    Parameters
    ----------
    angle1,...,angleN : RealNumber
        One angle results in a (2x2) matrix representing a
        counter-clockwise rotation. Two or three angles result in a
        (3x3) matrix and are interpreted as Euler angles of a 3d
        rotation according to the 'ZXZ' rotation order, see the
        Wikipedia article `Euler angles`_.

    Returns
    -------
    mat : `numpy.ndarray`, shape `(2, 2)` or `(3, 3)`
        The rotation matrix

    .. _Euler angles:
        https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    """
    dim = 3

    if len(angles) == 1:
        phi = angles[0]
        theta = psi = 0.
        dim = 2
    elif len(angles) == 2:
        phi = angles[0]
        theta = angles[1]
        psi = 0.
    elif len(angles) == 3:
        phi = angles[0]
        theta = angles[1]
        psi = angles[2]
    else:
        raise ValueError('Number of angles must be between 1 and 3')

    cph = cos(phi)
    sph = sin(phi)
    cth = cos(theta)
    sth = sin(theta)
    cps = cos(psi)
    sps = sin(psi)

    if dim == 2:
        mat = np.array([[cph, -sph], [sph, cph]])
    else:
        mat = np.array([[cph * cps - sph * cth * sps,
                         -cph * sps - sph * cth * cps, sph * sth],
                        [sph * cps + cph * cth * sps,
                         -sph * sps + cph * cth * cps, -cph * sth],
                        [sth * sps, sth * cps, cth]])

    return mat


def axis_rotation(axis, angle, vectors):
    """Rotate a vector or an array of vectors around an axis in 3d.

    The rotation is computed by `Rodriguez' rotation formula`_.

    Parameters
    ----------
    axis : array-like, shape (3,)
        The rotation axis, assumed to be a unit vector
    angle : `float`
        The rotation angle
    vectors : array-like, shape `(3,)` or `(N, 3)`
        The vector(s) to be rotated

    Returns
    -------
    rot_vec : `np.ndarray`
        The rotated vector(s)

    .. _Rodriguez' rotation formula:
        https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
    """
    if not (hasattr(vectors, 'shape') and hasattr(vectors, 'ndim')):
        vectors = np.asarray(vectors)

    if not (vectors.shape == (3,) or (vectors.ndim == 2 and
                                      vectors.shape[1] == 3)):
        raise ValueError('`vector` shape {} not (3,) or (N, 3).'
                         ''.format(vectors.shape))

    if not hasattr(axis, 'shape'):
        axis = np.asarray(axis)

    if axis.shape != (3,):
        raise ValueError('`axis` shape {} not (3,).'.format(axis.shape))

    angle = float(angle)

    if np.isclose(angle / (2 * pi), int(angle / (2 * pi)), atol=1e-15):
        return vectors.copy()
    else:
        cos_ang = cos(angle)
        sin_ang = sin(angle)
        scal = np.asarray(np.dot(vectors, axis))
        cross = np.asarray(np.cross(vectors, axis))
        cross *= sin_ang

        rot_vecs = cos_ang * vectors
        rot_vecs += (1. - cos_ang) * scal[:, None] * axis[None, :]
        rot_vecs += cross


def axis_rotation_matrix(axis, angle):
    """Matrix of the rotation around an axis in 3d.

    The matrix is computed according to `Rodriguez' rotation formula`_.

    Parameters
    ----------
    axis : array-like, shape (3,)
        The rotation axis, assumed to be a unit vector
    angle : `float`
        The rotation angle

    Returns
    -------
    mat : `numpy.ndarray`, shape (3, 3)
        The axis rotation matrix

    .. _Rodriguez' rotation formula:
        https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
    """
    if not hasattr(axis, 'shape'):
        axis = np.asarray(axis)

    if axis.shape != (3,):
        raise ValueError('`axis` shape {} not (3,).'.format(axis.shape))

    angle = float(angle)

    cross_mat = np.matrix([[0, -axis[2], axis[1]],
                           [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])
    dy_mat = np.asmatrix(np.outer(axis, axis))
    id_mat = np.asmatrix(np.eye(3))
    cos_ang = cos(angle)
    sin_ang = sin(angle)

    return cos_ang * id_mat + (1. - cos_ang) * dy_mat + sin_ang * cross_mat


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
        theta = atan2(rot_matrix[1, 0], rot_matrix[0, 0])
        return theta,
    elif rot_matrix.shape == (3, 3):
        if rot_matrix[2, 2] == 1.:  # cannot use last row and column
            theta = 0.
            # upper-left block is 2d rotation for phi + psi, so one needs
            # to be fixed
            psi = 0.
            phi = atan2(rot_matrix[1, 0], rot_matrix[0, 0])
            if phi < 0:
                phi += 2 * pi  # in [0, 2pi)
        else:
            phi = atan2(rot_matrix[0, 2], -rot_matrix[1, 2])
            psi = atan2(rot_matrix[2, 0], rot_matrix[2, 1])
            theta = acos(rot_matrix[2, 2])

            if phi < 0. or psi < 0.:
                phi += pi
                psi += pi
                theta = -theta

        return phi, theta, psi

    else:
        raise ValueError('shape of `rot_matrix` is {}, expected (2, 2) '
                         'or (3, 3).'.format(rot_matrix.shape))


def to_lab_sys(vec_in_local_coords, local_sys):
    vec_in_local_coords = np.array(vec_in_local_coords)
    trafo_matrix = np.matrix(local_sys).T
    return np.dot(trafo_matrix, vec_in_local_coords)


def to_local_sys(vec_in_lab_coords, local_sys):
    vec_in_lab_coords = np.array(vec_in_lab_coords)
    trafo_matrix = np.matrix(local_sys)
    return np.dot(trafo_matrix, vec_in_lab_coords)


def perpendicular_vector(vec):
    """ Get a vector perpendicular to v.

    Parameters
    ----------
    vec : array-like
        Array-like of any dimension > 2.

    Returns
    -------
    perp_vec : `numpy.ndarray`
        Array of same size such that ``<vec, perp_vec> == 0``

    Examples
    --------
    Works in 2d
    >>> perpendicular_vector([1, 0])
    array([ 0.,  1.])
    >>> perpendicular_vector([0, 1])
    array([-1.,  0.])

    And in 3d
    >>> perpendicular_vector([1, 0, 0])
    array([ 0.,  1.,  0.])
    >>> perpendicular_vector([0, 1, 0])
    array([-1.,  0.,  0.])
    >>> perpendicular_vector([0, 0, 1])
    array([ 1.,  0.,  0.])
    """
    vec = np.asarray(vec)

    if np.all(vec == 0):
        raise ValueError('zero vector')

    result = np.zeros(vec.shape)
    if np.any(vec[:2] != 0):
        result[:2] = [-vec[1], vec[0]]
    else:
        result[0] = 1

    return result / np.linalg.norm(result)
