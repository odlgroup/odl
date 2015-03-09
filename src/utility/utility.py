# -*- coding: utf-8 -*-
"""
utility.py -- utility functions

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division, unicode_literals, print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()

import numpy as np
from textwrap import dedent, fill


SQRT_2PI = 2.5066282746310002


def plot3d_scatter(arr, figsize=None, savefig=None):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=figsize)
    sub = plt.subplot(111, projection='3d')
    sub.set_xlabel('x')
    sub.set_ylabel('y')
    sub.set_zlabel('z')
    csub = sub.scatter(arr[:, 0], arr[:, 1], arr[:, 2])  # TODO: axes??
    plt.show()
    if savefig:
        fig.savefig(savefig)


def vec_list_from_arg(arg):
    """Turn argument into a vector tuple. The argument can be a single vector,
    a list or an array containing the vectors as columns."""

    if len(arg) > 1:  # a number of vectors was passed
        vecs = list(arg)
        for i in range(len(vecs)):
            vecs[i] = np.asarray(vecs[i])

    else:  # an array, a vector or a vector list was passed
        arg = arg[0]  # unwrap the tuple
        try:
            vecs = [arg[:, axis]
                    for axis in range(arg.shape[1])]  # array
        except AttributeError:  # list of vectors or single vector
            try:
                len(arg[0])  # list of vectors
                vecs = [np.asarray(arg[axis])
                        for axis in range(len(arg))]
            except TypeError:  # single vector
                vecs = [np.asarray(arg)]

    return vecs


def errfmt(errstr):
    return fill(dedent(errstr))


def flat_tuple(seq):
    try:
        iter(seq)
    except TypeError:
        return tuple([seq])

    while True:
        try:
            seq = [it for subseq in seq for it in subseq]
        except TypeError:
            return tuple(seq)


class InputValidationError(ValueError):
    """A simple exception class for input validation."""
    def __init__(self, input_, expected, input_name=None):
        self.got = input_
        self.expected = expected
        self.input_name = input_name

    def __str__(self):
        if self.input_name is not None:
            return errfmt("{}: got {}, expected {}".format(self.input_name,
                          self.got, self.expected))
        else:
            return errfmt("got {}, expected {}".format(self.got,
                          self.expected))


def euler_matrix(*angles):
    """Compute Euler rotation matrix from angles given in radians. Its rows
    represent components the canonical unit vectors in the rotated system
    while the columns are the rotated unit vectors as seen from the
    canonical system.
    TODO: write properly
    """

    from math import sin, cos

    if len(angles) == 1:
        theta = angles[0]
        phi = psi = 0.
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

    mat = np.matrix(
        [[cph*cps - sph*cth*sps, -cph*sps - sph*cth*cps,  sph*sth],
         [sph*cps + cph*cth*sps, -sph*sps + cph*cth*cps, -cph*sth],
         [              sth*sps,                sth*cps,      cth]])

    return mat


def axis_rotation(vec, axis, angle):

    from math import cos, sin, pi

    vec = np.array(vec)
    axis = np.array(axis)
    angle = float(angle)

    if angle == 0. or angle == 2 * pi:
        return vec
    else:
        cos_ang = cos(angle)
        sin_ang = sin(angle)
        scal = np.dot(axis, vec)
        cross = np.cross(axis, vec)

        return cos_ang * vec + (1. - cos_ang) * scal * axis + sin_ang * cross


def axis_rotation_matrix(axis, angle):

    from math import cos, sin

    axis = np.array(axis)
    angle = float(angle)
    cos_ang = cos(angle)
    sin_ang = sin(angle)

    x_mat = np.matrix([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
    dy_mat = np.outer(axis, axis)
    id_mat = np.eye(3)

    return cos_ang * id_mat + (1. - cos_ang) * dy_mat + sin_ang * x_mat


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

    from math import atan2, acos, pi

    if rot_matrix.shape == (2, 2):
        theta = atan2(rot_matrix[1, 0], rot_matrix[0, 0])
        return (theta,)
    elif rot_matrix.shape == (3, 3):
        if rot_matrix[2, 2] == 1.:  # cannot use last row and column
            theta = 0.
            # upper-left block is 2D rotation for phi + psi, so one needs
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

        return (phi, theta, psi)

    else:
        raise InputValidationError(rot_matrix.shape, '(2,2) or (3,3)',
                                   'rot_matrix.shape')


def to_lab_sys(vec_in_local_coords, local_sys):

    vec_in_local_coords = np.array(vec_in_local_coords)
    trafo_matrix = np.matrix(local_sys).T
    return np.dot(trafo_matrix, vec_in_local_coords)


def to_local_sys(vec_in_lab_coords, local_sys):

    vec_in_lab_coords = np.array(vec_in_lab_coords)
    trafo_matrix = np.matrix(local_sys)
    return np.dot(trafo_matrix, vec_in_lab_coords)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
