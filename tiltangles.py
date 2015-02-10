# -*- coding: utf-8 -*-
"""
tiltangles.py -- I/O for tiltangles files

Copyright 2014 Holger Kohr

This file is part of tomok.

tomok is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tomok is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with tomok.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from textwrap import dedent

#############################################################################


def fromfile(filename):
    try:
        f = open(filename, 'r')
    except IOError:
        print "Unable to read-only open file '" + filename + "'."
        return None

    with f:
        lines = []
        while 1:
            line = f.readline().strip()
            if line == '':
                break
            if not line.startswith('#'):
                lines.append(line)

    # Get the meta-info (number of tilts and number of angles)
    try:
        nlst = lines.pop(0).split()
        ntlt = int(nlst[0])
        nang = int(nlst[1])
    except IndexError:
        print dedent("""\
            Expected a line "<ntilts>  <nangles>" at the beginning of the file.
            """)

    if nang > 3:
        print '<nangles> must not be larger than 3.'
        return None

    # Check if number of angle lines corresponds to provided tilt number
    if not ntlt == len(lines):
        print 'Number of angle lines ({}) not equal to <ntilts> ({}).'.format(
            len(lines), ntlt)
        return None

    # Store the angles as floats in a new array and return it
    angles = np.zeros((ntlt, nang), dtype=np.float32)

    for i in xrange(ntlt):
        ang_strlst = lines.pop(0).split()
        if not nang == len(ang_strlst):
            print dedent("""\
            Number of angles at tilt index {} ({}) not equal to <nangles> ({}).
            """.format(i + 1, len(ang_strlst), nang))
            return None

        angles[i, :] = np.asarray(ang_strlst, dtype=np.float32)

    return angles

#############################################################################


def angles2matrix(ang_deg):
    phi = 0.0
    theta = 0.0
    psi = 0.0

    one_degree = np.pi / 180.0

    if len(ang_deg) == 1:
        theta = ang_deg[0] * one_degree
    elif len(ang_deg) == 2:
        phi = ang_deg[0] * one_degree
        theta = ang_deg[1] * one_degree
    elif len(ang_deg) == 3:
        phi = ang_deg[0] * one_degree
        theta = ang_deg[1] * one_degree
        psi = ang_deg[2] * one_degree
    else:
        raise ValueError('Number of angles must be between 1 and 3')

    from math import sin, cos
    cph = cos(phi)
    sph = sin(phi)
    cth = cos(theta)
    sth = sin(theta)
    cps = cos(psi)
    sps = sin(psi)

    R = np.matrix(
        [[cph*cps - sph*cth*sps, -cph*sps - sph*cth*cps,  sph*sth],
         [sph*cps + cph*cth*sps, -sph*sps + cph*cth*cps, -cph*sth],
         [              sth*sps,                sth*cps,      cth]])

    return R

#############################################################################
