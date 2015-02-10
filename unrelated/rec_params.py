# -*- coding: utf-8 -*-
"""
rec_params.py -- I/O for reconstruction parameter config files

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

from textwrap import dedent
try:
    from configobj import ConfigObj, flatten_errors
except ImportError:
    raise ImportError(dedent("""\

    Error: The 'configobj' module seems to be missing. Please install it
    through your OS's package manager or directly from PyPI:

    https://pypi.python.org/pypi/configobj/
    """))

try:
    from validate import Validator
except ImportError:
    raise ImportError(dedent("""\

    Error: The 'validate' module seems to be missing. Please install it
    through your OS's package manager or directly from PyPI:

    https://pypi.python.org/pypi/validate/
    """))


def fromfile(filename):
    """Import settings from specified file to a dictionary.
    TODO: finish this properly"""

    import StringIO
    cfgspec_stringio = StringIO.StringIO(dedent("""\

    [volume]

    nx = integer(min=1)
    ny = integer(min=1)
    nz = integer(min=1)
    voxel_size = float(min=0.0)
    shift_x = integer(default=0)
    shift_y = integer(default=0)
    shift_z = integer(default=0)

    [geometry]

    tilt_axis = float(min=-180, max=180, default=0)
    tilt_axis_shift_x = integer(default=0)
    tilt_axis_shift_y = integer(default=0)

    [electronbeam]

    acc_voltage = float(min=0, max=1000, default=0)
    energy_spread = float(min=0, max=10, default=1)

    [optics]

    magnification = float(min=1, max=1000000, default=1)
    cs = float(min=0, max=10, default=2)
    cc = float(min=0, max=10, default=2)
    aperture = float(min=0, max=1000, default=40)
    focal_length = float(min=0, max=100, default=2)
    cond_ap_angle = float(min=0, max=10, default=0.1)
    defocus_nominal = float(min=0, max=100, default=5)

    [detector]

    pixel_size = float(min=0.01, max=1000)
    mtf_a = float(min=0, max=10, default=0)
    mtf_b = float(min=0, max=10, default=0)
    mtf_c = float(min=0, max=10, default=1)
    mtf_alpha = integer(min=1, max=200, default=1)
    mtf_beta = integer(min=1, max=200, default=1)
    """))

    cfgspec = ConfigObj(cfgspec_stringio, list_values=False, _inspec=True)

    try:
        cfg = ConfigObj(infile=filename, list_values=False, file_error=True,
                        configspec=cfgspec)
    except IOError:
        print("Unable to read-only open file '" + filename + "'.")
        return None

    # TODO: fix validation error reporting

    val = Validator()
    validated = cfg.validate(val, preserve_errors=True)

    if validated is not True:
        for (section_list, key, _) in flatten_errors(cfg, validated):
            if key is not None:
                print("Key '{}' in section '{}' failed validation.".format(
                    key, ', '.join(section_list)))
            else:
                print("Section '{}' missing.".format(', '.join(section_list)))

        return None

    return cfg.dict()


#    cfg = cp.RawConfigParser()
#
#    l = cfg.read(filename)
#
#    if not l:
#        print("Unable to read-only open file '" + filename + "'.")
#        return None
#
#    d = {}
#
#    # Options with defaults first
#    try:
#        d['shift x'] = cfg.getfloat('volume', 'shift_x')
#    except cp.NoOptionError:
#        d['shift x'] = 0.0
#
#    try:
#        d['shift y'] = cfg.getfloat('volume', 'shift_y')
#    except cp.NoOptionError:
#        d['shift y'] = 0.0
#
#    try:
#        d['shift z'] = cfg.getfloat('volume', 'shift_z')
#    except cp.NoOptionError:
#        d['shift z'] = 0.0
#
#    try:
#        d['tiltaxis rotation'] = cfg.getfloat('geometry', 'tilt_axis')
#    except cp.NoOptionError:
#        d['tiltaxis rotation'] = 0.0
#
#    if abs(d['tiltaxis rotation']) % 180 == 90.0:
#        option = 'tilt_axis_shift_y'
#    else:
#        option = 'tilt_axis_shift_x'
#
#    try:
#        d['tiltaxis shift'] = cfg.getfloat('geometry', option)
#    except cp.NoOptionError:
#        d['tiltaxis shift'] = 0.0
#
#    try:
#        d['mtf a'] = cfg.getfloat('detector', 'mtf_a')
#    except cp.NoOptionError:
#        d['mtf a'] = 0.0
#
#    try:
#        d['mtf b'] = cfg.getfloat('detector', 'mtf_b')
#    except cp.NoOptionError:
#        d['mtf b'] = 0.0
#
#    try:
#        d['mtf c'] = cfg.getfloat('detector', 'mtf_c')
#    except cp.NoOptionError:
#        d['mtf c'] = 1.0
#
#    try:
#        d['mtf alpha'] = cfg.getint('detector', 'mtf_alpha')
#    except cp.NoOptionError:
#        d['mtf alpha'] = 1
#
#    try:
#        d['mtf beta'] = cfg.getint('detector', 'mtf_beta')
#    except cp.NoOptionError:
#        d['mtf beta'] = 1
#
#    # The rest is obligatory
#    try:
#        d['voxel size'] = cfg.getfloat('volume', 'voxel_size')
#        d['voltage'] = cfg.getfloat('electronbeam', 'acc_voltage') * 1E3
#        d['energy spread'] = cfg.getfloat('electronbeam', 'energy_spread')
#        d['magnification'] = cfg.getfloat('optics', 'magnification')
#        d['spherical aberration'] = cfg.getfloat('optics', 'cs') * 1E6
#        d['chromatic aberration'] = cfg.getfloat('optics', 'cc') * 1E6
#        d['aperture'] = cfg.getfloat('optics', 'aperture') * 1E3
#        d['focal length'] = cfg.getfloat('optics', 'focal_length') * 1E6
#        d['cond aperture angle'] = cfg.getfloat('optics', 'cond_ap_angle') \
#            * 1E-3
#        d['defocus'] = cfg.getfloat('optics', 'defocus_nominal') * 1E3
#        d['pixel size'] = cfg.getfloat('detector', 'pixel_size') * 1E3
#    except cp.NoOptionError as e:
#        print 'Error reading from ' + filename + ':'
#        print e.message
#        return None
#
#    return d
