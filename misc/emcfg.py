# -*- coding: utf-8 -*-
"""
emcfg.py -- I/O for electron microscope configuration

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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

import numpy as np
from math import sqrt, pi
from textwrap import dedent


# TODO: use decorators for error handling
# TODO: define standard error string
# TODO: write simple docstring tests for doctest
# TODO: use subclassing for CTF and file import (later)

try:
    from configobj import ConfigObj, ConfigObjError, flatten_errors
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


def import_cfg(cfgfile):
    """Import settings from specified file to a dictionary.
    TODO: finish this properly"""

    import io
    cfgspec_stringio = io.StringIO(dedent("""

    [geometry]

    tilt_axis = float(min=-180, max=180, default=0)
    tilt_axis_shift_x = integer(default=0)
    tilt_axis_shift_y = integer(default=0)

    [electronbeam]

    acc_voltage = float(min=0, max=1000)
    energy_spread = float(min=0, max=10, default=1)

    [optics]

    magnification = float(min=1, max=1000000)
    cs = float(min=0, max=10, default=2)
    cc = float(min=0, max=10, default=2)
    aperture = float(min=0, max=1000, default=50)
    focal_length = float(min=0, max=100, default=2)
    cond_ap_angle = float(min=0, max=10, default=0.1)
    defocus_nominal = float(min=0, max=100)
    """))

    cfgspec = ConfigObj(cfgspec_stringio, list_values=False, _inspec=True)

    try:
        cfg = ConfigObj(infile=cfgfile, list_values=False, file_error=True,
                        configspec=cfgspec)
    except IOError:
        print("Unable to read-only open file '" + cfgfile + "'.")
        return None

    # TODO: check error reporting

    val = Validator()
    validated = cfg.validate(val, preserve_errors=True)

    if validated is not True:
        for (section_list, key, error) in flatten_errors(cfg, validated):
            if key is not None:
                if error is False:
                    emsg = "Key '{}' in section '{}' not found.".format(
                        key, ': '.join(section_list))
                else:
                    emsg = "Key '{}' in section '{}': {}.".format(
                        key, ': '.join(section_list), error.message)
            else:
                emsg = "Section '{}' missing.".format(', '.join(section_list))

        raise ConfigObjError(emsg)

    return cfg.dict()


E_REST_ENERGY = 510998.928  # [eV]
H_C = 1239.84193  # [eV * nm]

KILOVOLT = 1000  # [V]
MICROMETER = 1000  # [nm]
MILLIMETER = 1000000  # [nm]
MILLIRADIAN = 1E-4  # [rad]


class EMConfig(object):
    """Class for electron microscope configuration"""

    def __init__(self, cfgfile):

        cfg = import_cfg(cfgfile)

        self._voltage = cfg['electronbeam']['acc_voltage'] * KILOVOLT  # [V]
        self._spread = cfg['electronbeam']['energy_spread']  # [eV]
        self._magnif = cfg['optics']['magnification']
        self._defocus = cfg['optics']['defocus_nominal'] * MICROMETER  # [nm]
        self._sph_aberr = cfg['optics']['cs'] * MILLIMETER  # [nm]
        self._chr_aberr = cfg['optics']['cc'] * MILLIMETER  # [nm]
        self._focal_len = cfg['optics']['focal_length'] * MILLIMETER  # [nm]
        self._aperture = cfg['optics']['aperture'] * MICROMETER  # [nm]
        self._cond_ap_angle = cfg['optics']['cond_ap_angle'] * MILLIRADIAN
        # [rad]

    @property
    def voltage(self):
        return self._voltage

    @property
    def energy_spread(self):
        return self._spread

    @property
    def magnification(self):
        return self.magnif

    @property
    def defocus(self):
        return self._defocus

    @property
    def CS(self):
        return self._sph_aberr

    @property
    def CC(self):
        return self._chr_aberr

    @property
    def focal_length(self):
        return self._focal_len

    @property
    def aperture(self):
        return self._aperture

    @property
    def cond_ap_angle(self):
        return self._cond_ap_angle

    @property
    def wavenum(self):
        """The relativistic wave number of an electron"""
        p_c = sqrt(self.voltage * (self.voltage + 2 * E_REST_ENERGY))

        return 2 * pi * p_c / H_C  # [nm^(-1)]

    @property
    def src_factor(self):
        """A factor in the source size envelope of the CTF"""
        efac = 1. / (2 * E_REST_ENERGY)

        return (1. + 2 * efac * self.voltage) / (
            self.voltage * (1. + efac * self.voltage))

    # FIXME: check coefficient ranges (env and osc don't match well)
    @property
    def env_polycoeff(self):
        """Polynomial coefficients for CTF envelope function.
        TODO: finish this"""

        # spread: (espr * CC' / (4 * k) * mxi^2)^2,  CC' = factor * CC
        # srcsize: alpha_c / 2 * mxi^2 * (CS / k^2 * mxi^2 - defoc)^2

        # Polynomial coefficients (index 0 = highest order)
        coeff = np.empty((4,))

        coeff[0] = self.magnif**6 / self.wavenum**4 * (
            self.cond_ap_angle * self.sph_aberr**2 / 2.)

        coeff[1] = self.magnif**4 / self.wavenum**2 * (
            self.spread**2 * (self.src_factor * self.chr_aberr)**2 / 16. -
            self.cond_ap_angle * self.defocus * self.sph_aberr)

        coeff[2] = self.magnif**2 * (
            self.cond_ap_angle * self.defocus**2 / 2.)

        coeff[3] = 0

        return coeff

    @property
    def osc_polycoeff(self):
        """Polynomial coefficients for CTF oscillating part.
        TODO: finish this"""

        # -mxi^2 / (4*k) * (CS / k^2 * mxi^2 - 2*defoc)

        # Polynomial coefficients (index 0 = highest order)
        coeff = np.empty((4,))

        coeff[0] = 0.0

        coeff[1] = self.magnif**4 / self.wavenum**3 * (
            -self.sph_aberr / 4.)

        coeff[2] = self.magnif**2 / self.wavenum * (
            self.defocus / 2.)

        coeff[3] = 0

        return coeff

    def __str__(self):
        return dedent("""\
        acceleration voltage: {0.voltage}
        energy spread       : {0.spread}
        magnification       : {0.magnif}
        nominal defocus     : {0.defocus}
        sperical aberration : {0.sph_aberr}
        chromatic aberration: {0.chr_aberr}
        focal length        : {0.focal_len}
        aperture            : {0.aperture}
        cond. aperture angle: {0.cond_ap_angle}
        wave number         : {0.wavenum}
        source factor       : {0.src_factor}

        polynomial coefficients envelope (highest order first):
        {0.env_polycoeff}

        polynomial coefficients oscillating part (highest order first):
        {0.osc_polycoeff}
        """.format(self))
