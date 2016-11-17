# Copyright 2014-2016 The ODL development group
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

"""Extended MRC header as used by FEI, with print function."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


__all__ = ('MRC_FEI_EXT_HEADER_SECTION', 'print_fei_ext_header_spec')


MRC_FEI_EXT_HEADER_SECTION = """
+---------+---------+---------+---------------+------------------------------+
|Long word|Byte     |Data type|Name           |Description                   |
+=========+=========+=========+===============+==============================+
|1        |1025-1028|Float32  |A_TILT         |Alpha tilt, in degrees        |
+---------+---------+---------+---------------+------------------------------+
|2        |1029-1032|Float32  |B_TILT         |Beta tilt, in degrees         |
+---------+---------+---------+---------------+------------------------------+
|3        |1033-1036|Float32  |X_STAGE        |Stage x position. Normally in |
|         |         |         |               |SI units (meters), but some   |
|         |         |         |               |older files may be in         |
|         |         |         |               |micrometers.(values larger    |
|         |         |         |               |than 1)                       |
+---------+---------+---------+---------------+------------------------------+
|4        |1037-1040|Float32  |Y_STAGE        |Stage y position              |
+---------+---------+---------+---------------+------------------------------+
|5        |1041-1044|Float32  |Z_STAGE        |Stage z position              |
+---------+---------+---------+---------------+------------------------------+
|6        |1045-1048|Float32  |X_SHIFT        |Stage x shift. For units see  |
|         |         |         |               |remarks on X_STAGE            |
+---------+---------+---------+---------------+------------------------------+
|7        |1049-1052|Float32  |Y_SHIFT        |Stage y shift                 |
+---------+---------+---------+---------------+------------------------------+
|8        |1053-1056|Float32  |DEFOCUS        |Defocus as read from the      |
|         |         |         |               |microscope. For units see     |
|         |         |         |               |remarks on X_STAGE.           |
+---------+---------+---------+---------------+------------------------------+
|9        |1057-1060|Float32  |EXP_TIME       |Exposure time in seconds      |
+---------+---------+---------+---------------+------------------------------+
|10       |1061-1064|Float32  |MEAN_INT       |Mean value of the image       |
+---------+---------+---------+---------------+------------------------------+
|11       |1065-1068|Float32  |TILT_AXIS      |Orientation of the tilt axis  |
|         |         |         |               |in the image in degrees.      |
|         |         |         |               |Vertical to the top is 0      |
|         |         |         |               |degrees, the direction of     |
|         |         |         |               |positive rotation is          |
|         |         |         |               |anti-clockwise.               |
+---------+---------+---------+---------------+------------------------------+
|12       |1069-1072|Float32  |PIXEL_SIZE     |Pixel size of the images in SI|
|         |         |         |               |units (meters)                |
+---------+---------+---------+---------------+------------------------------+
|13       |1073-1076|Float32  |MAGNIFICATION  |Magnification used for        |
|         |         |         |               |recording the images          |
+---------+---------+---------+---------------+------------------------------+
|14       |1077-1080|Float32  |HT             |Value of the high tension in  |
|         |         |         |               |SI units (volts)              |
+---------+---------+---------+---------------+------------------------------+
|15       |1081-1084|Float32  |BINNING        |The binning of the CCD or STEM|
|         |         |         |               |acquisition                   |
+---------+---------+---------+---------------+------------------------------+
|16       |1085-1088|Float32  |APPLIED_DEFOCUS|The intended application      |
|         |         |         |               |defocus in SI units (meters), |
|         |         |         |               |as defined for example in the |
|         |         |         |               |tomography parameters view    |
+---------+---------+---------+---------------+------------------------------+
"""


def print_fei_ext_header_spec():
    """Print the specification table of an FEI extended header section.

    The specification table is as follows:
    """
    print(MRC_FEI_EXT_HEADER_SECTION)

print_fei_ext_header_spec.__doc__ += MRC_FEI_EXT_HEADER_SECTION


MRC_FEI_SECTION_SIZE = 128
MRC_FEI_MAX_SECTIONS = 1024

ANGSTROM_IN_METERS = 1e-10
MICRON_IN_METERS = 1e-6
