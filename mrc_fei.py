# -*- coding: utf-8 -*-
"""
mrc_fei.py -- I/O for MRC files as used by FEI

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

format_spec = """
The extended MRC format for tomography as used by FEI. Length 1024 bytes.
-------------------------------------------------------------------------

Value   Format  Byte    Explanation


nx      int4      0     The number of pixels in the x direction of the image
ny      int4      4     The number of pixels in the x direction of the image
nz      int4      8     The number of pixels in the x direction of the image

mode    int4     12     Defines the data type. Should always be 1 (2-byte
                        integer) in the case of tomography (or 2 for float).

nxstart int4     16     set to 0 : not used
nystart int4     20     set to 0 : not used
nzstart int4     24     set to 0 : not used

mx      int4     28     set to nx : not used
mx      int4     32     set to ny : not used
mx      int4     36     set to nz : not used

xlen    float4   40     set to mx : not used
ylen    float4   44     set to my : not used
zlen    float4   48     set to mz : not used

alpha   float4   52     set to 90 : not used
beta    float4   56     set to 90 : not used
gamma   float4   60     set to 90 : not used

mapc    int4     64     set to 1 : not used
mapr    int4     68     set to 2 : not used
maps    int4     72     set to 3 : not used

amin    float4   76     minimum pixel value of all images in file
amax    float4   80     maximum pixel value of all images in file
amean   float4   84     mean pixel value of all images in file

ispg    int2     88     set to 0 : not used
nsymbt  int2     90     set to 0 : not used
next    int4     92     number of bytes in extended header
creatid int2     96     set to 0 : not used

extra   30 B     98     set to 0 : not used
nint    int2    128     set to 0 : not used
nreal   int2    130     set to 32; we always expect a extended header of 32
                        floats
sub     int2    132
zfac    int2    134
min2    float4  136
max2    float4  140
min3    float4  144
max3    float4  148
min4    float4  152
max4    float4  156
idtype  int2    160     0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins
lens    int2    162
nd1     int2    164     for idtype = 1, nd1 = axis (1, 2, or 3)
nd2     int2    166
vd1     int2    168     vd1 = 100. * tilt increment
vd2     int2    170     vd1 = 100. * starting angle
tilts   float4* 172     set to 0 : not used (array of 9 floats)
zorg    float4  208     set to 0 : not used
xorg    float4  212     set to 0 : not used
yorg    float4  216     set to 0 : not used

nlabl   int4    220     set to 1; number of labels
labl    char**  224     10 labels of 80 charactors. Label 0 is used for
                        copyright information (FEI)

------------------------------------------------------------------------

The extended header.

Contains the information about a maximum of 1024 images. Each section is 128
bytes long. The extended header is thus 1024 * 128 bytes (always the same
length, regardless of how many images are present.

Spatial values are normally in SI units (meters), but some older files may be
in micrometers. Check by looking at [xyz]_stage. If one of these exceeds 1,
it will be micrometers.


Value           Format  Explanation


a_tilt          float4  Alpha tilt, in degrees
b_tilt          float4  Beta tilt, in degrees

x_stage         float4  Stage x position
y_stage         float4  Stage y position
z_stage         float4  Stage z position

x_shift         float4  Image x shift
y_shift         float4  Image y shift

defocus         float4  Defocus as read from microscope
exp_time        float4  Exposure time in seconds

mean_int        float4  Mean value of image
tilt_axis       float4  The orientation of the tilt axis in the image in
                        degrees. Vertical to to top is 0°, the direction of
                        positive rotation is anti-clockwise.
pixel_size      float4  The pixel size of the images in SI units (meters)
magnification   float4  The magnification used for recording the images
ht              float4  Value of the high tension in SI units (Volts)
binning         float4  The binning of the CCD or STEM acquisition
appliedDefocus  float4  The intended application defocus in SI units (meters)
                        as defined for example in the tomography parameters
                        view

remainder       float4* unused values, filling up to 128 bytes (64 bytes)
"""

_usage = """
Usage: %s <testfile>.mrc

Reads the file, displays the header information and writes a plot of
the data into <testfile>.png
"""

_mode_types = {1: "short", 2: "float32"}

_type_modes = {}    # reversed
for key, val in _mode_types.items():
    _type_modes[val] = key

# ----------------------------------------------------------------------------


def get_header(file):
    """
    Read the header information from an MRC file and return it as a
    dictionary.

    Parameters
    ----------
    file: file-like
        where the data is read from

    Returns
    -------
    out: dictionary
        the header information
    """

    import struct

    header = {}
    file.seek(0)

    # Read the data from the file

    # STANDARD MRC HEADER

    # Bytes 0 -- 11: (x,y,z) shape
    shape = struct.unpack("3i", file.read(12))

    # Bytes 12 -- 15: data mode
    mode = struct.unpack("i", file.read(4))[0]

    # Bytes 16 -- 27: (nxstart,nystart,nzstart) (unused)
    file.read(12)

    # Bytes 28 -- 39: (mx,my,mz) (unused)
    file.read(12)

    # Bytes 40 -- 51: (xlen,ylen,zlen) (unused)
    file.read(12)

    # Bytes 52 -- 63: (alpha,beta,gamaa) (unused)
    file.read(12)

    # Bytes 64 -- 75: (mapc,mapr,maps) (unused)
    file.read(12)

    # Bytes 76 -- 87: (amin,amax,amean) data scaling
    dmin, dmax, davg = struct.unpack("3f", file.read(12))

    # Bytes 88 -- 91: (ispg,nsymbt) (unused)
    file.read(4)

    # Bytes 92 -- 95: length of extended header
    ext_header_len = struct.unpack("i", file.read(4))[0]

    # Bytes 96 -- 97: creator id (unused)
    file.read(2)

    # Bytes 98 -- 127: unused extra data
    file.read(30)

    # Bytes 128 -- 131: (numint, numfloats) (unused)
    file.read(4)

    # Bytes 132 -- 135: sub, zfac
    sub, zfac = struct.unpack("2h", file.read(4))

    # Bytes 136 -- 159: (min2, max2, min3, max3, min4, max4)
    min2, max2, min3, max3, min4, max4 = struct.unpack("6f", file.read(24))

    # Bytes 160 -- 161: data ID
    idtype = struct.unpack("h", file.read(2))[0]

    # Bytes 162 -- 163: lens
    lens = struct.unpack("h", file.read(2))[0]

    # Bytes 164 -- 167: (nd1, nd2)
    nd1, nd2 = struct.unpack("2h", file.read(4))

    # Bytes 168 -- 171: (vd1, vd2)
    vd1, vd2 = struct.unpack("2h", file.read(4))

    # Bytes 172 -- 219 (tilts, origin etc) (unused)
    file.read(48)

    # Bytes 220 -- 223: number of additional labels (should be 1)
    n_labels = struct.unpack("i", file.read(4))[0]

    # Bytes 224 -- 1023: additional labels
    labels = [file.read(80).strip(" \x00") for j in xrange(n_labels)]

    # EXTENDED HEADER

    file.seek(1024)

    ab_tilts = []
    xyz_stage_pos = []
    xy_shifts = []
    defoci = []
    exp_times = []
    means = []
    tiltaxes = []
    pixel_sizes = []
    magnifications = []
    voltages = []
    binnings = []
    applied_defoci = []

    for i in xrange(1024):
        ab_tilts.append(struct.unpack("2f", file.read(8)))
        xyz_stage_pos.append(struct.unpack("3f", file.read(12)))
        xy_shifts.append(struct.unpack("2f", file.read(8)))
        defoci.append(struct.unpack("f", file.read(4))[0])
        exp_times.append(struct.unpack("f", file.read(4))[0])
        means.append(struct.unpack("f", file.read(4))[0])
        tiltaxes.append(struct.unpack("f", file.read(4))[0])
        pixel_sizes.append(struct.unpack("f", file.read(4))[0])
        magnifications.append(struct.unpack("f", file.read(4))[0])
        voltages.append(struct.unpack("f", file.read(4))[0])
        binnings.append(struct.unpack("f", file.read(4))[0])
        applied_defoci.append(struct.unpack("f", file.read(4))[0])
        file.read(64)  # read remaining bytes up to 128

    # Assign dictionary values

    header["shape"] = shape
    try:
        header["dtype"] = _mode_types[mode]
    except KeyError:
        print "Unknown data type"
        raise KeyError

    header["minimum value"] = dmin
    header["maximum value"] = dmax
    header["average value"] = davg
    header["extended header length"] = ext_header_len
    header["sub"] = sub
    header["zfac"] = zfac
    header["minima 2--4"] = (min2, min3, min4)
    header["maxima 2--4"] = (max2, max3, max4)
    header["data id type"] = idtype
    header["lens"] = lens
    header["nd"] = (nd1, nd2)
    header["vd"] = (vd1, vd2)
    header["additional labels"] = n_labels
    for j in xrange(n_labels):
        header["label {}".format(j)] = labels[j]

    header["(alpha, beta) tilts"] = ab_tilts
    header["stage positions"] = xyz_stage_pos
    header["image shifts"] = xy_shifts
    header["defoci"] = defoci
    header["exposure times"] = exp_times
    header["image mean values"] = means
    header["tilt axis rotations"] = tiltaxes
    header["pixel sizes"] = pixel_sizes
    header["magnifications"] = magnifications
    header["voltages"] = voltages
    header["binnings"] = binnings
    header["applied defoci"] = applied_defoci

    return header

# ----------------------------------------------------------------------------


def get_data(file, shape=None, dtype=int, offset=129*1024):
    """
    Read the data from an MRC format file and return it as a NumPy array

    Parameters
    ----------
    file: file-like
        where the data is read from

    shape: tuple-like, optional
        Array dimensions. If empty, the whole file is read and stored in a
        flat array.

    dtype: data-type, optional
        how the bytes are to be interpreted

    Returns
    -------
    out: ndarray
        Array object holding the data read from the file
    """

    import numpy as np

    file.seek(offset)

    dt = dtype
    ct = -1 if shape is None else np.prod(shape)

    data = np.fromfile(file, dtype=dt, count=ct)

    if shape is not None:
        data = data.reshape(shape, order="F").squeeze()

    return data

# ----------------------------------------------------------------------------


def fromfile(file):
    """
    Read the contents of an MRC file and store header and data.

    Parameters
    ----------
    file: file-like
        where the data is read from

    Return
    ------
    out1: dictionary
        header information

    out2: ndarray
        data
    """

    header = get_header(file)
    shp = header["shape"]
    dt = header["dtype"]
    xt_len = header["extended header length"]
    data = get_data(file, shape=shp, dtype=dt, offset=1024+xt_len)

    return header, data

# ----------------------------------------------------------------------------


def put_header(file, header):
    """
    Write the header information from a dictionary to an MRC file

    Parameters
    ----------
    file: file-like
        where the data is written

    header: dictionary
        contains the header information

    """

    import struct

    file.seek(0)

    labels = []

    shape = header["shape"]
    mode = _type_modes[header["dtype"]]
    dmin = header["minimum value"]
    dmax = header["maximum value"]
    davg = header["average value"]
    ext_header_len = header["extended header length"]
    sub = header["sub"]
    zfac = header["zfac"]
    min2, min3, min4 = header["minima 2--4"]
    max2, max3, max4 = header["maxima 2--4"]
    idtype = header["data id type"]
    lens = header["lens"]
    nd1, nd2 = header["nd"]
    vd1, vd2 = header["vd"]
    n_labels = header["additional labels"]
    for j in xrange(n_labels):
        labels.append(header["label {}".format(j)])

    ab_tilts = header["(alpha, beta) tilts"]
    xyz_stage_pos = header["stage positions"]
    xy_shifts = header["image shifts"]
    defoci = header["defoci"]
    exp_times = header["exposure times"]
    means = header["image mean values"]
    tiltaxes = header["tilt axis rotations"]
    pixel_sizes = header["pixel sizes"]
    magnifications = header["magnifications"]
    voltages = header["voltages"]
    binnings = header["binnings"]
    applied_defoci = header["applied defoci"]

    # Write the bytes to file

    # STANDARD MRC HEADER

    # Bytes 0 -- 11: (x,y,z) shape
    file.write(struct.pack("3i", shape[0], shape[1], shape[2]))

    # Bytes 12 -- 15: data mode
    file.write(struct.pack("i", mode))

    # Bytes 16 -- 27: unused
    file.write(struct.pack("3i", 0, 0, 0))

    # Bytes 28 -- 39: same as shape
    file.write(struct.pack("3i", shape[0], shape[1], shape[2]))

    # Bytes 40 -- 51: same as shape (but float)
    file.write(struct.pack("3f", shape[0], shape[1], shape[2]))

    # Bytes 52 -- 63: cell angles (unused)
    file.write(struct.pack("3f", 90., 90., 90.))

    # Bytes 64 -- 75: dimension mapping (unused)
    file.write(struct.pack("3i", 1, 2, 3))

    # Bytes 76 -- 87: (min,max,mean) data scaling
    file.write(struct.pack("3f", dmin, dmax, davg))

    # Bytes 88 -- 91: unused
    file.write(struct.pack("2h", 0, 0))

    # Bytes 92 -- 95: length of extended header
    file.write(struct.pack("i", ext_header_len))

    # Bytes 96 -- 97: unused
    file.write(struct.pack("h", 0))

    # Bytes 98 -- 127: unused
    file.write(30*"\x00")

    # Bytes 128 -- 131: number of int/float values in extended header; should
    #                   always be 0, 32
    file.write(struct.pack("2h", 0, 32))

    # Bytes 132 -- 135: sub, zfac
    file.write(struct.pack("2h", sub, zfac))

    # Bytes 136 -- 159: min/max 2,3,4
    file.write(struct.pack("6f", min2, max2, min3, max3, min4, max4))

    # Bytes 160 -- 163: idtype, lens
    file.write(struct.pack("2h", idtype, lens))

    # Bytes 164 -- 167: nd
    file.write(struct.pack("2h", nd1, nd2))

    # Bytes 168 -- 171: vd
    file.write(struct.pack("2h", vd1, vd2))

    # Bytes 172 -- 219: unused
    file.write(struct.pack("12f", 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0.))

    # Bytes 220 -- 223: number of additional labels
    file.write(struct.pack("i", n_labels))

    # Bytes 224 -- 1023: additional labels
    for j in xrange(n_labels):
        label = labels[j]
        label = label + (80-len(label))*"\x00"  # stretch to 80 chars
        file.write(label)

    file.write((10-len(labels))*80*"\x00")  # fill the remaining space

    # EXTENDED HEADER

    for i in xrange(ext_header_len):
        file.write(struct.pack("2f", ab_tilts[i][0], ab_tilts[i][1]))
        file.write(struct.pack("3f", xyz_stage_pos[i][0], xyz_stage_pos[i][1],
                               xyz_stage_pos[i][2]))
        file.write(struct.pack("2f", xy_shifts[i][0], xy_shifts[i][1]))

        file.write(struct.pack("f", defoci[i]))
        file.write(struct.pack("f", exp_times[i]))
        file.write(struct.pack("f", means[i]))
        file.write(struct.pack("f", tiltaxes[i]))
        file.write(struct.pack("f", pixel_sizes[i]))
        file.write(struct.pack("f", magnifications[i]))
        file.write(struct.pack("f", voltages[i]))
        file.write(struct.pack("f", binnings[i]))
        file.write(struct.pack("f", applied_defoci[i]))

        file.write(64*"\x00")

# ----------------------------------------------------------------------------


def put_data(file, array, offset=129*1024):
    """
    Write the data from a NumPy array to an MRC format file

    Parameters
    ----------
    file: file-like
        where the data is written to

    array: ndarray
        holds the data values

    """

    file.seek(offset)
    array.flatten(order="F").tofile(file)

# ----------------------------------------------------------------------------


def tofile(file, header, array):
    """
    Write the contents of a header and a NumPy array to a file

    Parameters
    ----------
    file: file-like
        where the data is written to

    array: ndarray
        contains the data

    header: dictionary
        holds the header information

    """

    xt_len = header["extended header length"]

    put_header(file, header)
    put_data(file, array, offset=1024+xt_len)
    file.close()

# ----------------------------------------------------------------------------


def print_header(h, style="basic"):
    """
    Print the contents of an MRC header

    Parameters
    ----------
    h: dictionary
        contains the header information

    style: string
        specifies which information are printed\n
            "basic"     : compressed output\n
            "extended"  : with extended header\n
            "complete"  : all dictionary entries

    """

    if style == "basic":
        print "Shape (x,y,z)         : ", h["shape"]
        print "Data type             : ", h["dtype"]
        print "Minimum value         : ", h["minimum value"]
        print "Maximum value         : ", h["maximum value"]
        print "Average value         : ", h["average value"]
        print "Additional labels     : ", h["additional labels"]
        for i in xrange(h["additional labels"]):
            print "Label {:>2}              : ".format(i)
            print h["label {}".format(i)]

    elif style == "extended":
        print "Shape (x,y,z)         : ", h["shape"]
        print "Data type             : ", h["dtype"]
        print "Minimum value         : ", h["minimum value"]
        print "Maximum value         : ", h["maximum value"]
        print "Average value         : ", h["average value"]
        print "Additional labels     : ", h["additional labels"]
        for i in xrange(h["additional labels"]):
            print "Label {:>2}              : ".format(i)
            print h["label {}".format(i)]
        nz = h["shape"][2]
        print "Extended header length: ", h["extended header length"], "Bytes"
        print ""
        print """           Tilt angles                        Stage positions\
                          Shifts\n"""
        for i in xrange(nz):
            print """{3:>3}|  ({0[0]:> .2e}, {0[1]:> .2e})\
        ({1[0]:> .2e}, {1[1]:> .2e}, {1[2]:> .2e}) \
        ({2[0]:> .2e}, {2[1]:> .2e})""".format(h["(alpha, beta) tilts"][i],
                                               h["stage positions"][i],
                                               h["image shifts"][i],
                                               i+1)
        print "\n\n"
        print """         Defoci   Exp times     Means    Tiltaxes   Px sizes \
    Magn     Voltages   Binnings  Appl defoci\n"""
        for i in xrange(nz):
            print """{:>3}|   {:> .2e}  {:> .2e}  {:> .2e}  {:> .2e}  \
{:> .2e}  {:> .2e}  {:> .2e}  {:> .2e}  \
{:> .2e}""".format(i+1,
                   h["defoci"][i],
                   h["exposure times"][i],
                   h["image mean values"][i],
                   h["tilt axis rotations"][i],
                   h["pixel sizes"][i],
                   h["magnifications"][i],
                   h["voltages"][i],
                   h["binnings"][i],
                   h["applied defoci"][i])

    elif style == "complete":
        print "Shape (x,y,z)         : ", h["shape"]
        print "Data type             : ", h["dtype"]
        print "Minimum value         : ", h["minimum value"]
        print "Maximum value         : ", h["maximum value"]
        print "Average value         : ", h["average value"]
        print "Sub                   : ", h["sub"]
        print "Zfac                  : ", h["zfac"]
#    zfac = header["zfac"]
#    min2, min3, min4 = header["minima 2--4"]
#    max2, max3, max4 = header["maxima 2--4"]
#    idtype = header["data id type"]
#    lens = header["lens"]
#    nd1, nd2 = header["nd"]
#    vd1, vd2 = header["vd"]

        print "Additional labels     : ", h["additional labels"]
        for i in xrange(h["additional labels"]):
            print "Label {:>2}              : ".format(i), \
                                              h["label {}".format(i)]
        nz = h["shape"][2]
        print "Extended header length: ", h["extended header length"], "Bytes"
        print ""
        print """           Tilt angles                        Stage positions\
                          Shifts\n"""
        for i in xrange(nz):
            print """{3:>3}|  ({0[0]:> .2e}, {0[1]:> .2e})\
        ({1[0]:> .2e}, {1[1]:> .2e}, {1[2]:> .2e}) \
        ({2[0]:> .2e}, {2[1]:> .2e})""".format(\
               h["(alpha, beta) tilts"][i], \
               h["stage positions"][i], \
               h["image shifts"][i], \
               i+1)
        print "\n\n"
        print """         Defoci   Exp times     Means    Tiltaxes   Px sizes \
    Magn     Voltages   Binnings  Appl defoci\n"""
        for i in xrange(nz):
            print """{:>3}|   {:> .2e}  {:> .2e}  {:> .2e}  {:> .2e}  \
{:> .2e}  {:> .2e}  {:> .2e}  {:> .2e}  {:> .2e}""".format(i+1, \
            h["defoci"][i], \
            h["exposure times"][i], \
            h["image mean values"][i], \
            h["tilt axis rotations"][i], \
            h["pixel sizes"][i], \
            h["magnifications"][i], \
            h["voltages"][i], \
            h["binnings"][i], \
            h["applied defoci"][i])

#----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys, matplotlib.pyplot as plt, matplotlib.cm as cm

    try:
        fname = sys.argv[1]
    except IndexError:
        print _usage
        sys.exit(1)

    with open(fname, "rb") as f:
        h, A = fromfile(f, style="all")

    print_header(h, style="all")
    imname = fname.rsplit(".", 1)[0] + ".png"

    if len(A.shape) == 2:
        plt.imsave(imname, A, cmap=cm.gray)
        print "Image " + imname + " saved to current directory."
