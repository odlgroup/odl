# -*- coding: utf-8 -*-
"""
mrc_imod.py -- I/O for MRC files as used by IMOD

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
The MRC file format used by IMOD.

The MRC header. length 1024 bytes

SIZE DATA    NAME          Description

4    int     nx;           Number of Columns
4    int     ny;        Number of Rows
4    int     nz;        Number of Sections.

4    int     mode;      Types of pixel in image.  Values used by IMOD:
                     0 = unsigned bytes,
                        1 = signed short integers (16 bits),
                        2 = float,
                        3 = short * 2, (used for complex data)
                        4 = float * 2, (used for complex data)
                        6 = unsigned 16-bit integers (non-standard)
                        16 = unsigned char * 3 (for rgb data, non-standard)

4    int     nxstart;     Starting point of sub image (not used in IMOD)
4    int     nystart;
4    int     nzstart;

4    int     mx;         Grid size in X, Y, and Z
4    int     my;
4    int     mz;

4    float   xlen;       Cell size; pixel spacing = xlen/mx
4    float   ylen;
4    float   zlen;

4    float   alpha;      cell angles - ignored by IMOD
4    float   beta;
4    float   gamma;

                        Ignored by IMOD.
4    int     mapc;       map column  1=x,2=y,3=z.
4    int     mapr;       map row     1=x,2=y,3=z.
4    int     maps;       map section 1=x,2=y,3=z.

                         These need to be set for proper scaling of data
4    float   amin;       Minimum pixel value.
4    float   amax;       Maximum pixel value.
4    float   amean;      Mean pixel value.

2    short   ispg;       space group number (0 for image)
2    short   nsymbt;     bytes for symmetry operators
4    int     next;       number of bytes in extended header
2    short   creatid;    Creator ID
30   ---     extra data (not used)

                         These two values specify the structure of data in the
                         extended header; their meaning depend on whether the
                         extended header has the Agard format, a series of
                         4-byte integers then real numbers, or has data
                         produced by SerialEM, a series of short integers.
                         SerialEM stores a float as two shorts, s1 and s2, by:
                           value = (sign of s1)*(|s1|*256 + (|s2| modulo 256))
                              * 2**((sign of s2) * (|s2|/256))
2    short   nint;       Number of integers per section (Agard format) or
                         number of bytes per section (SerialEM format)
2    short   nreal;      Number of reals per section (Agard format) or
                         flags for which types of short data (SerialEM format):
                         1 = tilt angle * 100  (2 bytes)
                         2 = piece coordinates for montage  (6 bytes)
                         4 = Stage position * 25    (4 bytes)
                         8 = Magnification / 100 (2 bytes)
                         16 = Intensity * 25000  (2 bytes)
                         32 = Exposure dose in e-/A2, a float in 4 bytes
                         128, 512: Reserved for 4-byte items
                         64, 256, 1024: Reserved for 2-byte items
                         If the number of bytes implied by these flags does
                         not add up to the value in nint, then nint and nreal
                         are interpreted as ints and reals per section

28   ---     extra data (not used)

                      Explanation of type of data.
2    short   idtype;  ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
2    short   lens;
2    short   nd1;    for idtype = 1, nd1 = axis (1, 2, or 3)
2    short   nd2;
2    short   vd1;                       vd1 = 100. * tilt increment
2    short   vd2;                       vd2 = 100. * starting angle

                Current angles are used to rotate a model to match a
                        new rotated image.  The three values in each set are
                        rotations about X, Y, and Z axes, applied in the order
                        Z, Y, X.
24   float   tiltangles[6];  0,1,2 = original:  3,4,5 = current

                       The image origin is the location of the origin of the
                       coordinate system relative to the first pixel in the
                       file.  It is in pixel spacing units rather than in
                       pixels.  If an original volume has an origin of 0, a
                       subvolume should have negative origin values.
OLD-STYLE MRC HEADER - IMOD 2.6.19 and below:
2    short   nwave;     # of wavelengths and values
2    short   wave1;
2    short   wave2;
2    short   wave3;
2    short   wave4;
2    short   wave5;

4    float   zorg;      Origin of image.
4    float   xorg;
4    float   yorg;

NEW-STYLE MRC image2000 HEADER - IMOD 2.6.20 and above:
4    float   xorg;      Origin of image
4    float   yorg;
4    float   zorg;

4    char    cmap;      Contains "MAP "
4    char    stamp;     First byte has 17 for big- or 68 for little-endian
4    float   rms;       RMS deviation of densities from mean density

ALL HEADERS:
4    int     nlabl;      Number of labels with useful data.
800  char[10][80]        10 labels of 80 charactors.
------------------------------------------------------------------------

Total size of header is 1024 bytes plus the size of the extended header.

Image data follows with the origin in the lower left corner,
looking down on the volume.

The size of the image is nx * ny * nz * (mode data size).
"""

_usage = """
Usage: %s <testfile>.mrc

Reads the file, displays the header information and writes a plot of
the data into <testfile>.png
"""

_mode_types = {0: "ubyte",
               1: "short",
               2: "float32",
               3: "short*2",
               4: "csingle",
               6: "ushort",
               16: "uint8*3"
               }
_type_modes = {}    # reversed
for key, val in _mode_types.items():
    _type_modes[val] = key

_ids = ["mono", "tilt", "tilts", "lina", "lins"]

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

    # Bytes 1 -- 12: (x,y,z) shape
    shape = struct.unpack("3i", file.read(12))

    # Bytes 13 -- 16: data mode
    mode = struct.unpack("i", file.read(4))[0]

    # Bytes 17 -- 28: (nxo, nyo, nzo) offsets (subimage)
    offset_sub = struct.unpack("3i", file.read(12))

    # Bytes 29 -- 40: (x,y,z) grid shape
    grid_shape = struct.unpack("3i", file.read(12))

    # Bytes 41 -- 52: (x,y,z) cell size
    cell_size = struct.unpack("3f", file.read(12))

    # Bytes 53 -- 64: (alpha,beta,gamaa) cell angles
    cell_angles = struct.unpack("3f", file.read(12))

    # Bytes 65 -- 76: (x,y,z) dimension mapping
    dim_map = struct.unpack("3i", file.read(12))

    # Bytes 77 -- 88: (min,max,avg) data scaling
    dmin, dmax, davg = struct.unpack("3f", file.read(12))

    # Bytes 89 -- 92: space group numer (0 = image), bytes for symmetry op's
    sg_num, bsymm = struct.unpack("2h", file.read(4))

    # Bytes 93 -- 96: length of extended header
    ext_header_len = struct.unpack("i", file.read(4))[0]

    # Bytes 97 -- 98: creator id
    creator_id = struct.unpack("h", file.read(2))[0]

    # Bytes 99 -- 128: unused extra data
    file.read(30)

    # Bytes 129 -- 132: bytes/section, type flag (SerialEM)
    bytes_per_section, type_flag = struct.unpack("2h", file.read(4))

    # Bytes 133 -- 160: unused extra data
    file.read(28)

    # Bytes 161 -- 162: data ID ( 0=mono, 1=tilt, 2=tilts, 3=lina, 4=lins)
    i = struct.unpack("h", file.read(2))[0]

    # Bytes 163 -- 164: lens
    lens = struct.unpack("h", file.read(2))[0]

    # Bytes 165 -- 168: nd1 (=axis if data id = 1)
    nd1, nd2 = struct.unpack("2h", file.read(4))

    # Bytes 169 -- 172: tilt increment and start
    tilt_inc, tilt_start = struct.unpack("2h", file.read(4))

    # Bytes 173 -- 196 rotation angles (original/current)
    orig_angles = struct.unpack("3f", file.read(12))
    cur_angles = struct.unpack("3f", file.read(12))

    # Bytes 197 -- 208: image origin
    origin = struct.unpack("3f", file.read(12))

    # Bytes 209 -- 212: cmap
    cmap = file.read(4).strip("\x00 ")

    # Bytes 213 -- 216: first byte determines endianness
    s = file.read(4)
    s0 = struct.unpack("b", s[0])[0]
    endian = "little" if s0 == 68 else "big"

    # Bytes 217 -- 220: root mean square deviation
    rms_deviation = struct.unpack("f", file.read(4))[0]

    # Bytes 221 -- 224: number of additional labels
    n_labels = struct.unpack("i", file.read(4))[0]

    # Bytes 225 -- 1024: additional labels
    labels = [file.read(80).strip(" \x00") for j in xrange(n_labels)]

    # Assign dictionary values

    header["shape"] = shape
    try:
        header["dtype"] = _mode_types[mode]
    except KeyError:
        print "Unknown data type"
        raise KeyError

    header["subimage offset"] = offset_sub
    header["grid shape"] = grid_shape
    header["cell size"] = cell_size
    header["cell angles"] = cell_angles
    header["dimensions mapping"] = dim_map
    header["minimum value"] = dmin
    header["maximum value"] = dmax
    header["average value"] = davg
    header["space group number"] = sg_num
    header["symmetry operator bytes"] = bsymm
    header["extended header length"] = ext_header_len
    header["creator id"] = creator_id
    header["bytes/section"] = bytes_per_section
    header["type flag"] = type_flag
    try:
        header["data id"] = _ids[i]
    except IndexError:
        print "Unknown data ID"
        raise IndexError

    header["lens"] = lens
    if header["data id"] == "tilt":
        header["axis"] = nd1 - 1

    header["tilt increment"] = tilt_inc/100.0
    header["starting angle"] = tilt_start/100.0
    header["original angles"] = orig_angles
    header["current angles"] = cur_angles
    header["image origin"] = origin
    header["cmap"] = cmap
    header["endianness"] = endian
    header["rms deviation"] = rms_deviation
    header["additional labels"] = n_labels
    for j in xrange(n_labels):
        header["label {}".format(j+1)] = labels[j]

    return header

# ----------------------------------------------------------------------------


def get_data(file, shape=None, dtype=int, extlen=0):
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

    file.seek(1024 + extlen)
    data = np.fromfile(file, dtype)
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
    ext_h_len = header["extended header length"]
    data = get_data(file, shape=shp, dtype=dt, extlen=ext_h_len)

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
    offset_sub = header["subimage offset"]
    grid_shape = header["grid shape"]
    cell_size = header["cell size"]
    cell_angles = header["cell angles"]
    dim_map = header["dimensions mapping"]
    dmin = header["minimum value"]
    dmax = header["maximum value"]
    davg = header["average value"]
    sg_num = header["space group number"]
    bsymm = header["symmetry operator bytes"]
    ext_header_len = 0  # drop this
    creator_id = header["creator id"]
    bytes_per_section = header["bytes/section"]
    type_flag = header["type flag"]
    id = _ids.index(header["data id"])
    lens = header["lens"]

    if header["data id"] == "tilt":
        nd1 = header["axis"] + 1
    else:
        nd1 = 0

    nd2 = 0

    tilt_inc = int(100.0 * header["tilt increment"])
    tilt_start = int(100.0 * header["starting angle"])
    orig_angles = header["original angles"]
    cur_angles = header["current angles"]
    origin = header["image origin"]
    cmap = header["cmap"]
    rms_deviation = header["rms deviation"]
    n_labels = header["additional labels"]
    for j in xrange(n_labels):
        labels.append(header["label {}".format(j+1)])

    # Write the bytes to file

    # Bytes 1 -- 12: (x,y,z) shape
    file.write(struct.pack("3i", shape[0], shape[1], shape[2]))

    # Bytes 13 -- 16: data mode
    file.write(struct.pack("i", mode))

    # Bytes 17 -- 28: (nxo, nyo, nzo) offsets (subimage)
    file.write(struct.pack("3i", offset_sub[0], offset_sub[1], offset_sub[2]))

    # Bytes 29 -- 40: (x,y,z) grid shape
    file.write(struct.pack("3i", grid_shape[0], grid_shape[1], grid_shape[2]))

    # Bytes 41 -- 52: (x,y,z) cell size
    file.write(struct.pack("3f", cell_size[0], cell_size[1], cell_size[2]))

    # Bytes 43 -- 64: (alpha,beta,gamaa) cell angles
    file.write(struct.pack("3f", cell_angles[0], cell_angles[1],
                           cell_angles[2]))

    # Bytes 65 -- 76: (x,y,z) dimension mapping
    file.write(struct.pack("3i", dim_map[0], dim_map[1], dim_map[2]))

    # Bytes 77 -- 88: (min,max,avg) data scaling
    file.write(struct.pack("3f", dmin, dmax, davg))

    # Bytes 89 -- 92: image type (?), space group numer (?)
    file.write(struct.pack("2h", sg_num, bsymm))

    # Bytes 93 -- 96: length of extended header
    file.write(struct.pack("i", ext_header_len))

    # Bytes 97 -- 98: creator id
    file.write(struct.pack("h", creator_id))

    # Bytes 99 -- 128: unused extra data
    file.write(30*"\x00")

    # Bytes 129 -- 132: bytes/section, type flag (SerialEM)
    file.write(struct.pack("2h", bytes_per_section, type_flag))

    # Bytes 133 -- 160: unused extra data
    file.write(28*"\x00")

    # Bytes 161 -- 162: data ID ( 0=mono, 1=tilt, 2=tilts, 3=lina, 4=lins)
    file.write(struct.pack("h", id))

    # Bytes 163 -- 164: lens
    file.write(struct.pack("h", lens))

    # Bytes 165 -- 168: nd1 (=axis if data id = 1)
    file.write(struct.pack("2h", nd1, nd2))

    # Bytes 169 -- 172: tilt increment and start
    file.write(struct.pack("2h", tilt_inc, tilt_start))

    # Bytes 173 -- 196 rotation angles (original/current)
    file.write(struct.pack("3f", orig_angles[0], orig_angles[1],
               orig_angles[2]))
    file.write(struct.pack("3f", cur_angles[0], cur_angles[1],
               cur_angles[2]))

    # Bytes 197 -- 208: image origin
    file.write(struct.pack("3f", origin[0], origin[1], origin[2]))

    # Bytes 209 -- 212: cmap
    cmap = cmap + (4-len(cmap))*"\x00"  # Make cmap 4 characters long
    file.write(cmap)

    # Bytes 213 -- 216: first byte determines endianness
    file.write(struct.pack("b", 68))
    file.write(3*"\x00")

    # Bytes 217 -- 220: root mean square deviation
    file.write(struct.pack("f", rms_deviation))

    # Bytes 221 -- 224: number of additional labels
    file.write(struct.pack("i", n_labels))

    # Bytes 225 -- 1024: additional labels
    for j in xrange(n_labels):
        label = labels[j]
        label = label + (80-len(label))*"\x00"  # stretch to 80 chars
        file.write(label)

    file.write((10-len(labels))*80*"\x00")  # fill the remaining space

# ----------------------------------------------------------------------------


def put_data(file, array):
    """
    Write the data from a NumPy array to an MRC format file

    Parameters
    ----------
    file: file-like
        where the data is written to

    array: ndarray
        holds the data values

    """

    file.seek(1024)
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

    put_header(file, header)
    put_data(file, array)
    file.close()

# ----------------------------------------------------------------------------


def print_header(h, style=None):
    """
    Print the contents of an MRC header

    Parameters
    ----------
    h: dictionary
        contains the header information

    style: string
        specifies which information are printed\n
            None   : only basic parameters\n
            "imod" : additional parameters used by IMOD\n
            "all"  : all parameters

    """
    if style is None:
        print "Shape (x,y,z): ", h["shape"]
        print "Data type    : ", h["dtype"]
        print "Minimum value: ", h["minimum value"]
        print "Maximum value: ", h["maximum value"]
        print "Average value: ", h["average value"]
        print "Data ID      : ", h["data id"]
        print "Endianness   : ", h["endianness"]
    elif style == "imod":
        print "Shape (x,y,z)  : ", h["shape"]
        print "Data type      : ", h["dtype"]
        print "Grid shape     : ", h["grid shape"]
        print "Cell size      : ", h["cell size"]
        print "Minimum value  : ", h["minimum value"]
        print "Maximum value  : ", h["maximum value"]
        print "Average value  : ", h["average value"]
        print "Creator ID     : ", h["creator id"]
        print "Bytes/section  : ", h["bytes/section"]
        print "Type flag      : ", h["type flag"]
        print "Data ID        : ", h["data id"]
        print "Lens           : ", h["lens"]
        print "Starting angle : ", h["starting angle"]
        print "Tilt increment : ", h["tilt increment"]
        print "Original angles: ", h["original angles"]
        print "Current angles : ", h["current angles"]
        print "Image origin   : ", h["image origin"]
        print "Endianness     : ", h["endianness"]
        print "RMS deviation  : ", h["rms deviation"]
    elif style == "all":
        print "Shape (x,y,z)         : ", h["shape"]
        print "Data type             : ", h["dtype"]
        print "Offset                : ", h["subimage offset"]
        print "Grid shape            : ", h["grid shape"]
        print "Cell size             : ", h["cell size"]
        print "Cell angles           : ", h["cell angles"]
        print "Dimension mapping     : ", h["dimensions mapping"]
        print "Minimum value         : ", h["minimum value"]
        print "Maximum value         : ", h["maximum value"]
        print "Average value         : ", h["average value"]
        print "Space group number    : ", h["space group number"]
        print "Symmetry op bytes     : ", h["symmetry operator bytes"]
        print "Extended header length: ", h["extended header length"]
        print "Creator ID            : ", h["creator id"]
        print "Bytes/section         : ", h["bytes/section"]
        print "Type flag             : ", h["type flag"]
        print "Data ID               : ", h["data id"]
        print "Lens                  : ", h["lens"]
        print "Starting angle        : ", h["starting angle"]
        print "Tilt increment        : ", h["tilt increment"]
        print "Original angles       : ", h["original angles"]
        print "Current angles        : ", h["current angles"]
        print "Image origin          : ", h["image origin"]
        print "Cmap                  : ", h["cmap"]
        print "Endianness            : ", h["endianness"]
        print "RMS deviation         : ", h["rms deviation"]
        print "Additional labels     : ", h["additional labels"]
        for i in xrange(h["additional labels"]):
            print "Label {:>2}              : ".format(i+1), \
                h["label {}".format(i+1)]

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from matplotlib import cm

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
