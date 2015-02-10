# -*- coding: utf-8 -*-
"""
mrc_basic.py -- I/O for basic MRC files

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

SIZE DATA    NAME	      Description

4    int     nx;	       Number of Columns
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
2    short   nd1;	for idtype = 1, nd1 = axis (1, 2, or 3)
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
4    int     nlabl;  	Number of labels with useful data.
800  char[10][80]    	10 labels of 80 charactors.
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

#----------------------------------------------------------------------------

def get_header(f):
    """
    Read the header information from an MRC file and return it as a
    dictionary.

    Parameters
    ----------
    f: file-like
        where the data is read from

    Returns
    -------
    out: dictionary
        the header information
    """

    import struct

    h = {}

    # Read the data from the file

    # Bytes 1 -- 12: (x,y,z) shape
    f.seek(0)
    shape = struct.unpack("3i", f.read(12))

    # Bytes 13 -- 16: data mode
    f.seek(12)
    mode = struct.unpack("i", f.read(4))[0]

    # Bytes 41 -- 52: (x,y,z) cell size
    f.seek(40)
    cell_size = struct.unpack("3f", f.read(12))

    # Bytes 77 -- 88: (min,max,avg) data scaling
    f.seek(76)
    dmin, dmax, davg = struct.unpack("3f", f.read(12))

    # Bytes 93 -- 96: length of extended header
    f.seek(92)
    ext_hlen = struct.unpack("i", f.read(4))[0]

    # Bytes 197 -- 208: image origin
    f.seek(196)
    origin = struct.unpack("3f", f.read(12))


    # Assign dictionary values

    h["shape"] = shape
    try:
        h["dtype"] = _mode_types[mode]
    except KeyError:
        print "Unknown data type"
        raise KeyError

    h["cell size"] = cell_size
    h["minimum value"] = dmin
    h["maximum value"] = dmax
    h["average value"] = davg
    h["extended header length"] = ext_hlen
    h["image origin"] = origin

    return h

#----------------------------------------------------------------------------

def get_data(f, shape=None, dtype=int, offset_bytes=0):
    """
    Read the data from an MRC format file and return it as a NumPy array

    Parameters
    ----------
    f: file-like
        where the data is read from

    shape: tuple-like, optional
        Array dimensions. If empty, the whole file is read and stored in a
        flat array.

    dtype: data-type, optional
        how the bytes are to be interpreted
    
    offset_bytes: int
        additional (to 1024) offset of data in the file

    Returns
    -------
    out: ndarray
        Array object holding the data read from the file
    """

    import numpy as np

    
    f.seek(1024 + offset_bytes)
    d = np.fromfile(f, dtype)
    if shape is not None:
        d = d.reshape(shape, order="F").squeeze()

    return d

#----------------------------------------------------------------------------

def fromfile(f):
    """
    Read the contents of an MRC file and store header and data.

    Parameters
    ----------
    f: file-like
        where the data is read from

    Return
    ------
    out1: dictionary
        header information

    out2: ndarray
        data
    """

    h = get_header(f)
    shp = h["shape"]
    dt = h["dtype"]
    ext_hlen = h["extended header length"]
    d = get_data(f, shape=shp, dtype=dt, offset_bytes=ext_hlen)

    return h, d

#----------------------------------------------------------------------------

def put_header(f, h):
    """
    Write the header information from a dictionary to an MRC file

    Parameters
    ----------
    f: file-like
        where the data is written

    h: dictionary
        contains the header information

    """

    import struct

    shape = h["shape"]
    mode = _type_modes[h["dtype"]]
    
    try:
        cell_size = h["cell size"]
    except KeyError:
        cell_size = (1.0, 1.0, 1.0)
        
    dmin = h["minimum value"]
    dmax = h["maximum value"]
    davg = h["average value"]
    
    try:
        origin = h["image origin"]
    except KeyError:
        origin = (0.0, 0.0, 0.0)

    # Write the bytes to file

    # Bytes 1 -- 12: (x,y,z) shape
    f.seek(0)
    f.write(struct.pack("3i", shape[0], shape[1], shape[2]))

    # Bytes 13 -- 16: data mode
    f.seek(12)
    f.write(struct.pack("i", mode))

    # Bytes 41 -- 52: (x,y,z) cell size
    f.seek(40)
    f.write(struct.pack("3f", cell_size[0], cell_size[1], cell_size[2]))

    # Bytes 77 -- 88: (min,max,avg) data scaling
    f.seek(76)
    f.write(struct.pack("3f", dmin, dmax, davg))

    # Bytes 197 -- 208: image origin
    f.seek(196)
    f.write(struct.pack("3f", origin[0], origin[1], origin[2]))

    # Bytes 213 -- 216: first byte determines endianness
    f.seek(212)
    f.write(struct.pack("b", 68))


#----------------------------------------------------------------------------

def put_data(f, arr):
    """
    Write the data from a NumPy array to an MRC format file

    Parameters
    ----------
    f: file-like
        where the data is written to

    arr: ndarray
        holds the data values

    """

    f.seek(1024)
    arr.flatten(order="F").tofile(f)

#----------------------------------------------------------------------------

def tofile(f, h, arr):
    """
    Write the contents of a header and a NumPy array to a file

    Parameters
    ----------
    f: file-like
        where the data is written to

    arr: ndarray
        contains the data

    h: dictionary
        holds the header information

    """

    put_header(f, h)
    put_data(f, arr)
    f.close()

#----------------------------------------------------------------------------

def print_header(h):
    """
    Print the contents of an MRC header

    Parameters
    ----------
    h: dictionary
        contains the header information

 
    """
    print "Shape (x,y,z): ", h["shape"]
    print "Cell size    : ", h["cell size"]
    print "Data type    : ", h["dtype"]
    print "Minimum value: ", h["minimum value"]
    print "Maximum value: ", h["maximum value"]
    print "Average value: ", h["average value"]
    print "Image origin : ", h["image origin"]

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

    print_header(h)
    imname = fname.rsplit(".", 1)[0] + ".png"

    if len(A.shape) == 2:
        plt.imsave(imname, A, cmap=cm.gray)
        print "Image " + imname + " saved to current directory."
