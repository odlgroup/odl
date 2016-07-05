# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:02:55 2016

@author: hkohr
"""
from __future__ import division, print_function
from builtins import range

import numpy as np
import odl
import os
import scipy as sp


tifdir = '/data/hkohr/Phase Contrast/AlSiC_nodeco_B1/tif'
os.chdir(tifdir)
tifdir = os.getcwd()
os.chdir(os.pardir)
basename = os.getcwd()
samplename = os.path.basename(basename)

# --- Get metadata from log file ---
os.chdir(tifdir)
filename_log = os.path.join(tifdir, samplename + '.log')

#defocus = 0.030
#wavelength = 20 / 1.2e-10
#pixelsize = 1e-6

logdict = {}
with open(filename_log, "r") as f:
    for line in f:
        line_list = line.split(':')[:2]
        if len(line_list) == 2:
            logkey, logval = line_list
            logdict[logkey.strip()] = logval.strip()

energy_kev = float(logdict['Beam energy [keV]'])
defocus_um = float(logdict['Sample Out  [um]'])
x_roi = logdict['X-ROI']
y_roi = logdict['Y-ROI']
nproj = int(logdict['Number of projections'])
ndark = int(logdict['Number of darks'])
nflat = int(logdict['Number of flats'])
pix_size_um = float(logdict['Actual pixel size [um]'])
rot_center_pix = float(logdict['Original rotation center'])
# TODO: angular step? Or use 180 / nproj? scaling parameters? Center shift?

print('=== Log file parameters ===')
print('Beam energy [keV]: ', energy_kev)
print('Sample Out  [um]: ', defocus_um)
print('X-ROI: ', x_roi)
print('Y-ROI: ', y_roi)
print('Number of projections: ', nproj)
print('Number of darks: ', ndark)
print('Number of flats: ', nflat)
print('Actual pixel size [um]: ', pix_size_um)
print('Rotation center (x coord) [pixels]: ', rot_center_pix)
print('')


# --- Calculate parameters from log file metadata ---

# Wave length in nm is equal to h*c / (E * 1000), where E = energy_kv and
# h*c = 1239.84 [eV]
wavelen_nm = 1239.84 / (energy_kev * 1000)
wavenum = 2 * np.pi * 1e9 / wavelen_nm  # [1/m]
defocus = defocus_um * 1e-6
pix_size = pix_size_um * 1e-6
xstart, xstop = x_roi.split('-')
nx = int(xstop) - int(xstart) + 1
ystart, ystop = y_roi.split('-')
ny = int(ystop) - int(ystart) + 1

# Read one image first, determine the data type and check size
first_proj_idx = ndark + nflat
im_proj_name = samplename + '{:04}.tif'.format(first_proj_idx + 1)  # one-based
first_image = sp.misc.imread(im_proj_name).T
if first_image.shape != (nx, ny):
    raise ValueError('inconsistent data: first image has shape {}, but the '
                     'log file says (nx, ny) = {}'
                     ''.format(first_image.shape, (nx, ny)))
dtype = first_image.dtype

# Print out some stuff before going on
print('=== Calculated parameters ===')
print('X-ray wave length [nm]: ', wavelen_nm)
print('X-ray wave number [1/m]: ', wavenum)
print('Defocus [m]: ', defocus)
print('Detector pixel size [m]: ', pix_size)
print('Detector shape (nx, ny): ', (nx, ny))
print('Image data type: ', dtype)
print('')


# --- Read dark and flat images ---

def fill_stack(stack, file_fmt_str, indices):
    """Fill a stack with data from files according to format and indices.

    The file names are generated according to::

        for index in indices:
            file_name = file_fmt_str.format(index)
            ...

    The stacking is performed along the first axis of the ``stack`` array.
    """
    for i, idx in enumerate(indices):
        stack[i, ...] = np.rot90(sp.misc.imread(file_fmt_str.format(idx)), -1)


dark_stack = np.empty((ndark, nx, ny), dtype=dtype)
fill_stack(dark_stack, samplename + '{:04}.tif', range(1, ndark + 1))
dark_avg = np.mean(dark_stack, axis=0, dtype='float32')

print('Imported {} dark images, total size in bytes: {}.'
      ''.format(ndark, dark_stack.size * dark_stack.itemsize))
dark_stack = None  # Release memory


flat_stack = np.empty((nflat, nx, ny), dtype=dtype)
fill_stack(flat_stack, samplename + '{:04}.tif',
           range(ndark + 1, ndark + nflat + 1))
flat_avg = np.mean(flat_stack, axis=0, dtype='float32')

print('Imported {} flat images, total size in bytes: {}.'
      ''.format(nflat, flat_stack.size * flat_stack.itemsize))
flat_stack = None  # Release memory


# --- Read data into a large stack ---

proj_start_idx = ndark + nflat + 1
proj_idx_step = 10
proj_indices = range(proj_start_idx, proj_start_idx + nproj, proj_idx_step)
assert proj_indices[0] >= proj_start_idx
assert proj_indices[-1] < proj_start_idx + nproj

# Make a volume for the projections and insert the projection images along
# the first axis.

data_stack = np.empty((len(proj_indices), nx, ny), dtype='float32')

flat_stack = np.empty((nflat, nx, ny), dtype=dtype)
fill_stack(data_stack, samplename + '{:04}.tif', proj_indices)

print('Imported {} projection images, total size in bytes: {}.'
      ''.format(len(proj_indices), data_stack.size * data_stack.itemsize))


# --- Pre-process the data ---

def preprocess_data_simple(data_stack, flat_avg, dark_avg):
    """Perform a simple data preprocessing using flat and dark images.

    For each projection image in the stack, we compute::

        proj_image = (proj_image - dark_avg) / (flat_avg - dark_avg)

    The ``flat_avg`` and ``dark_avg`` are assumed to be averages of
    projection data without object, for the flats with illumination,
    for the darks without.
    """
    data_stack -= dark_avg[None, ...]
    data_stack /= (flat_avg - dark_avg)[None, ...]


preprocess_data_simple(data_stack, flat_avg, dark_avg)


# --- Define ODL quantities ---

# Angles
angle_incr = np.pi / nproj
angle_extent = np.array(((proj_indices[0] - proj_start_idx) * angle_incr,
                         (proj_indices[-1] - proj_start_idx) * angle_incr))
angle_part = odl.uniform_partition(angle_extent[0], angle_extent[1],
                                   len(proj_indices), nodes_on_bdry=True)

# Detector
det_extent = np.array((nx * pix_size, ny * pix_size))
det_part = odl.uniform_partition(-det_extent / 2, det_extent / 2, (nx, ny))

# Parallel 3D geometry (z axis rotation)
geometry = odl.tomo.Parallel3dAxisGeometry(angle_part, det_part)

# Reconstruction space
# Choose x and y sizes according to the horizontal detector size (* sqrt(2)
# to cover all of it), and the z size equal to the vertical detector size.
vol_size = 1.5 * max(det_extent)
vol_extent = np.array([vol_size] * 3)
scal_reco_space = odl.uniform_discr(-vol_extent / 2, vol_extent / 2,
                                    (300,) * 3, dtype='float32')

scal_ray_trafo = odl.tomo.RayTransform(scal_reco_space, geometry,
                                       impl='astra_cuda')

# Data space
data_space = scal_ray_trafo.range
data = data_space.element(data_stack)


# --- Filtered back-projection ---
ft = odl.trafos.FourierTransform(scal_ray_trafo.range, axes=(1, 2),
                                 impl='pyfftw')
filter_relative_gamma = 0.01
filter_gamma = filter_relative_gamma * ft.range.partition.extent()[1]


def ram_lak_filter(x, **kwargs):
    gamma = kwargs.pop('gamma', float('inf'))
    return np.where(abs(x[1]) < gamma, abs(x[1]), 0)


filter_func = ft.range.element(ram_lak_filter, gamma=filter_gamma)

filter_op = ft.inverse * filter_func * ft
fbp_op = scal_ray_trafo.adjoint * filter_op

fbp_reco = fbp_op(data)
fig = fbp_reco.show(indices=np.s_[150, :, :])
