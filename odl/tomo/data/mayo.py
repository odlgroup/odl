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

"""Helpers for reading the MAYO clinic data format

Described in the article

Implementation of an Open Data Format for CT Projection Data
Chen et. al. 2014
."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import dicom
from tqdm import tqdm
import os

from odl.discr import uniform_partition
from odl.tomo.operators import RayTransform
from odl.tomo.geometry import HelicalConeFlatGeometry

__all__ = ('mayo_projector_from_folder',)


def read_mayo_projections(folder, nr_start=1, nr_end=-1):
    """Read mayo projections from a folder."""
    projections = []
    datasets = []

    # Get the relevant file names
    file_names = [f for f in os.listdir(folder) if f.endswith(".dcm")]
    file_names = file_names[nr_start: nr_end]

    for file_name in tqdm(file_names):
        # read the file
        dataset = dicom.read_file(folder + '/' + file_name)

        # Get some required data
        rows = dataset.NumberofDetectorRows
        cols = dataset.NumberofDetectorColumns
        rescale_intercept = dataset.RescaleIntercept
        rescale_slope = dataset.RescaleSlope

        # Load the array as bytes
        data_array = np.fromstring(dataset.PixelData, 'H')
        data_array = data_array.reshape([rows, cols], order='F').T

        # Rescale array
        rescaled_array = rescale_intercept + data_array * rescale_slope

        # Store results
        projections += [rescaled_array]
        datasets += [dataset]

    return datasets, projections


def mayo_projector_from_folder(reco_space, folder, nr_start=1, nr_end=-1):
    datasets, projections = read_mayo_projections(folder, nr_start, nr_end)

    data_array = np.empty((len(projections),) + projections[0].shape,
                          dtype='float32')

    # Move data to a big array, change order
    for i, proj in enumerate(projections):
        data_array[i] = proj[::-1, ::-1]

    # Get the angles
    # TODO: handle not uniformly spaced
    angles = [d.DetectorFocalCenterAngularPosition for d in datasets]
    angles = -np.unwrap(angles) - np.pi / 2  # different defintion of angles

    # Make a parallel beam geometry with flat detector
    angle_partition = uniform_partition(angles.min(), angles.max(),
                                        angles.size, nodes_on_bdry=True)

    # Set minimum and maximum point
    shape = np.array([datasets[0].NumberofDetectorColumns,
                      datasets[0].NumberofDetectorRows])
    pixel_size = np.array([datasets[0].DetectorElementTransverseSpacing,
                           datasets[0].DetectorElementAxialSpacing])

    maxp = (np.array(datasets[0].DetectorCentralElement) - 0.5) * pixel_size
    minp = maxp - shape * pixel_size

    # Create partition for detector
    detector_partition = uniform_partition(minp, maxp, shape)

    # Select geometry parameters
    src_radius = datasets[0].DetectorFocalCenterRadialDistance
    det_radius = (datasets[0].ConstantRadialDistance -
                  datasets[0].DetectorFocalCenterRadialDistance)

    # Convert pitch and offset to odl defintions
    pitch = (pixel_size[1] * shape[1] * datasets[0].SpiralPitchFactor *
             src_radius / (src_radius + det_radius))
    pitch_offset = (datasets[0].DetectorFocalCenterAxialPosition -
                    angles[0] / (2 * np.pi) * pitch)

    # Get flying focal spot data
    offset_axial = np.array([d.SourceAxialPositionShift for d in datasets])
    offset_angular = np.array([d.SourceAngularPositionShift for d in datasets])
    offset_radial = np.array([d.SourceRadialDistanceShift for d in datasets])

    angles_offset = angles - offset_angular
    src_rad_offset = src_radius + offset_radial
    offset_x = (np.cos(angles_offset) * (-src_rad_offset) -
                np.cos(angles) * (-src_radius))
    offset_y = (np.sin(angles_offset) * (-src_rad_offset) -
                np.sin(angles) * (-src_radius))
    offset_z = offset_axial

    source_offsets = np.array([offset_x, offset_y, offset_z]).T

    # Assemble geometry
    geometry = HelicalConeFlatGeometry(angle_partition,
                                       detector_partition,
                                       src_radius=src_radius,
                                       det_radius=det_radius,
                                       pitch=pitch,
                                       pitch_offset=pitch_offset,
                                       source_offsets=source_offsets)

    # ray transform aka forward projection. We use ASTRA CUDA backend.
    ray_trafo = RayTransform(reco_space, geometry, impl='astra_cuda',
                             interp='linear')

    # convert coordinates
    theta, up, vp = ray_trafo.range.grid.meshgrid
    d = src_radius + det_radius
    u = d * np.arctan(up / d)
    v = d / np.sqrt(d**2 + up**2) * vp

    # Calculate projection data in rectangular coordinates
    proj_data_cylinder = ray_trafo.range.element(data_array)
    interpolated_values = proj_data_cylinder.interpolation((theta, u, v))
    proj_data = ray_trafo.range.element(interpolated_values)

    return ray_trafo, proj_data


if __name__ == '__main__':
    import odl
    import matplotlib.pyplot as plt

    folder = 'E:/Data/MayoClinic data/Training Cases/L067/full_DICOM-CT-PD'
    nr_start = 14000
    nr_end = nr_start + 4000

    # Discrete reconstruction space: discretized functions on the cube
    # [-20, 20]^3 with 300 samples per dimension.
    reco_space = odl.uniform_discr(
        min_pt=[-170, -170, 200], max_pt=[170, 170, 242.5],
        shape=[512, 512, 64], dtype='float32')

    ray_trafo, proj_data = mayo_projector_from_folder(reco_space, folder,
                                                      nr_start, nr_end)

    if True:
        # Test FBP reconstruction
        print('Performing FBP')

        # Fourier transform in detector direction
        fourier = odl.trafos.FourierTransform(ray_trafo.range, axes=[1])

        def fft_filter(x):
            return np.maximum(np.abs(x) * np.cos(x / 2.0), 0.0)

        # Create ramp in the detector direction
        ramp_function = fourier.range.element(
            lambda x: fft_filter(x[1]))

        # Create ramp filter via the
        # convolution formula with fourier transforms
        ramp_filter = fourier.inverse * ramp_function * fourier

        # Create filtered backprojection by composing the backprojection
        # (adjoint) with the ramp filter. Also apply a scaling.
        fbp = ray_trafo.adjoint * ramp_filter / (2 * np.pi ** 2)
        # calculate fbp
        fbp_result = fbp(proj_data)
        fbp_result.show('FBP', coords=[None, None, 227], clim=[0.010, 0.025])
        fbp_result.show('FBP', coords=[None, 0, 227])
        xl, yl = plt.xlim(), plt.ylim()
        plt.scatter([-104.71], [33.68], s=300,
                    facecolors='none', edgecolors='r')
        plt.xlim(xl)
        plt.ylim(yl)

        # compare to ref
        folder = 'E:/Data/MayoClinic data/Training Cases/L067/full_1mm_sharp'
        file_name = 'L067_FD_1_SHARP_1.CT.0002.0201.2016.01.21.18.11.40.977560.404633815.IMA'

        dataset = dicom.read_file(folder + '/' + file_name)

        data_array = np.fromstring(dataset.PixelData, 'H')

        data_array = data_array.reshape([512, 512], order='F').T

        arr = np.tile(np.roll(np.rot90(data_array, -1), -17, 0)[:, :, None], (1, 1, 64))
        y = reco_space.element(arr)

        (y * 0.00002 - fbp_result).show('reference',
               coords=[None, None, 227.5],
               clim=[-0.003, 0.003])
    else:
        # Conjugate gradient
        print('Performing CG')

        partial = odl.solvers.ShowPartial('Conjugate gradient',
                                          coords=[None, None, 227],
                                          clim=[0.015, 0.025])


        x = ray_trafo.domain.zero()
        odl.solvers.conjugate_gradient_normal(ray_trafo, x,
                                              proj_data, niter=20,
                                              partial=partial)
