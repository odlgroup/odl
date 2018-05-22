# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tomographic datasets from Mayo Clinic.

In addition to the standard ODL requirements, this library also requires:

    - tqdm
    - dicom
    - A copy of the Mayo dataset, see
    https://www.aapm.org/GrandChallenge/LowDoseCT/#registration
"""

from __future__ import division
import numpy as np
import os
import dicom
import odl
import tqdm

from dicom.datadict import DicomDictionary, NameDict, CleanName
from odl.contrib.datasets.ct.mayo_dicom_dict import new_dict_items

# Update the DICOM dictionary with the extra Mayo tags
DicomDictionary.update(new_dict_items)
NameDict.update((CleanName(tag), tag) for tag in new_dict_items)


__all__ = ('load_projections', 'load_reconstruction')


def _read_projections(folder, proj_start=1, proj_end=-1):
    """Read mayo projections from a folder."""
    projections = []
    datasets = []

    # Get the relevant file names
    file_names = sorted([f for f in os.listdir(folder) if f.endswith(".dcm")])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(folder))

    file_names = file_names[proj_start:proj_end]

    for file_name in tqdm.tqdm(file_names, 'Loading projection data'):
        # read the file
        dataset = dicom.read_file(folder + '/' + file_name)

        # Get some required data
        rows = dataset.NumberofDetectorRows
        cols = dataset.NumberofDetectorColumns
        hu_factor = dataset.HUCalibrationFactor
        rescale_intercept = dataset.RescaleIntercept
        rescale_slope = dataset.RescaleSlope

        # Load the array as bytes
        data_array = np.array(np.frombuffer(dataset.PixelData, 'H'),
                              dtype='float32')
        data_array = data_array.reshape([rows, cols], order='F').T

        # Rescale array
        data_array *= rescale_slope
        data_array += rescale_intercept
        data_array /= hu_factor

        # Store results
        projections.append(data_array)
        datasets.append(dataset)

    return datasets, projections


def load_projections(folder, proj_start=1, proj_end=-1):
    """Load geometry and data stored in Mayo format from folder.

    Parameters
    ----------
    folder : str
        Path to the folder where the Mayo DICOM files are stored.
    proj_start : int
        Index of the first projection to use. Used for subsampling.
    proj_end : int
        Index of the final projection to use.

    Returns
    -------
    geometry : ConeFlatGeometry
        Geometry corresponding to the Mayo projector.
    proj_data : `numpy.ndarray`
        Projection data, given as the line integral of the linear attenuation
        coefficient (g/cm^3). Its unit is thus g/cm^2.
    """
    datasets, projections = _read_projections(folder, proj_start, proj_end)

    data_array = np.empty((len(projections),) + projections[0].shape,
                          dtype='float32')

    # Move data to a big array, change order
    for i, proj in enumerate(projections):
        data_array[i] = proj[:, ::-1]

    # Get the angles
    angles = [d.DetectorFocalCenterAngularPosition for d in datasets]
    angles = -np.unwrap(angles) - np.pi  # different defintion of angles

    # Make a parallel beam geometry with flat detector
    angle_partition = odl.nonuniform_partition(angles)

    # Set minimum and maximum point
    shape = np.array([datasets[0].NumberofDetectorColumns,
                      datasets[0].NumberofDetectorRows])
    pixel_size = np.array([datasets[0].DetectorElementTransverseSpacing,
                           datasets[0].DetectorElementAxialSpacing])

    minp = -(np.array(datasets[0].DetectorCentralElement) - 0.5) * pixel_size
    maxp = minp + shape * pixel_size

    # Create partition for detector
    detector_partition = odl.uniform_partition(minp, maxp, shape)

    # Select geometry parameters
    src_radius = datasets[0].DetectorFocalCenterRadialDistance
    det_radius = (datasets[0].ConstantRadialDistance -
                  datasets[0].DetectorFocalCenterRadialDistance)

    # Convert pitch and offset to odl defintions
    pitch = (pixel_size[1] * shape[1] * datasets[0].SpiralPitchFactor *
             src_radius / (src_radius + det_radius))
    offset_along_axis = (datasets[0].DetectorFocalCenterAxialPosition -
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

    # TODO: WE CURRENTLY IGNORE THE OFFSETS DUE TO FLYING FOCAL SPOT
    source_offsets = np.array([offset_x, offset_y, offset_z]).T

    # Assemble geometry
    geometry = odl.tomo.ConeFlatGeometry(angle_partition,
                                         detector_partition,
                                         src_radius=src_radius,
                                         det_radius=det_radius,
                                         pitch=pitch,
                                         offset_along_axis=offset_along_axis)

    # Create a *temporary* ray transform (we need its range)
    spc = odl.uniform_discr([-1] * 3, [1] * 3, [32] * 3)
    ray_trafo = odl.tomo.RayTransform(spc, geometry, interp='linear')

    # convert coordinates
    theta, up, vp = ray_trafo.range.grid.meshgrid
    d = src_radius + det_radius
    u = d * np.arctan(up / d)
    v = d / np.sqrt(d**2 + up**2) * vp

    # Calculate projection data in rectangular coordinates since we have no
    # backend that supports cylindrical
    proj_data_cylinder = ray_trafo.range.element(data_array)
    interpolated_values = proj_data_cylinder.interpolation((theta, u, v))
    proj_data = ray_trafo.range.element(interpolated_values)

    return geometry, proj_data.asarray()


def load_reconstruction(folder, slice_start=0, slice_end=-1):
    """Load a volume from folder, also returns the corresponding partition.

    Parameters
    ----------
    folder : str
        Path to the folder where the DICOM files are stored.
    slice_start : int
        Index of the first slice to use. Used for subsampling.
    slice_end : int
        Index of the final slice to use.

    Returns
    -------
    partition : `odl.RectPartition`
        Partition describing the geometric positioning of the voxels.
    data : `numpy.ndarray`
        Volumetric data. Scaled such that data = 1 for water (0 HU).

    Notes
    -----
    Note that DICOM data is highly non trivial. Typically, each slice has been
    computed with a slice tickness (e.g. 3mm) but the slice spacing might be
    different from that.

    Further, the coordinates in DICOM is typically the *middle* of the pixel,
    not the corners as in ODL.

    This function should handle all of these peculiarities and give a volume
    with the correct coordinate system attached.
    """
    file_names = sorted([f for f in os.listdir(folder) if f.endswith(".IMA")])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(folder))

    volumes = []
    datasets = []

    file_names = file_names[slice_start:slice_end]

    for file_name in tqdm.tqdm(file_names, 'loading volume data'):
        # read the file
        dataset = dicom.read_file(folder + '/' + file_name)

        # Get parameters
        pixel_size = np.array(dataset.PixelSpacing)
        pixel_thickness = float(dataset.SliceThickness)
        rows = dataset.Rows
        cols = dataset.Columns

        # Get data array and convert to correct coordinates
        data_array = np.array(np.frombuffer(dataset.PixelData, 'H'),
                              dtype='float32')
        data_array = data_array.reshape([cols, rows], order='C')
        data_array = np.rot90(data_array, -1)

        # Convert from CT numbers to densities
        data_array /= 1024.0

        # Store results
        volumes.append(data_array)
        datasets.append(dataset)

    # Compute geometry parameters
    voxel_size = np.array(list(pixel_size) + [pixel_thickness])
    shape = np.array([rows, cols, len(volumes)])
    min_pt = (np.array(dataset.ImagePositionPatient) -
              np.array(dataset.DataCollectionCenterPatient))
    min_pt[:2] += 0.5 * np.array(dataset.PixelSpacing)
    max_pt = min_pt + voxel_size * np.array([rows - 1, cols - 1, 0])

    min_pt[2] = -np.array(datasets[0].DataCollectionCenterPatient)[2]
    min_pt[2] -= 0.5 * pixel_thickness
    max_pt[2] = -np.array(datasets[-1].DataCollectionCenterPatient)[2]
    max_pt[2] += 0.5 * pixel_thickness

    partition = odl.uniform_partition(min_pt, max_pt, shape)

    volume = np.transpose(np.array(volumes), (1, 2, 0))

    return partition, volume


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
