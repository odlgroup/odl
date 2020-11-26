# Copyright 2014-2020 The ODL contributors
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
import pydicom
import odl
import tqdm

from pydicom.datadict import DicomDictionary, keyword_dict
from odl.discr.discr_utils import linear_interpolator
from odl.contrib.datasets.ct.mayo_dicom_dict import new_dict_items

# Update the DICOM dictionary with the extra Mayo tags
DicomDictionary.update(new_dict_items)
# Update the reverse mapping from name to tag
new_names_dict = dict([(val[4], tag) for tag, val in new_dict_items.items()])
keyword_dict.update(new_names_dict)


__all__ = ('load_projections', 'load_reconstruction')


def _read_projections(folder, indices):
    """Read mayo projections from a folder."""
    datasets = []

    # Get the relevant file names
    file_names = sorted([f for f in os.listdir(folder) if f.endswith(".dcm")])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(folder))

    file_names = file_names[indices]

    data_array = None

    for i, file_name in enumerate(tqdm.tqdm(file_names,
                                            'Loading projection data')):
        # read the file
        dataset = pydicom.read_file(folder + '/' + file_name)

        if data_array is None:
            # Get some required data
            rows = dataset.NumberofDetectorRows
            cols = dataset.NumberofDetectorColumns
            rescale_intercept = dataset.RescaleIntercept
            rescale_slope = dataset.RescaleSlope

            data_array = np.empty((len(file_names), cols, rows),
                                  dtype='float32')
            angles = np.empty(len(file_names), dtype='float32')

        else:
            # Sanity checks
            assert rows == dataset.NumberofDetectorRows
            assert cols == dataset.NumberofDetectorColumns
            assert rescale_intercept == dataset.RescaleIntercept
            assert rescale_slope == dataset.RescaleSlope


        # Load the array as bytes
        proj_array = np.array(np.frombuffer(dataset.PixelData, 'H'),
                              dtype='float32')
        proj_array = proj_array.reshape([cols, rows])
        # proj_array = proj_array.reshape([rows, cols], order='F').T

        # Rescale array (no HU)
        proj_array *= rescale_slope
        proj_array += rescale_intercept

        data_array[i] = proj_array[:, ::-1]
        angles[i] = dataset.DetectorFocalCenterAngularPosition
        datasets.append(dataset)

    return datasets, data_array, angles


def load_projections(folder, indices=None):
    """Load geometry and data stored in Mayo format from folder.

    Parameters
    ----------
    folder : str
        Path to the folder where the Mayo DICOM files are stored.
    indices : optional
        Indices of the projections to load.
        Accepts advanced indexing such as slice or list of indices.
    num_slices: int, optional
        Number of slices to consider for the reconstruction volume;
        the other parameters are hard-coded. With default value (None)
        a *temporary* volume is created.

    Returns
    -------
    geometry : ConeBeamGeometry
        Geometry corresponding to the Mayo projector.
    proj_data : `numpy.ndarray`
        Projection data, given as the line integral of the linear attenuation
        coefficient (g/cm^3). Its unit is thus g/cm^2.
    """
    datasets, data_array, angles = _read_projections(folder, indices)

    # Reverse angular axis and set origin at 6 o'clock
    angles = -np.unwrap(angles) - np.pi 

    # Set minimum and maximum corners
    det_shape = np.array([datasets[0].NumberofDetectorColumns,
                          datasets[0].NumberofDetectorRows])
    det_pixel_size = np.array([datasets[0].DetectorElementTransverseSpacing,
                               datasets[0].DetectorElementAxialSpacing])

    # Correct from center of pixel to corner of pixel
    det_minp = -(np.array(datasets[0].DetectorCentralElement) - 0.5) * det_pixel_size
    det_maxp = det_minp + det_shape * det_pixel_size

    # Select geometry parameters
    src_radius = datasets[0].DetectorFocalCenterRadialDistance
    det_radius = (datasets[0].ConstantRadialDistance -
                  datasets[0].DetectorFocalCenterRadialDistance)

    # For unknown reasons, mayo does not include the tag
    # "TableFeedPerRotation", which is what we want.
    # Instead we manually compute the pitch
    table_dist = datasets[-1].DetectorFocalCenterAxialPosition - \
                 datasets[0].DetectorFocalCenterAxialPosition
    num_rot = (angles[-1] - angles[0]) / (2 * np.pi)
    pitch = table_dist / num_rot

    # TODO: Understand and re-implement the flying focal spot
    # # Get flying focal spot data
    # offset_axial = np.array([d.SourceAxialPositionShift for d in datasets])
    # offset_angular = np.array([d.SourceAngularPositionShift for d in datasets])
    # offset_radial = np.array([d.SourceRadialDistanceShift for d in datasets])

    # # TODO(adler-j): Implement proper handling of flying focal spot.
    # # Currently we do not fully account for it, merely making some "first
    # # order corrections" to the detector position and radial offset.

    # # Update angles with flying focal spot (in plane direction).
    # # This increases the resolution of the reconstructions.
    # angles = angles - offset_angular

    # # We correct for the mean offset due to the rotated angles, we need to
    # # shift the detector.
    # offset_detector_by_angles = det_radius * np.mean(offset_angular)
    # det_minp[0] -= offset_detector_by_angles
    # det_maxp[0] -= offset_detector_by_angles

    # # We currently apply only the mean of the offsets
    # src_radius = src_radius + np.mean(offset_radial)

    # # Partially compensate for a movement of the source by moving the object
    # # instead. We need to rescale by the magnification to get the correct
    # # change in the detector. This approximation is only exactly valid on the
    # # axis of rotation.
    # mean_offset_along_axis_for_ffz = np.mean(offset_axial) * (
    #     src_radius / (src_radius + det_radius))

    # Create partition for detector
    detector_partition = odl.uniform_partition(det_minp, det_maxp, det_shape)

    # Convert offset to odl definitions
    # offset_along_axis = (mean_offset_along_axis_for_ffz +
    offset_along_axis = datasets[0].DetectorFocalCenterAxialPosition - \
                        angles[0] / (2 * np.pi) * pitch

    # Assemble geometry
    angle_partition = odl.nonuniform_partition(angles)
    geometry = odl.tomo.ConeBeamGeometry(angle_partition,
                                         detector_partition,
                                         src_radius=src_radius,
                                         det_radius=det_radius,
                                         pitch=pitch,
                                         offset_along_axis=offset_along_axis)

    return geometry, data_array

    

def get_default_recon_space():
    # Create a *temporary* ray transform (we need its range)
    num_slices = 97
    pixel_spacing = np.array([0.75,0.75])
    num_pixel = np.array([512,512])
    slice_dist = 5.
    origin = np.zeros(3)
    mid_table = (datasets[0].DetectorFocalCenterAxialPosition +
                    datasets[-1].DetectorFocalCenterAxialPosition) / 2
    min_pt = np.copy(origin)
    min_pt[:2] -= pixel_spacing * num_pixel / 2
    min_pt[2] += mid_table - num_slices * slice_dist / 2

    max_pt = np.copy(min_pt)
    max_pt[:2] += pixel_spacing * num_pixel
    max_pt[2] += num_slices * slice_dist

    recon_dim = np.array([*num_pixel, num_slices], dtype=np.int32)
    recon_space = odl.uniform_discr(min_pt, max_pt,
                                    shape=recon_dim,
                                    dtype=np.float32)
    return recon_space


# ray_trafo = odl.tomo.RayTransform(recon_space, geometry, interp='linear')

def interpolate_flat_grid(data_array, range_grid, radial_dist):
    # convert coordinates
    theta, up, vp = range_grid.meshgrid #ray_trafo.range.grid.meshgrid
    # d = src_radius + det_radius
    u = radial_dist * np.arctan(up / radial_dist)
    v = radial_dist / np.sqrt(radial_dist**2 + up**2) * vp

    # Calculate projection data in rectangular coordinates since we have no
    # backend that supports cylindrical
    interpolator = linear_interpolator(
        data_array, range_grid.coord_vectors # ray_trafo.range.grid.coord_vectors
    )
    proj_data = interpolator((theta, u, v))

    return proj_data



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
    DICOM data is highly non trivial. Typically, each slice has been computed
    with a slice tickness (e.g. 3mm) but the slice spacing might be
    different from that.

    Further, the coordinates in DICOM is typically the *middle* of the pixel,
    not the corners as in ODL.

    This function should handle all of these peculiarities and give a volume
    with the correct coordinate system attached.
    """
    file_names = sorted([f for f in os.listdir(folder) if f.endswith(".dcm")])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(folder))

    volumes = []
    datasets = []

    file_names = file_names[slice_start:slice_end]

    for file_name in tqdm.tqdm(file_names, 'loading volume data'):
        # read the file
        dataset = pydicom.read_file(folder + '/' + file_name)

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

        # Convert from storage type to densities
        # TODO: Optimize these computations
        hu_values = (dataset.RescaleSlope * data_array +
                     dataset.RescaleIntercept)

        # Store results
        volumes.append(hu_values)
        datasets.append(dataset)

    voxel_size = np.array(list(pixel_size) + [pixel_thickness])
    shape = np.array([rows, cols, len(volumes)])

    # Compute geometry parameters
    mid_pt = np.array(dataset.ReconstructionTargetCenterPatient)
    mid_pt[1] += datasets[0].TableHeight
    reconstruction_size = (voxel_size * shape)
    min_pt = mid_pt - reconstruction_size / 2
    max_pt = mid_pt + reconstruction_size / 2

    # axis 1 has reversed convention
    min_pt[1], max_pt[1] = -max_pt[1], -min_pt[1]

    if len(datasets) > 1:
        slice_distance = np.abs(
            float(datasets[1].DataCollectionCenterPatient[2]) -
            float(datasets[0].DataCollectionCenterPatient[2]))
    else:
        # If we only have one slice, we must approximate the distance.
        slice_distance = pixel_thickness

    # The middle of the minimum/maximum slice can be computed from the
    # DICOM attribute "DataCollectionCenterPatient". Since ODL uses corner
    # points (e.g. edge of volume) we need to add half a voxel thickness to
    # both sides.
    min_pt[2] = -np.array(datasets[0].DataCollectionCenterPatient)[2]
    min_pt[2] -= 0.5 * slice_distance
    max_pt[2] = -np.array(datasets[-1].DataCollectionCenterPatient)[2]
    max_pt[2] += 0.5 * slice_distance

    partition = odl.uniform_partition(min_pt, max_pt, shape)
    recon_space = odl.uniform_discr_frompartition(partition, dtype='float32')

    volume = np.transpose(np.array(volumes), (1, 2, 0))

    return recon_space, volume


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
