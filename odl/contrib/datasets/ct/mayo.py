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
    - pydicom
    - Samples from the Mayo dataset, see
    https://ctcicblog.mayo.edu/hubcap/patient-ct-projection-data-library/
"""

from __future__ import division
import numpy as np
import os
import sys
import pydicom
import odl
import tqdm
from functools import partial

from pydicom.datadict import DicomDictionary, keyword_dict
from odl.discr.discr_utils import linear_interpolator
from odl.contrib.datasets.ct.mayo_dicom_dict import new_dict_items

# Update the DICOM dictionary with the extra Mayo tags
DicomDictionary.update(new_dict_items)
# Update the reverse mapping from name to tag
new_names_dict = dict([(val[4], tag) for tag, val in new_dict_items.items()])
keyword_dict.update(new_names_dict)


__all__ = ('load_projections', 'load_reconstruction')


def _read_projections(dir, indices):
    """Read mayo projections from a directory."""
    datasets = []
    data_array = []

    # Get the relevant file names
    file_names = sorted([f for f in os.listdir(dir) if f.endswith(".dcm")])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(dir))

    file_names = file_names[indices]

    for i, file_name in enumerate(tqdm.tqdm(file_names,
                                            'Loading projection data')):
        # read the file
        try:
            dataset = pydicom.read_file(os.path.join(dir, file_name))
        except:
            print("corrupted file: {}".format(file_name), file=sys.stderr)
            print("error:\n{}".format(sys.exc_info()[1]), file=sys.stderr)
            raise

        if not data_array:
            # Get some required data
            rows = dataset.NumberofDetectorRows
            cols = dataset.NumberofDetectorColumns
            rescale_intercept = dataset.RescaleIntercept
            rescale_slope = dataset.RescaleSlope

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
        data_array.append(proj_array[:, ::-1])
        datasets.append(dataset)

    data_array = np.stack(data_array)
    # Rescale array
    data_array *= rescale_slope
    data_array += rescale_intercept

    return datasets, data_array


def load_projections(dir, indices=None, use_ffs=True):
    """Load geometry and data stored in Mayo format from dir.

    Parameters
    ----------
    dir : str
        Path to the directory where the Mayo DICOM files are stored.
    indices : optional
        Indices of the projections to load.
        Accepts advanced indexing such as slice or list of indices.
    use_ffs : bool, optional
        If ``True``, a source shift is applied to compensate the flying focal spot.
        Default: ``True``

    Returns
    -------
    geometry : ConeBeamGeometry
        Geometry corresponding to the Mayo projector.
    proj_data : `numpy.ndarray`
        Projection data, given as the line integral of the linear attenuation
        coefficient (g/cm^3). Its unit is thus g/cm^2.
    """
    datasets, data_array = _read_projections(dir, indices)

    # Get the angles
    angles = np.array([d.DetectorFocalCenterAngularPosition for d in datasets])
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
    table_dist = (datasets[-1].DetectorFocalCenterAxialPosition -
                  datasets[0].DetectorFocalCenterAxialPosition)
    num_rot = (angles[-1] - angles[0]) / (2 * np.pi)
    pitch = table_dist / num_rot

    # offsets: detector’s focal center -> focal spot
    offset_angular = np.array([d.SourceAngularPositionShift for d in datasets])
    offset_radial = np.array([d.SourceRadialDistanceShift for d in datasets])
    offset_axial = np.array([d.SourceAxialPositionShift for d in datasets])

    # angles have inverse convention
    shift_d = np.cos(-offset_angular) * (src_radius + offset_radial) - src_radius
    shift_t = np.sin(-offset_angular) * (src_radius + offset_radial)
    shift_r = + offset_axial

    shifts = np.transpose(np.vstack([shift_d, shift_t, shift_r]))

    # Create partition for detector
    detector_partition = odl.uniform_partition(det_minp, det_maxp, det_shape)

    # Convert offset to odl definitions
    offset_along_axis = (datasets[0].DetectorFocalCenterAxialPosition -
                         angles[0] / (2 * np.pi) * pitch)

    # Assemble geometry
    angle_partition = odl.nonuniform_partition(angles)

    # Flying focal spot
    src_shift_func = None
    if use_ffs:
        src_shift_func = partial(
            odl.tomo.flying_focal_spot, apart=angle_partition, shifts=shifts)
    else:
        src_shift_func = None

    geometry = odl.tomo.ConeBeamGeometry(angle_partition,
                                         detector_partition,
                                         src_radius=src_radius,
                                         det_radius=det_radius,
                                         pitch=pitch,
                                         offset_along_axis=offset_along_axis,
                                         src_shift_func=src_shift_func)

    return geometry, data_array


def interpolate_flat_grid(data_array, range_grid, radial_dist):
    """Return the linear interpolator of the projection data on a flat detector.

    Parameters
    ----------
    data_array : `numpy.ndarray`
        Projection data on the cylindrical detector that should be interpolated.
    range_grid : RectGrid
        Rectilinear grid on the flat detector.
    radial_dist : float
        The constant radial distance, that is the distance between the detector’s
        focal center and its central element.

    Returns
    -------
    proj_data : `numpy.ndarray`
        Interpolated projection data on the flat rectilinear grid.
    """

    # convert coordinates
    theta, up, vp = range_grid.meshgrid

    u = radial_dist * np.arctan(up / radial_dist)
    v = radial_dist / np.sqrt(radial_dist**2 + up**2) * vp

    # Calculate projection data in rectangular coordinates since we have no
    # backend that supports cylindrical
    interpolator = linear_interpolator(
        data_array, range_grid.coord_vectors
    )
    proj_data = interpolator((theta, u, v))

    return proj_data


def load_reconstruction(dir, slice_start=0, slice_end=-1):
    """Load a volume from dir, also returns the corresponding partition.

    Parameters
    ----------
    dir : str
        Path to the directory where the DICOM files are stored.
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
    file_names = sorted([f for f in os.listdir(dir) if f.endswith(".dcm")])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(dir))

    volumes = []
    datasets = []

    file_names = file_names[slice_start:slice_end]

    for file_name in tqdm.tqdm(file_names, 'loading volume data'):
        # read the file
        dataset = pydicom.read_file(os.path.join(dir, file_name))

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
