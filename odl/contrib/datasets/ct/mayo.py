# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tomographic datasets from Mayo Clinic.
In addition to the standard ODL requirements, this library also requires:
    - pydicom
    - Samples from the Mayo dataset, see
    https://ctcicblog.mayo.edu/hubcap/patient-ct-projection-data-library/
"""

from __future__ import division
import numpy as np
import os
import sys
import pydicom
from functools import partial

from pydicom.datadict import DicomDictionary, keyword_dict
from mayo_dicom_dict import new_dict_items

# Update the DICOM dictionary with the extra Mayo tags
DicomDictionary.update(new_dict_items)
NameDict.update((CleanName(tag), tag) for tag in new_dict_items)


__all__ = ('load_projections', 'load_reconstruction')


def _read_projections(folder, indices):
    """Read mayo projections from a folder."""
    datasets = []

    # Get the relevant file names
    file_names = sorted([f for f in os.listdir(folder) if f.endswith(".dcm")])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(folder))

    file_names = file_names[indices]

    data_array = []

    for i, file_name in enumerate(file_names):
        # read the file
        try:
            dataset = pydicom.read_file(os.path.join(folder, file_name))
        except:
            print("corrupted file: {}".format(file_name), file=sys.stderr)
            print("error:\n{}".format(sys.exc_info()[1]), file=sys.stderr)
            raise

        if not data_array:
            # Get some required data
            rows = dataset.NumberofDetectorRows
            cols = dataset.NumberofDetectorColumns
        else:
            # Sanity checks
            assert rows == dataset.NumberofDetectorRows
            assert cols == dataset.NumberofDetectorColumns

        rescale_intercept = dataset.RescaleIntercept
        rescale_slope = dataset.RescaleSlope
        
        # Load the array as bytes
        proj_array = np.array(np.frombuffer(dataset.PixelData, 'H'),
                              dtype='float32')
        #proj_array = proj_array.reshape([rows, cols], order='F').T
        proj_array = proj_array.reshape([cols, rows])

        # Rescale array
        proj_array *= rescale_slope
        proj_array += rescale_intercept

        data_array.append(proj_array[:, ::-1])
        datasets.append(dataset)
    data_array = np.stack(data_array)
    
    return datasets, data_array


def load_projections(folder, indices=None):

    datasets, data_array = _read_projections(folder, indices)

    # Get the angles
    angles = np.array([d.DetectorFocalCenterAngularPosition for d in datasets])
    angles = -np.unwrap(angles) - np.pi 
    
    # position along the rotation axis
    axial_position = -np.array([d.DetectorFocalCenterAxialPosition for d in datasets])

    # detector
    detector_shape = np.array([datasets[0].NumberofDetectorColumns,
                          datasets[0].NumberofDetectorRows])
    pixel_size = np.array([datasets[0].DetectorElementTransverseSpacing,
                               datasets[0].DetectorElementAxialSpacing])


    # Select geometry parameters
    source_radius = datasets[0].DetectorFocalCenterRadialDistance
    detector_radius = (datasets[0].ConstantRadialDistance -
                  datasets[0].DetectorFocalCenterRadialDistance)

    
    # Flying focal spot
    offset_angular = np.array([d.SourceAngularPositionShift for d in datasets]) 
    offset_radial = np.array([d.SourceRadialDistanceShift for d in datasets]) 
    offset_axial = np.array([d.SourceAxialPositionShift for d in datasets])
    # angles have inverse convention
    shift_d = np.cos(-offset_angular) * (src_radius + offset_radial) - src_radius
    shift_t = np.sin(-offset_angular) * (src_radius + offset_radial)
    shift_r = - offset_axial    
    source_shift = np.transpose(np.vstack([shift_d, shift_t, shift_r]))

        
    # Detector shift 
    shift_angle = datasets[0].DetectorCentralElement[0] - 0.5 - det_shape[0] / 2

    # number of photons is inverse to the noise variance
    photon_stat = []
    for d in datasets:
        photon_stat.append(d.PhotonStatistics)
    photon_stat = np.expand_dims(np.array(photon_stat), axis=-1)
    
        
    return angles, axial_position, detector_shape, pixel_size, source_radius, detetor_radius, detector_shift, source_shift, sdata_array, photon_stat



def load_reconstruction(folder, slice_start=0, slice_end=None):
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

    if slice_end is None:
        slice_end = len(file_names)
    file_names = file_names[slice_start:slice_end]

    for file_name in file_names:
        # read the file
        dataset = pydicom.read_file(os.path.join(folder, file_name))

        # Get parameters
        #pixel_size = np.array(dataset.PixelSpacing)
        #pixel_thickness = float(dataset.SliceThickness)
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
        mu_water = 0.0192
        densities = (hu_values + 1000) / 1000 * mu_water

        # Store results
        volumes.append(densities)
        datasets.append(dataset)
        
    shape = np.array([rows, cols, len(volumes)])

    # Compute geometry parameters
    if datasets[0].get("ReconstructionTargetCenterPatient") is not None:
        mid_pt = np.array(datasets[0].ReconstructionTargetCenterPatient)
        mid_pt[1] += dataset.TableHeight
    else:
        mid_pt = np.zeros(3)
    reconstruction_size = datasets[0].ReconstructionDiameter
    min_pt = mid_pt - reconstruction_size / 2
    max_pt = mid_pt + reconstruction_size / 2

    if len(datasets) > 1:
        slice_distance = np.abs(
            float(datasets[1].ImagePositionPatient[2]) -
            float(datasets[0].ImagePositionPatient[2]))
    else:
        # If we only have one slice, we must approximate the distance.
        slice_distance = pixel_thickness

    if 'Siemens'.upper() in dataset.Manufacturer.upper(): 
        if 'HEAD' in dataset.BodyPartExamined.upper():
            min_pt[2] = np.array(datasets[0].ImagePositionPatient)[2]
            max_pt[2] = np.array(datasets[-1].ImagePositionPatient)[2]
        else:
            min_pt[2] = -np.array(datasets[0].ImagePositionPatient)[2]
            max_pt[2] = -np.array(datasets[-1].ImagePositionPatient)[2]
    else:
        if 'HEAD' in dataset.BodyPartExamined.upper():
            min_pt[2] = np.array(datasets[0].ImagePositionPatient)[2]
            max_pt[2] = np.array(datasets[-1].ImagePositionPatient)[2]
        else:
            min_pt[2] = np.array(datasets[-1].ImagePositionPatient)[2]
            max_pt[2] = np.array(datasets[0].ImagePositionPatient)[2]
            volumes = volumes[::-1]
    # The middle of the minimum/maximum slice can be computed from the
    # DICOM attribute "DataCollectionCenterPatient". Since ODL uses corner
    # points (e.g. edge of volume) we need to add half a voxel thickness to
    # both sides. 
    min_pt[2] -= 0.5 * slice_distance
    max_pt[2] += 0.5 * slice_distance
    


    volume = np.transpose(np.array(volumes), (1, 2, 0))

    return min_pt, max_pt, shape, volume

        
