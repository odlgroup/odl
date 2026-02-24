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
import odl
#import tqdm
from functools import partial

from pydicom.datadict import DicomDictionary, keyword_dict
from odl.core.discr.discr_utils import linear_interpolator
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

    data_array = []

    for i, file_name in enumerate(file_names): #enumerate(tqdm.tqdm(file_names,'Loading projection data')):
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
            #rescale_intercept = dataset.RescaleIntercept
            #rescale_slope = dataset.RescaleSlope

        else:
            # Sanity checks
            assert rows == dataset.NumberofDetectorRows
            assert cols == dataset.NumberofDetectorColumns
            #assert rescale_intercept == dataset.RescaleIntercept
            #assert rescale_slope == dataset.RescaleSlope

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


def load_projections(folder, indices=None, use_ffs=True, flat=False, interpolate=True):
    """Load geometry and data stored in Mayo format from folder.
    Parameters
    ----------
    folder : str
        Path to the folder where the Mayo DICOM files are stored.
    indices : optional
        Indices of the projections to load.
        Accepts advanced indexing such as slice or list of indices.
    use_ffs : bool, optional
        If ``True``, a source shift is applied to compensate the flying focal spot.
        Default: ``True``
    flat : bool, optional
        If ``True``, the data is projected on a flat detector.
        Default: ``Flat``
    Returns
    -------
    geometry : ConeBeamGeometry
        Geometry corresponding to the Mayo projector.
    proj_data : `numpy.ndarray`
        Projection data, given as the line integral of the linear attenuation
        coefficient (g/cm^3). Its unit is thus g/cm^2.
    """
    datasets, data_array = _read_projections(folder, indices)

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
    # det_minp = -(np.array(datasets[0].DetectorCentralElement) - 0.5) * det_pixel_size
    # in the implementation of the curved detector it is assumed 
    # that the detector axis are "attached" to the 0-point of the detector,
    # but in the ray transform it is assumed that 
    # the axis are "attached" to the center of the detector.
    # To avoid problems, we make sure that these two point coincide and
    # shift the detector through the detector shift fuction 
    # and rotate the detector axes accordingly 
    det_minp = -det_shape * det_pixel_size / 2 
    det_maxp = det_minp + det_shape * det_pixel_size

    # Select geometry parameters
    src_radius = datasets[0].DetectorFocalCenterRadialDistance
    det_radius = (datasets[0].ConstantRadialDistance -
                  datasets[0].DetectorFocalCenterRadialDistance)
    curv_radius = src_radius + det_radius
    if flat:
        det_curvature_radius = None
    else:
        det_curvature_radius = (curv_radius, None)

    # For unknown reasons, mayo does not include the tag
    # "TableFeedPerRotation", which is what we want.
    # Instead we manually compute the pitch
    table_dist = (datasets[-1].DetectorFocalCenterAxialPosition -
                  datasets[0].DetectorFocalCenterAxialPosition)
    num_rot = (angles[-1] - angles[0]) / (2 * np.pi)
    pitch = table_dist / num_rot
    
    # Convert offset to odl definitions
    offset_along_axis = (datasets[0].DetectorFocalCenterAxialPosition -
                         angles[0] / (2 * np.pi) * pitch)

    # offsets: detector’s focal center -> focal spot
    offset_angular = np.array([d.SourceAngularPositionShift for d in datasets]) 
    offset_radial = np.array([d.SourceRadialDistanceShift for d in datasets]) 
    offset_axial = np.array([d.SourceAxialPositionShift for d in datasets])

    # angles have inverse convention
    shift_d = np.cos(-offset_angular) * (src_radius + offset_radial) - src_radius
    shift_t = np.sin(-offset_angular) * (src_radius + offset_radial)
    # correcting for non-uniform pitch
    det_offset_axial = np.array([d.DetectorFocalCenterAxialPosition for d in datasets])
    shift_z = np.zeros(len(angles))-(det_offset_axial - angles / (2 * np.pi) * pitch - offset_along_axis)
    shift_r = shift_z - offset_axial
    
    # Create partition for detector
    if not flat:
        det_minp[0] /= curv_radius 
        det_maxp[0] /= curv_radius 
    detector_partition = odl.uniform_partition(det_minp, det_maxp, det_shape)

    # Assemble geometry
    angle_partition = odl.nonuniform_partition(angles)
    
    # Flying focal spot
    src_shift_func = None
    if use_ffs:
        shifts = np.transpose(np.vstack([shift_d, shift_t, shift_r]))
        src_shift_func = partial(
            odl.tomo.flying_focal_spot, apart=angle_partition, shifts=shifts)
    else:
        src_shift_func = None
        
    # Detector shift 
    n_agles = len(angles)
    shift_angle =  -((det_shape[0] / 2 - (datasets[0].DetectorCentralElement[0] - 0.5)) 
                   * det_pixel_size[0] / curv_radius)
    shift_d = curv_radius * (np.cos(shift_angle) - 1)
    shift_t = curv_radius * np.sin(shift_angle)
    shifts = np.transpose(np.vstack([np.ones(n_agles) * shift_d, 
                                     np.ones(n_agles) * shift_t, 
                                     shift_z]))
    det_shift_func = partial(
        odl.tomo.flying_focal_spot, apart=angle_partition, shifts=shifts)
    
    # Detector axes (rotate the first axis (1, 0, 0) by shift_angle)
    det_axes_init = [(np.cos(shift_angle), np.sin(shift_angle), 0),
                     (0, 0, 1)]
    
    geometry = odl.tomo.ConeBeamGeometry(angle_partition,
                                         detector_partition,
                                         src_radius=src_radius,
                                         det_radius=det_radius,
                                         pitch=pitch,
                                         det_curvature_radius=det_curvature_radius,
                                         det_axes_init=det_axes_init, 
                                         offset_along_axis=offset_along_axis,
                                         src_shift_func=src_shift_func,
                                         det_shift_func=det_shift_func)

    # number of photons is inverse to the noise variance
    photon_stat = []
    for d in datasets:
        photon_stat.append(d.PhotonStatistics)
    photon_stat = np.expand_dims(np.array(photon_stat), axis=-1)
    
    if flat and interpolate:
        # project the data on a flat grid
        space = odl.discr.uniform_discr([-1,-1,-1], [1,1,1], (3,3,3))
        grid = odl.tomo.RayTransform(space, geometry).range.grid
        data_array = interpolate_flat_grid(data_array, grid, curv_radius)
        
    return geometry, data_array, photon_stat


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
    proj_data_interpolated = interpolator((theta, u, v))

    return proj_data_interpolated


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

    for file_name in file_names:#tqdm.tqdm(file_names, 'loading volume data'):
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
        
    # since pixel_size and pixel_thickness are same accros volume, 
    # we take it from the first .dicom file    
    pixel_size = np.array(datasets[0].PixelSpacing)
    pixel_thickness = float(datasets[0].SliceThickness)

    voxel_size = np.array(list(pixel_size) + [pixel_thickness])
    shape = np.array([rows, cols, len(volumes)])

    # Compute geometry parameters
    #print(datasets[0].ReconstructionDiameter, datasets[0].TableHeight, datasets[0].ReconstructionTargetCenterPatient, datasets[0].DataCollectionCenterPatient)
    if datasets[0].get("ReconstructionTargetCenterPatient") is not None:
        mid_pt = np.array(datasets[0].ReconstructionTargetCenterPatient)
        mid_pt[1] += dataset.TableHeight
    else:
        mid_pt = np.zeros(3)
    reconstruction_size = datasets[0].ReconstructionDiameter
    min_pt = mid_pt - reconstruction_size / 2
    max_pt = mid_pt + reconstruction_size / 2

    # axis 1 has reversed convention
    #min_pt[1], max_pt[1] = -max_pt[1], -min_pt[1]

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
    

    recon_space = odl.uniform_discr(min_pt, max_pt, shape)

    volume = np.transpose(np.array(volumes), (1, 2, 0))

    return recon_space, volume




if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests
    run_doctests()
        
