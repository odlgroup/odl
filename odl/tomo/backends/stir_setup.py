from __future__ import print_function, division, absolute_import
from future import standard_library

import odl

standard_library.install_aliases()

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import operator

from odl.util.graphics import show_discrete_data
from odl.discr.grid import RegularGrid

from odl.discr.lp_discr import DiscreteLp, DiscreteLpVector
from odl.space.ntuples import FnVector
from odl.tomo.geometry import (\
    Geometry, Parallel2dGeometry, DivergentBeamGeometry, ParallelGeometry,\
    FlatDetector, Flat2dDetector)
from odl.tomo.util.utility import perpendicular_vector
from odl.tomo.geometry.pet import CylindricalPetGeom

try:
    import stir
    # Fix for stirextra being moved around in various stir versions
    try:
        stirextra = stir.stirextra
    except AttributeError:
        import stirextra

    STIR_AVAILABLE = True
except ImportError:
    STIR_AVAILABLE = False

__all__ = ('STIR_AVAILABLE',
           'stir_get_projection_data_info',\
           'stir_get_projection_data',\
           'stir_operate_STIR_and_ODL_vectors',\
           'stir_get_ODL_domain_which_honours_STIR_restrictions',
           'stir_get_ODL_geometry_which_honours_STIR_restrictions',\
           'stir_get_STIR_geometry',\
           'stir_get_STIR_domain_from_ODL',\
           'stir_get_ODL_domain_from_STIR',\
           'stir_get_STIR_image_from_ODL_Vector',\
           'stir_get_STIR_data_as_array',\
           'stir_unified_display_function' )

#
#
# INTERFACE FUNCTIONS FROM ODL TO STIR
#

def stir_get_ODL_domain_which_honours_STIR_restrictions(_vox_num, _vox_size):
    """
    In the future a geometry should be imported to handle scanner alignment restrictions.
    Returns
    -------

    .. warning:: STIR coordinates are currently related to the scanner, i.e. not to the patient (as in DICOM).
        For an image with zero offset, the origin is assumed to coincide with the centre of the first plane.
        I strongly suggest for the time being to let the origin default to (0,0,0) as STIR regards it.


    """

    range = [a*b for a,b in zip(_vox_num,_vox_size)]
    min_p = [-x / 2 for x in range]
    max_p = [a+b for a,b in zip(min_p,range)]

    min_p[2] = 0.0
    max_p[2] = range[2]

    return odl.uniform_discr(
            min_corner=min_p, max_corner=max_p, nsamples= _vox_num,
            dtype='float32')


def stir_get_ODL_geometry_which_honours_STIR_restrictions(_det_y_size_mm, _det_z_size_mm,\
                                                           _num_rings, _num_dets_per_ring,\
                                                           _det_radius):
    """
    This function will return a CylindricalPETGeom which will match with a
    STIR Scanner object.

    .The first ring [0] of the scanner should be the one furthest from the bed.
    .The y axis should be pointing downwards.
    .The z axis should be the longitude.
    .The crystal with the first transverse ID should be the one with the most
     negative y [x=0, y = -r , z= 0].

    Parameters
    ----------
    _det_y_size_mm
    _det_z_size_mm
    _num_rings
    _num_dets_per_ring
    _det_radius

    Returns
    -------

    """
    if _det_radius <= 0:
        raise ValueError('ring circle radius {} is not positive.'
                             ''.format(_det_radius))

    axis = [0, 0, 1]

    first_tmpl_det_point = [-_det_y_size_mm/2, -_det_z_size_mm/2]
    last_tmpl_det_point = [_det_y_size_mm/2, _det_z_size_mm/2]

    # Template of detectors
    tmpl_det = odl.uniform_partition(first_tmpl_det_point,\
                                         last_tmpl_det_point,\
                                         [1, 1])

    # Perpendicular vector from the ring center to
    # the first detector of the same ring
    ring_center_to_tmpl_det = perpendicular_vector(axis)

    det_init_axis_0 = np.cross(axis, ring_center_to_tmpl_det)
    det_init_axes = (det_init_axis_0, axis)

    apart = odl.uniform_partition(0, 2 * np.pi, _num_dets_per_ring,
                                      nodes_on_bdry=True)

    detector = Flat2dDetector(tmpl_det, det_init_axes)

    # Axial (z-axis) movement parameters.
    # The middle of the first ring should be on (r,r,0.0)
    axialpart = odl.uniform_partition(0, _num_rings*_det_z_size_mm, _num_rings)

    return odl.tomo.geometry.pet.CylindricalPetGeom(_det_radius,
                                                ring_center_to_tmpl_det,
                                                apart,
                                                detector,
                                                axialpart)


def stir_get_STIR_geometry(_num_rings, _num_dets_per_ring,
                           _det_radius, _ring_spacing,
                           _average_depth_of_inter,
                           _voxel_size_xy,
                           _axial_crystals_per_block = 1, _trans_crystals_per_block= 1,
                           _axials_blocks_per_bucket = 1, _trans_blocks_per_bucket = 1,
                           _axial_crystals_per_singles_unit = 1, _trans_crystals_per_singles_unit = 1,
                           _num_detector_layers = 1, _intrinsic_tilt = 0):

    # Roughly speaking number of detectors on the diameter
    # bin_size = (_det_radius*2) / (_num_dets_per_ring/2)
    max_num_non_arc_cor_bins = int(_num_dets_per_ring/2)

    scanner = stir.Scanner.get_scanner_from_name('Unknown_scanner')

    scanner.set_num_rings(_num_rings)

    scanner.set_default_bin_size(np.float32(_voxel_size_xy))
    scanner.set_default_num_arccorrected_bins(np.int32(max_num_non_arc_cor_bins))
    scanner.set_default_intrinsic_tilt(np.float32(_intrinsic_tilt))
    scanner.set_inner_ring_radius(np.float32(_det_radius))
    scanner.set_ring_spacing(np.float32(_ring_spacing))
    scanner.set_average_depth_of_interaction(np.float32(_average_depth_of_inter))
    scanner.set_max_num_non_arccorrected_bins(np.int32(max_num_non_arc_cor_bins))
    scanner.set_num_axial_blocks_per_bucket(np.int32(_axials_blocks_per_bucket))
    scanner.set_num_transaxial_blocks_per_bucket(np.int32(_trans_blocks_per_bucket))
    scanner.set_num_axial_crystals_per_block(np.int32(_axial_crystals_per_block))
    scanner.set_num_transaxial_crystals_per_block(np.int32(_trans_crystals_per_block))
    scanner.set_num_axial_crystals_per_singles_unit(np.int32(_axial_crystals_per_singles_unit))
    scanner.set_num_transaxial_crystals_per_singles_unit(np.int32(_trans_crystals_per_singles_unit))
    scanner.set_num_detector_layers(np.int32(_num_detector_layers))

    return scanner


def stir_get_projection_data_info(_domain,\
                                  _stir_scanner, _span_num,\
                                  _max_num_segments, _num_of_views,\
                                  _num_non_arccor_bins, _data_arc_corrected):
    """
    ... more documentation needed ...
    Parameters
    ----------
    _domain
    _stir_scanner
    _span_num
    _max_num_segments
    _num_of_views
    _num_non_arccor_bins
    _data_arc_corrected

    Returns
    -------

    """

    if not isinstance( _domain, stir.FloatVoxelsOnCartesianGrid):
        raise TypeError('The domain must be a STIR FloatVoxelsOnCartesianGrid'
                        'object')

    scanner_vox_size = _stir_scanner.get_ring_spacing()
    domain_vox_size = _domain.get_voxel_size()

    if not np.fmod( np.float32(scanner_vox_size), np.float32(domain_vox_size[3])) == 0.0:
        raise ValueError('The domain voxel size should divide the scanner\'s ring spacing')

    num_rings = _stir_scanner.get_num_rings()

    span_num = np.int32(_span_num)
    if _max_num_segments == -1:
        max_ring_diff = np.int32(num_rings -1)
    else:
        max_ring_diff = np.int32(_max_num_segments)

    num_of_views = np.int32(_num_of_views)
    num_non_arccor_bins = np.int32(_stir_scanner.get_default_num_arccorrected_bins())

    return stir.ProjDataInfo.ProjDataInfoCTI(_stir_scanner, span_num,\
                                             max_ring_diff, num_of_views,\
                                             num_non_arccor_bins, _data_arc_corrected)





def stir_get_projection_data(_projdata_info,
                             _zeros):
    """
    Initialize a ProjData object based on the ProjDataInfo
    Parameters
    ----------
    _projdata_info
    _zeros

    Returns
    -------

    """

    exam_info = get_examination_info()

    return stir.ProjDataInMemory(exam_info, _projdata_info, _zeros)


def stir_get_STIR_domain_from_ODL(_discreteLP, _fill_value=0.0):
    """
    Interface function to get a STIR domain without caring about the classes names.

    Parameters
    ----------
    _discreteLP

    Returns
    -------

    """

    return create_empty_VoxelsOnCartesianGrid_from_DiscreteLP(_discreteLP, _fill_value)


def stir_get_ODL_domain_from_STIR(_voxels):
    """
    Interface function to get an ODL domain without caring about the classes names.

    Parameters
    ----------
    _voxelsF

    Returns
    -------

    """
    return create_DiscreteLP_from_STIR_VoxelsOnCartesianGrid(_voxels)

def stir_get_STIR_image_from_ODL_Vector(_domain, _data):
    """
    This function can be used to get a STIR phantom (emmition or attenuation
    data) from an ODL vector.

    Parameters
    ----------
    _data

    Returns
    -------

    """
    if not isinstance( _domain, DiscreteLp):
            raise TypeError('An ODL DiscreteLP is required as first input')

    if not isinstance( _data, DiscreteLpVector):
            raise TypeError('An ODL DiscreteLPVector is required as second input')

    stir_image = stir_get_STIR_domain_from_ODL(_domain)

    stir_operate_STIR_and_ODL_vectors(stir_image, _data, '+')

    return stir_image


def stir_operate_STIR_and_ODL_vectors(_stir_data, _odl_data, _operator):
    """
    This function can perform some simple operations, Only addition and
    multiplication are currently available.

    Parameters
    ----------
    _stir_data
    _odl_data
    _operator

    Returns
    -------

    """
    if not isinstance( _odl_data, DiscreteLpVector) or not\
            isinstance( _stir_data, stir.FloatVoxelsOnCartesianGrid):
            raise TypeError('The first input should be the STIR data'
                            'and the second value should be ODL Vector')

    stir_ind = _stir_data.get_max_indices()
    stir_max_ind = (stir_ind[1]+1, stir_ind[2]+1, stir_ind[3]+1)

    odl_max_ind = _odl_data.shape

    if not stir_max_ind == odl_max_ind:
        raise ValueError('The arrays must have the same dimentions! stir array:{}, odl array{}'
                             .format(stir_max_ind, odl_max_ind))

    odl_array = _odl_data.asarray().astype(np.float32)
    trans_phantom_array = transform_array_to_STIR_orientation(odl_array)

    stir_array = stirextra.to_numpy(_stir_data)

    if _operator is '+':
        res = np.add(stir_array, trans_phantom_array)
    elif _operator is '*':
        res = np.multiply(stir_array, trans_phantom_array)

    for i in range(0, odl_max_ind[0],1):
        for j in range(0, odl_max_ind[1],1):
            for k in range(0, odl_max_ind[2], 1):
                _stir_data[i, j, k] = res[i,j,k]

def stir_get_STIR_data_as_array(_stir_data):
    """
    A wrapper to the stir.extra function
    Parameters
    ----------
    _stir_data

    Returns
    -------

    """
    return stirextra.to_numpy(_stir_data)


def stir_unified_display_function(_display_me, _in_this_grid, _title=""):
    """
    This is a helper function. STIR, ODL and NumPy used different functions to display images.
    I created this function, in order to avoid flips and rotates.
    It which calls odl.utils.graphics.show_discrete_data.

    Parameters
    ----------
    _display_me: A NumPy array.

    _in_this_grid: A suitable grid

    _title: A title for the figure

    Returns
    -------
    A matplotlib.pyplot figure
    """

    grid = get_2D_grid_from_domain(_in_this_grid)

    fig = plt.figure()
    show_discrete_data(_display_me, grid, fig = fig)
    fig.canvas.set_window_title(_title)

#
#
# FUNCTIONS FROM STIR TO ODL
#

#
#
# TRANSFORM FUNCTIONS
#

def transform_array_to_STIR_orientation(_this_array):
    """
    This is transformation function. The input is a numpy array from a DiscreteLPVector and returns
    an array which can be used to fill a STIR image and maintain structure.
    Parameters
    ----------
    _this_array: A NumPy array from a DiscreteLPVector

    Returns
    -------
    A NumPy array compatible to STIR

    """

    _this_array = np.rot90(_this_array,-1)
    _this_array = np.fliplr(_this_array)
    # STIR indices are [z, y, x]
    _this_array = np.swapaxes(_this_array,0,2)

    # I have to copy in order to transform the actual data
    return _this_array.copy()


#
#
# HELPERS
#

def get_volume_geometry(discr_reco):
    """
    This is a helper function which returns the total size and voxel number of a
    discretised object.

    Parameters
    ----------
    discr_reco: A discretised ODL object

    Returns
    -------

    """
    vol_shp = discr_reco.partition.shape
    voxel_size = discr_reco.cell_sides

    return np.asarray(vol_shp, dtype=np.int32), np.asarray(voxel_size, dtype=np.float32)


def create_empty_VoxelsOnCartesianGrid_from_DiscreteLP(_discr, _fill_vale = 0.0):
    """
    This class defines multi-dimensional (numeric) arrays.
    This class implements multi-dimensional arrays which can have 'irregular' ranges.
    See IndexRange for a description of the ranges. Normal numeric operations are defined.
    In addition, two types of iterators are defined, one which iterators through the outer index,
    and one which iterates through all elements of the array.

    Array inherits its numeric operators from NumericVectorWithOffset.
    In particular this means that operator+= etc. potentially grow the object.
    However, as grow() is a virtual function,
    Array::grow is called, which initialises new elements first to 0.

    Parameters
    ----------
    _discr

    Returns
    -------
        A stir.FloatVoxelsOnCartesianGrid object of the aforementioned dimensions, filled with zeros
    """

    im_dim, vox_size = get_volume_geometry(_discr)

    # Number of voxels
    # This function returns [ x, y, z]
    range_size = stir.Int3BasicCoordinate()
    range_size[1] = np.int32(im_dim[0]) #: x
    range_size[2] = np.int32(im_dim[1]) #: y
    range_size[3] = np.int32(im_dim[2]) #: z

    ind_range = stir.IndexRange3D(range_size)

    # Voxel size
    # This function returns [ x, y, z]
    voxel_size = stir.Float3BasicCoordinate()
    voxel_size[1] = np.float32(vox_size[0]) #: z
    voxel_size[2] = np.float32(vox_size[1]) #: y
    voxel_size[3] = np.float32(vox_size[2]) #: x

    # Shift initial point relatively to 0,0,0
    # This function returns [ x, y, z]
    im_origin = stir.FloatCartesianCoordinate3D()
    im_origin[1] = np.float32(- (im_dim[0] * vox_size[0]) / 2.0)
    im_origin[2] = np.float32(- (im_dim[1] * vox_size[1]) / 2.0)
    im_origin[3] = np.float32(0.0)

    domain = stir.FloatVoxelsOnCartesianGrid( ind_range, im_origin, voxel_size)
    domain.fill(np.float32(_fill_vale))

    return domain


def create_DiscreteLP_from_STIR_VoxelsOnCartesianGrid(_voxels):
    """
    This function tries to transform the VoxelsOnCartesianGrid to
    DicreteLP.

    Parameters
    ----------
    _voxels: A VoxelsOnCartesianGrid Object

    Returns
    -------
    An ODL DiscreteLP object with characteristics of the VoxelsOnCartesianGrid
    """

    # This function returns the coordinates as x - y - z
    stir_vox_num = _voxels.get_max_indices()
    vox_num = [stir_vox_num[1]+1, stir_vox_num[2]+1,stir_vox_num[3]+1]

    # This function returns the coordinates as x - y - z
    stir_vol_max = _voxels.get_physical_coordinates_for_indices(_voxels.get_max_indices())
    stir_vol_min = _voxels.get_physical_coordinates_for_indices(_voxels.get_min_indices())

    # This function returns the coordinates as x - y - z
    stir_vox_size = _voxels.get_voxel_size()

    # An one voxel size to get to most righ-most boundary
    vol_max = [stir_vol_max[1]+stir_vox_size[1], stir_vol_max[2]+stir_vox_size[2],stir_vol_max[3]+stir_vox_size[3]]
    vol_min = [stir_vol_min[1], stir_vol_min[2],stir_vol_min[3]]

    return odl.uniform_discr(
            min_corner=vol_min, max_corner=vol_max, nsamples=vox_num,
            dtype='float32')


def get_2D_grid_from_domain(_this_domain):
    """

    Parameters
    ----------
    _this_domain

    Returns
    -------

    """
    return  RegularGrid([_this_domain.space.partition.begin[0], _this_domain.space.partition.begin[1]],
                        [_this_domain.space.partition.end[0], _this_domain.space.partition.end[1]], (2, 3))


def get_examination_info():
    """
    Unless you do motion correction or list-mode reconstruction, default it to [0,1]
    And don't bother more.
    In think that a time frame [0, 1] - corresponds to one bed position
    in a generic way and STIR will ignore it,

    Parameters
    ----------
    _time_frame

    Returns
    -------

    """
    _time_frame = [0.0, 1.0]

    time_starts = np.array([_time_frame[0]], dtype=np.float64)
    time_ends = np.array([_time_frame[1]], dtype=np.float64)

    time_frame_def = stir.TimeFrameDefinitions(time_starts, time_ends)

    exam_info = stir.ExamInfo()
    exam_info.set_time_frame_definitions(time_frame_def)

    return exam_info