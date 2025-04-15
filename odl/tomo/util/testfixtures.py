from __future__ import division
import numpy as np
import pytest
import odl
from odl.tomo.util.testutils import skip_if_no_astra
from odl.util.testutils import simple_fixture

PARALLEL_2D_PROJECTORS_CPU = [
    'parallel 2 astra_cpu numpy uniform None',
    'parallel 2 astra_cpu numpy nonuniform None',
    'parallel 2 astra_cpu numpy random None',
    'parallel 2 astra_cpu numpy uniform None',
    'parallel 2 astra_cpu numpy nonuniform None',
    'parallel 2 astra_cpu numpy random None',
    'cone 2 astra_cpu numpy uniform None',
    'cone 2 astra_cpu numpy nonuniform None',
    'cone 2 astra_cpu numpy random None',
    'cone 2 astra_cpu numpy uniform None',
    'cone 2 astra_cpu numpy nonuniform None',
    'cone 2 astra_cpu numpy random None',
    ]

PARALLEL_2D_PROJECTORS = [
    'parallel 2 astra_cuda numpy uniform None',
    'parallel 2 astra_cuda numpy nonuniform None',
    'parallel 2 astra_cuda numpy random None',
    'parallel 2 astra_cuda pytorch uniform cpu',
    'parallel 2 astra_cuda pytorch nonuniform cpu',
    'parallel 2 astra_cuda pytorch random cpu',
    'parallel 2 astra_cuda pytorch uniform cuda:0',
    'parallel 2 astra_cuda pytorch nonuniform cuda:0',
    'parallel 2 astra_cuda pytorch random cuda:0'
    ]

CONE_2D_PROJECTORS = [
    'cone 2 astra_cuda numpy uniform None',
    'cone 2 astra_cuda numpy nonuniform None',
    'cone 2 astra_cuda numpy random None',
    'cone 2 astra_cuda pytorch uniform cpu',
    'cone 2 astra_cuda pytorch nonuniform cpu',
    'cone 2 astra_cuda pytorch random cpu',
    'cone 2 astra_cuda pytorch uniform cuda:0',
    'cone 2 astra_cuda pytorch nonuniform cuda:0',
    'cone 2 astra_cuda pytorch random cuda:0'
    ]

PARALLEL_3D_PROJECTORS = [
    'parallel 3 astra_cuda numpy uniform None',
    'parallel 3 astra_cuda numpy nonuniform None',
    'parallel 3 astra_cuda numpy random None',
    'parallel 3 astra_cuda pytorch uniform cuda:0',
    'parallel 3 astra_cuda pytorch nonuniform cuda:0',
    'parallel 3 astra_cuda pytorch random cuda:0'
    ]

CONE_3D_PROJECTORS = [
    'cone 3 astra_cuda numpy uniform None',
    'cone 3 astra_cuda numpy nonuniform None',
    'cone 3 astra_cuda numpy random None',
    'cone 3 astra_cuda pytorch uniform cuda:0',
    'cone 3 astra_cuda pytorch nonuniform cuda:0',
    'cone 3 astra_cuda pytorch random cuda:0'
    ]

HELICAL_PROJECTORS = [
    'helical 3 astra_cuda numpy uniform None',
    'helical 3 astra_cuda pytorch uniform cuda:0'
]

SKIMAGE_PROJECTORS = [
    'parallel 2 skimage numpy uniform None',
    'parallel 2 skimage numpy half_uniform None'
]

def parse_angular_partition(
        angles_name:str, 
        n_angles:int
        ):
    if angles_name == 'uniform':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
    elif angles_name == 'half_uniform':
        apart = odl.uniform_partition(0, np.pi, n_angles)
    elif angles_name == 'random':
        # Linearly spaced with random noise
        min_pt = 2 * (2.0 * np.pi) / n_angles
        max_pt = (2.0 * np.pi) - 2 * (2.0 * np.pi) / n_angles
        points = np.linspace(min_pt, max_pt, n_angles)
        points += np.random.rand(n_angles) * (max_pt - min_pt) / (5 * n_angles)
        apart = odl.nonuniform_partition(points)
    elif angles_name == 'nonuniform':
        # Angles spaced quadratically
        min_pt = 2 * (2.0 * np.pi) / n_angles
        max_pt = (2.0 * np.pi) - 2 * (2.0 * np.pi) / n_angles
        points = np.linspace(min_pt ** 0.5, max_pt ** 0.5, n_angles) ** 2
        apart = odl.nonuniform_partition(points)
    else:
        raise ValueError('angle not valid')
    return apart

def parse_geometry(
        geometry_name :str,
        dimension :int,
        angular_partition,
        detector_partition,
        reconstruction_space,
        ray_trafo_impl
    ):
    if geometry_name == 'parallel':
        if dimension == 2:
            geometry = odl.tomo.Parallel2dGeometry(
                angular_partition, detector_partition)
        elif dimension == 3 : 
            geometry = odl.tomo.Parallel3dAxisGeometry(
                angular_partition, detector_partition)
        else:
            raise(ValueError)

    elif geometry_name == 'cone':
        if dimension == 2:
            geometry = odl.tomo.FanBeamGeometry(
                angular_partition, detector_partition, 
                src_radius=200,det_radius=100
                )
        elif dimension == 3 : 
            geometry = odl.tomo.ConeBeamGeometry(
                angular_partition, detector_partition, 
                src_radius=200,det_radius=100
                )
        else:
            raise(ValueError)

    elif geometry_name == 'helical':
        assert dimension == 3
        geometry = odl.tomo.ConeBeamGeometry(
            angular_partition, detector_partition, pitch=5,
            src_radius=200, det_radius=100
            )

    else:
        raise ValueError('geom not valid')
    
    return odl.tomo.RayTransform(
        reconstruction_space, 
        geometry, 
        impl=ray_trafo_impl
        )

#### FIXTURES #### 
def projector(request):
    n = 100
    m = 100
    n_angles = 100
    dtype = 'float32'

    geometry_name, dimension, ray_trafo_impl, reco_space_impl, angles_keyword, device = request.param.split()

    dimension = int(dimension)

    if reco_space_impl == 'numpy':
        assert device =='None' 

    angular_partition = parse_angular_partition(angles_keyword, n_angles)
    
    reconstruction_space = odl.uniform_discr(
            [-20] * dimension, 
            [20] * dimension, 
            [n] * dimension,
            dtype=dtype, impl=reco_space_impl
        )
    detector_partition = odl.uniform_partition(
        [-30] * (dimension-1), 
        [30] * (dimension-1), 
        [m] * (dimension-1)
        )

    return parse_geometry(
        geometry_name,
        dimension,
        angular_partition,
        detector_partition,
        reconstruction_space,
        ray_trafo_impl
    )

@pytest.fixture(scope='module',
                params=[True, False],
                ids=[' in-place ', ' out-of-place '])
def in_place(request):
    return request.param

ray_trafo_impl = simple_fixture(
    name='ray_trafo_impl',
    params=[
        pytest.param('astra_cuda', marks=skip_if_no_astra),
        # pytest.param('astra_cpu', marks=skip_if_no_astra),
        # pytest.param('skimage', marks=skip_if_no_skimage)
        ]
)

geometry_params = ['par2d', 'par3d', 'cone2d', 'cone3d', 'helical']
geometry_ids = [" geometry='{}' ".format(p) for p in geometry_params]


@pytest.fixture(scope='module', ids=geometry_ids, params=geometry_params)
def geometry(request):
    geom = request.param
    m = 100
    n_angles = 100

    if geom == 'par2d':
        apart = odl.uniform_partition(0, np.pi, n_angles)
        dpart = odl.uniform_partition(-30, 30, m)
        return odl.tomo.Parallel2dGeometry(apart, dpart)
    elif geom == 'par3d':
        apart = odl.uniform_partition(0, np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -30], [30, 30], (m, m))
        return odl.tomo.Parallel3dAxisGeometry(apart, dpart)
    elif geom == 'cone2d':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
        dpart = odl.uniform_partition(-30, 30, m)
        return odl.tomo.FanBeamGeometry(apart, dpart, src_radius=200,
                                        det_radius=100)
    elif geom == 'cone3d':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-60, -60], [60, 60], (m, m))
        return odl.tomo.ConeBeamGeometry(apart, dpart,
                                         src_radius=200, det_radius=100)
    elif geom == 'helical':
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], (m, m))
        return odl.tomo.ConeBeamGeometry(apart, dpart, pitch=5.0,
                                         src_radius=200, det_radius=100)
    else:
        raise ValueError('geom not valid')

consistent_geometry_params = ['par2d', 'cone2d']
consistent_geometry_ids = [" geometry='{}' ".format(p) for p in consistent_geometry_params]

@pytest.fixture(scope='module', ids=consistent_geometry_ids, params=consistent_geometry_params)
def consistent_geometry(request):
    geom = request.param
    m = 100
    n_angles = 100

    if geom == 'par2d':
        apart = odl.uniform_partition(0, np.pi, n_angles)
        dpart = odl.uniform_partition(-30, 30, m)
        return odl.tomo.Parallel2dGeometry(apart, dpart)
    elif geom == 'cone2d':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
        dpart = odl.uniform_partition(-30, 30, m)
        return odl.tomo.FanBeamGeometry(apart, dpart, src_radius=200,
                                        det_radius=100)
    else:
        raise ValueError('geom not valid')

geometry_type = simple_fixture(
    'geometry_type',
    ['par2d', 'par3d', 'cone2d', 'cone3d']
)
