from typing import Dict
import time 

import numpy as np
import odl
import cProfile
import pstats
import io
pr = cProfile.Profile()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--kwd', required = True)
parser.add_argument('--reco_space_impl', required = True)
parser.add_argument('--ray_trafo_impl', required = True)
args = parser.parse_args()

metadata_name = args.kwd
kwd = args.kwd
dimension = 3
n_points  = 1024
reco_space_impl   = args.reco_space_impl
device_name = 'cuda:0'
ray_trafo_impl = args.ray_trafo_impl
n_angles = 360
# Create a space
if reco_space_impl == 'numpy':
    reco_space = odl.uniform_discr(
            min_pt = [-20 for _ in range(dimension)], 
            max_pt = [ 20 for _ in range(dimension)], 
            shape  = [n_points for _ in range(dimension)],
            dtype  = 'float32',
            impl   = reco_space_impl
        )
else:
    reco_space = odl.uniform_discr(
            min_pt = [-20 for _ in range(dimension)], 
            max_pt = [ 20 for _ in range(dimension)], 
            shape  = [n_points for _ in range(dimension)],
            dtype  = 'float32',
            impl   = reco_space_impl, 
            torch_device = device_name
        )

angle_partition = odl.uniform_partition(0, 2*np.pi, n_angles)
detector_partition = odl.uniform_partition(
    [-30, -30], 
    [30, 30], 
    [512, 512]
    )
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

ray_trafo = odl.tomo.RayTransform(
    reco_space, 
    geometry, 
    impl=ray_trafo_impl
    )

phantom = odl.phantom.shepp_logan(reco_space, modified=True)



if kwd == 'forward':
    pr.enable()
    proj_data = ray_trafo(phantom)
    # bk = ray_trafo.adjoint(proj_data)

    pr.disable()
elif kwd == 'backward':
    proj_data = ray_trafo(phantom)
    pr.enable()
    bk = ray_trafo.adjoint(proj_data)
    pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open(f'profiler_{kwd}_reco_space_{reco_space_impl}_ray_transformation_{ray_trafo_impl}.txt', 'w+') as f:
    f.write(s.getvalue())
