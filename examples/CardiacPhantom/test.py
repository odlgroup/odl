#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:16:06 2017

@author: bgris
"""
import odl
import numpy as np
##%% Create data from lddmm registration
import matplotlib.pyplot as plt

f = open("SPECT_Torso_act_1.bin", "r")
a =np.fromfile(f, dtype=np.uint32)
a=a.reshape([256,256,120])




space = odl.uniform_discr(
    min_pt=[-127, -127,-57], max_pt=[127, 127,57], shape=[256, 256,120],
    dtype='float32', interp='linear')

#I0=space.element(a)

I0.show('I0',indices=np.s_[space.shape[0] // 2, :, :], aspect='equal')