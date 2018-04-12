
# coding: utf-8

# # <center> Shearlab decomposition benchmarks </center>

# Some benchmarks comparing the performance of pure julia, python/julia and python implementation.

# Importing julia
import odl
import sys
sys.path.append('../')
import shearlab_operator
import pyshearlab
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import ceil
from odl.contrib.shearlab import shearlab_operator

# Calling julia
j = shearlab_operator.load_julia_with_Shearlab()


# Defining the parameters

n = 512
m = n
gpu = 0
square = 0
name = './lena.jpg';

nScales = 4;
shearLevels = [float(ceil(i/2)) for i in range(1,nScales+1)]
scalingFilter = 'Shearlab.filt_gen("scaling_shearlet")'
directionalFilter = 'Shearlab.filt_gen("directional_shearlet")'
waveletFilter = 'Shearlab.mirror(scalingFilter)'
scalingFilter2 = 'scalingFilter'
full = 0;

# Pure Julia
j.eval('X = Shearlab.load_image(name, n);');

# Read Data
j.eval('n = 512;')
# The path of the image
j.eval('name = "./lena.jpg";');
data = shearlab_operator.load_image(name,n);

sizeX = data.shape[0]
sizeY = data.shape[1]
rows = sizeX
cols = sizeY
X = data;

# ** Shearlet System generation **

# Pure julia
with odl.util.Timer('Shearlet System Generation julia'):
    j.eval('shearletSystem = Shearlab.getshearletsystem2D(n,n,4)');

# Python/Julia
with odl.util.Timer('Shearlet System Generation python/julia'):
    shearletSystem_jl = shearlab_operator.getshearletsystem2D(rows,cols,nScales,shearLevels,full,directionalFilter,scalingFilter);

# pyShearlab
with odl.util.Timer('Shearlet System Generation python'):
    shearletSystem_py = pyshearlab.SLgetShearletSystem2D(0,rows, cols, nScales)

# ** Coefficients computation **

# Pure Julia
with odl.util.Timer('Shearlet Coefficients Computation julia'):
    j.eval('coeffs = Shearlab.SLsheardec2D(X,shearletSystem);');

# Julia/Python
with odl.util.Timer('Shearlet Coefficients Computation python/julia'):
    coeffs_jl = shearlab_operator.sheardec2D(X,shearletSystem_jl)

# pyShearlab
with odl.util.Timer('Shearlet Coefficients Computation python'):
    coeffs_py = pyshearlab.SLsheardec2D(X, shearletSystem_py)


# ** Reconstruction **

# Pure Julia
with odl.util.Timer('Shearlet Reconstructon julia'):
    j.eval('Xrec=Shearlab.SLshearrec2D(coeffs,shearletSystem);');

# Julia/Python
with odl.util.Timer('Shearlet Reconstructon python/julia'):
    Xrec_jl = shearlab_operator.shearrec2D(coeffs_jl,shearletSystem_jl);

# pyShearlab
with odl.util.Timer('Shearlet Reconstructon python'):
    Xrec_py = pyshearlab.SLshearrec2D(coeffs_py, shearletSystem_py)