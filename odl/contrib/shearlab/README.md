# Shearlab.jl
**Original contribution by:** [@arsenal9971](https://github.com/arsenal9971)

This package adds functionality for integrating ODL with the [Julia API of Shearlab](https://github.com/arsenal9971/Shearlab.jl). This package requires different libraries installed, explained in the following.

## Dependencies and its installation.

- **Julia language**: One can either [precompiled packages](https://julialang.org/downloads/) or build from [source](https://github.com/JuliaLang/julia). This package requires the Julia version 0.6 or higher.

- **Shearlab.jl**: To install the library in Julia 0.6.x one needs to run the command `julia -e 'Pkg.add("Shearlab")`.

- **Pyjulia**: One can install the Python API of Julia with the command `pip install julia`, for more details on installation check the [documentation](https://odlgroup.github.io/odl/getting_started/installing.html).
   - One also needs to make the Julia and Python enviroment to coincide running the command `julia -e 'ENV["PYTHON"]="<your-python-executable>"; Pkg.add("PyCall"); Pkg.build("PyCall")'`. One can find its python executable path by running on the terminal `$(which python)`.

- **SSL certificates**: Sometimes you need to give (and add to bashrc.) the SSL certificates path using `export SSL_CERT_FILE=/etc/ssl/ca-bundle.pem`.
