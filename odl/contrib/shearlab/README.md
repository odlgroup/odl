# Shearlab.jl

This package functionality for integrating ODL with the [julia API of Shearlab](https://github.com/arsenal9971/Shearlab.jl). This package requires different libraries installed, explained in the following.

## Dependencies and its installation.

- **Julia language**: One can either installed [precompiled packages](https://julialang.org/downloads/) or build from [source](https://github.com/JuliaLang/julia). This package uses the version 0.6 of julia.

- **Shearlab.jl**: To install the library in julia 0.6.x one needs to run the command `julia -e 'Pkg.add("Shearlab")`.

- **Pyjulia**: One can install the python API of julia by building from [source](https://github.com/JuliaPy/pyjulia), or using `pip install julia` for Python 2.x and `pip3 install julia` for Python 3.x.
   - One also needs to make the julia and python enviroment to coincide running the command `julia -e 'ENV["PYTHON"]="/usr/bin/python"; Pkg.add("PyCall"); Pkg.build("PyCall")'`. Where the directory of the python binary needs to be the binary of the enviroment you want to use.

- **SSL certificates**: Sometimes you need to give (and add to bashrc.) the SSL certificates path using `export SSL_CERT_FILE=/etc/ssl/ca-bundle.pem`.
