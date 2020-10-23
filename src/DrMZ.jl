"""
This module contains the various libraries for generating data and training operator neural nets, finding the custom basis functions, solving PDEs using the custom basis functions, and construction of reduced order models.

Make sure the location of this file is specified in your Julia LOAD_PATH is you
are working locally.

To add a location: push!(LOAD_PATH, "/Users/me/myjuliaprojects")

Add this to your startup file also, "~/.julia/config/startup.jl"
"""
module DrMZ

# Export the functions for OperatorNN.jl
export predict, loss_all

# Export the functions for PDESolve.jl
export advection_pde!

# Load required packages
# using FFTW

# Load functions
include("OperatorNN.jl")
include("PDESolve.jl")

end
