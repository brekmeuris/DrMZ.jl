"""
This module contains the various libraries for generating data and training operator neural nets, finding the custom basis functions, solving PDEs using the custom basis functions, and construction of reduced order models.

Make sure the location of this file is specified in your Julia LOAD_PATH if you
are working locally.

Eventually will be public on GitHub

To add a location: push!(LOAD_PATH, "/Users/me/myjuliaprojects")

Add this to your startup file also, "~/.julia/config/startup.jl"
"""
module DrMZ

# Export the functions for OperatorNN.jl
export predict, loss_all, build_branch_model, build_trunk_model, train_model

# Export the functions for PDESolve.jl
export advection_pde!

# Load required packages - only load functions that are used
using FFTW: fft, ifft
using Flux
using Flux.Data: DataLoader
using Flux: mse
using ProgressMeter: @showprogress

# Load functions
include("OperatorNN.jl")
include("PDESolve.jl")

end
