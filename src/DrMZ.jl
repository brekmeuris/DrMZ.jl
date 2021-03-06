"""
This module contains the various libraries for generating data and training operator neural nets, finding the custom basis functions, solving PDEs using the custom basis functions, and construction of reduced order models.

Make sure the location of this file is specified in your Julia LOAD_PATH if you
are working locally.

Eventually will be public on GitHub

To add a location: push!(LOAD_PATH, "/Users/me/myjuliaprojects")

Add this to your startup file also, "~/.julia/config/startup.jl"
"""
module DrMZ

# Export the functions for General.jl
export error_test_sse, error_test_rel, periodic_fill_domain, periodic_fill_solution, solution_interpolation, reduced_initial_condition, mse_error, trapz, norm_rel_error, solution_spatial_sampling, fft_norm, ifft_norm, trapz1, simpson13, simpson38, simpson, average_error, gram_schmidt
export reduced_initial_condition_full

# Export the functions for OperatorNN.jl
export predict, loss_all, build_dense_model, build_branch_model, build_trunk_model, train_model, loss, exp_kernel_periodic, generate_periodic_functions, solution_extraction, generate_periodic_train_test, save_model, save_data, load_data, load_data_initial_conditions, build_trunk_model_layer_spec, load_data_train_test, generate_periodic_train_test_initial_conditions, load_data_initial_conditions, load_model
export generate_periodic_train_test_full, build_branch_model_reduced, build_trunk_model_reduced, nfan, ofeltype, epseltype, kaiming_uniform, glorot_uniform
export penalty, loss_all_MZ, train_model_MZ, train_model_MZ_coefficients

# Export the functions for PDESolve.jl
export advection_pde!, fourier_diff, fourier_two_diff, cheby_grid, cheby_diff_matrix, opnn_advection_pde!, advection_diffusion_pde!, opnn_advection_diffusion_pde!, generate_fourier_solution, kdv_pde!, quadratic_nonlinear, kdv_integrating_pde!, inviscid_burgers_pde!, generate_fourier_solution_stiff, kdv_implicit_pde!, kdv_explicit_pde!, viscous_burgers_explicit_pde!
export opnn_advection_pde_full!, opnn_advection_diffusion_pde_full!

# Export the functions for DBasis.jl
export basis_OpNN, orthonormal_check, build_basis, spectral_coefficients, spectral_coefficients_integral, spectral_approximation, spectral_matrix, generate_basis_solution, save_basis, spectral_truncation_error, spectral_redefined_error
export build_basis_factors

# Export the functions for Results.jl
# export generate_opnn_results

# Load required packages - only load functions that are used ##### Figure out how to add these if they don't exist...
using FFTW: fft, ifft
using Flux
using Flux.Data: DataLoader
using Flux: mse
using ProgressMeter: @showprogress
using Distributions: MvNormal
using LinearAlgebra: Symmetric, norm, eigmin, I, qr, diagm, svd, lq
using Random: randperm, AbstractRNG
using Random
using ColorSchemes # Cut this one down...
using DifferentialEquations
using Interpolations#: LinearInterpolation, interpolate
using BSON: @save
using BSON: @load
using Printf
using Plots; pyplot()
using ToeplitzMatrices
using Plots.PlotMeasures

# Load functions
include("General.jl")
include("OperatorNN.jl")
include("PDESolve.jl")
include("DBasis.jl")
# include("Results.jl")

end
