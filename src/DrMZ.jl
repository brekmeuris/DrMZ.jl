module DrMZ

# Export the functions for General.jl
export error_test_sse, error_test_rel, periodic_fill_domain, periodic_fill_solution, solution_interpolation, reduced_initial_condition, mse_error, trapz, norm_rel_error, solution_spatial_sampling, fft_norm, ifft_norm, trapz1, simpson13, simpson38, simpson, average_error, gram_schmidt, ic_error, average_ic_error, solution_temporal_sampling
export reduced_initial_condition_full

# Export the functions for OperatorNN.jl
export predict, loss_all, build_dense_model, build_branch_model, build_trunk_model, train_model, loss, exp_kernel_periodic, generate_periodic_functions, solution_extraction, generate_periodic_train_test, save_model, save_data, load_data, load_data_initial_conditions, build_trunk_model_layer_spec, load_data_train_test, generate_periodic_train_test_initial_conditions, load_data_initial_conditions, load_model, min_max_scaler, min_max_transform, standard_scaler, standard_transform, predict_min_max, generate_sinusoidal_functions_2_parameter
export generate_periodic_train_test_full, build_branch_model_reduced, build_trunk_model_reduced, nfan, ofeltype, epseltype, kaiming_uniform, glorot_uniform
export penalty, loss_all_MZ, train_model_MZ, train_model_MZ_coefficients

# Export the functions for PDESolve.jl
export advection_pde!, fourier_diff, cheby_grid, cheby_diff_matrix, opnn_advection_pde!, advection_diffusion_pde!, opnn_advection_diffusion_pde!, generate_fourier_solution, kdv_pde!, quadratic_nonlinear, kdv_integrating_pde!, inviscid_burgers_pde!, generate_fourier_solution_IMEX, kdv_implicit_pde!, kdv_explicit_pde!, generate_fourier_solution_split, quadratic_nonlinear_pde!, viscous_burgers_pde!, opnn_viscous_burgers_pde!, generate_basis_solution, quadratic_nonlinear_opnn, generate_basis_solution_nonlinear, quadratic_nonlinear_opnn_pseudo, opnn_inviscid_burgers_pde!, backward_upwind, forward_upwind, van_leer_limiter, gradient_ratio_backward_j, gradient_ratio_backward_jneg, gradient_ratio_forward_j, gradient_ratio_forward_jpos, backward_upwind_limited, forward_upwind_limited, generate_bwlimitersoupwind_solution
export opnn_advection_pde_full!, opnn_advection_diffusion_pde_full!, second_derivative_opnn

# Export the functions for DBasis.jl
export basis_OpNN, orthonormal_check, build_basis, spectral_coefficients, spectral_coefficients_integral, spectral_approximation, spectral_matrix, save_basis, spectral_truncation_error, spectral_redefined_error, quadratic_nonlinear_triple_product
export build_basis_factors, second_derivative_product

# Export the functions for Results.jl
# export generate_opnn_results

# Load required packages - only load functions that are used ##### Figure out how to add these if they don't exist...
# To Do: Add all of these packages as dependencies...
using FFTW: fft, ifft
using Flux
using Flux.Data: DataLoader
using Flux: mse
using ProgressMeter: @showprogress
using Distributions: MvNormal
using LinearAlgebra: Symmetric, norm, eigmin, I, qr, diagm, svd, lq, Diagonal
using Random: randperm, AbstractRNG
using Random
using ColorSchemes # Cut this one down...
using DifferentialEquations
using Interpolations#: LinearInterpolation, interpolate
using BSON: @save
using BSON: @load
using Printf
using Plots; pyplot() # Is this needed?
using ToeplitzMatrices
using Plots.PlotMeasures
using Statistics: mean, std

# Load functions
include("General.jl")
include("OperatorNN.jl")
include("PDESolve.jl")
include("DBasis.jl")
# include("Results.jl")

end
