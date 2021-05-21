module DrMZ

# Load required packages
using FFTW: fft, ifft
using Flux
using Flux.Data: DataLoader
using Flux: mse
using ProgressMeter: @showprogress
using Distributions: MvNormal
using LinearAlgebra: Symmetric, norm, eigmin, I, qr, diagm, svd, lq, Diagonal
using Random: randperm, AbstractRNG
using Random
using DifferentialEquations
using Interpolations
using BSON: @save
using BSON: @load
using Printf
using ToeplitzMatrices
using Statistics: mean, std
using FastGaussQuadrature

# Export the functions for General.jl
export error_se, error_rel, mse_error, norm_rel_error, ic_error, average_ic_error, average_error,
       periodic_fill_domain, periodic_fill_solution, solution_interpolation, reduced_initial_condition, solution_spatial_sampling, solution_temporal_sampling,
       fft_norm, ifft_norm,
       fourier_diff, cheby_grid, cheby_diff_matrix,
       trapz, trapz1, simpson13, simpson38, simpson,
       shifted_nodes, gauss_quad

# Export the functions for OperatorNN.jl
export train_model, loss_all, predict, predict_min_max,
       build_dense_model, build_branch_model, build_trunk_model,
       exp_kernel_periodic, generate_sinusoidal_functions_2_parameter, generate_periodic_train_test, generate_periodic_functions, solution_extraction,
       save_model, load_model, save_data, load_data, load_data_initial_conditions, load_data_train_test, load_data_initial_conditions,
       min_max_scaler, min_max_transform, standard_scaler, standard_transform

# Export the functions for PDESolve.jl
export advection_pde!, advection_diffusion_pde!, inviscid_burgers_pde!, viscous_burgers_pde!,
       opnn_advection_pde!, opnn_advection_diffusion_pde!, opnn_inviscid_burgers_pde!, opnn_viscous_burgers_pde!,
       quadratic_nonlinear,
       quadratic_nonlinear_opnn, quadratic_nonlinear_opnn_pseudo,
       generate_fourier_solution,
       generate_basis_solution, generate_basis_solution_nonlinear,
       backward_upwind, forward_upwind, backward_upwind_limited, forward_upwind_limited, central_difference,
       van_leer_limiter, gradient_ratio_backward_j, gradient_ratio_backward_jneg, gradient_ratio_forward_j, gradient_ratio_forward_jpos,
       generate_bwlimitersoupwind_solution, generate_bwlimitersoupwind_viscous_solution,
       get_1D_energy_fft, get_1D_energy_basis

# Export the functions for DBasis.jl
export basis_OpNN, build_basis, build_basis_redefinition,
       spectral_coefficients, spectral_approximation, spectral_matrix,
       quadratic_nonlinear_triple_product,
       save_basis,
       orthonormal_check

# Load functions
include("General.jl")
include("OperatorNN.jl")
include("PDESolve.jl")
include("DBasis.jl")

end
