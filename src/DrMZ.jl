module DrMZ

# Load required packages
using FFTW: fft, ifft
using Flux
using Flux.Data: DataLoader
using Flux: mse
using ProgressMeter: @showprogress
using Distributions: MvNormal
using LinearAlgebra
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
using ForwardDiff
using LegendrePolynomials
using BSplines
using MAT

# Export the functions for General.jl
export error_se, error_rel, mse_error, norm_rel_error, norm_rel_error_continuous, norm_infinity_error, ic_error, average_ic_error, average_error,
       periodic_fill_domain, periodic_fill_solution, solution_interpolation, reduced_initial_condition, solution_spatial_sampling, solution_temporal_sampling,
       fft_norm, ifft_norm,
       fourier_diff,
       trapz,
       gauss_legendre,
       clenshaw_curtis, cheby_grid, cheby_diff_matrix, cheby_diff,
       trapezoid,
       orthonormal_check,
       legendre_shift, legendre_norm, dlegendre_norm, legendre_norm_basis_build, dlegendre_norm_basis_build,
       linear_reg,
       data_extract_matlab

# Export the functions for OperatorNN.jl
export train_model, loss_all, predict,
       build_dense_model, build_branch_model, build_trunk_model,
       exp_kernel_periodic, generate_sinusoidal_functions_2_parameter, generate_periodic_train_test_initial_conditions, generate_periodic_train_test, generate_periodic_train_test_muscl, generate_periodic_train_test_esdirk, generate_periodic_train_test_implicit, generate_periodic_functions, solution_extraction,
       save_model, load_model, load_branch, load_trunk, save_data, load_data, load_data_initial_conditions, load_data_train_test, load_data_initial_conditions, save_data_initial_conditions,
       feature_expansion_single, feature_expansion_set, feature_expansion_set_x

# Export the functions for PDESolve.jl
export advection_pde!, advection_diffusion_pde!, inviscid_burgers_pde!, viscous_burgers_pde!, kdv_explicit_pde!, kdv_implicit_pde!, kdv_pde!, ks_explicit_pde!, ks_implicit_pde!, ks_pde!, rhs_advection!, rhs_advection_diffusion!, burgers_flux, quadratic_nonlinear_triple_product_basis, quadratic_nonlinear_basis, rhs_viscous_burgers!, rhs_inviscid_burgers!, rhs_explicit_kdv!, rhs_implicit_kdv!, rhs_kdv!, rhs_explicit_ks!, rhs_implicit_ks!, rhs_ks!,
       quadratic_nonlinear,
       generate_fourier_solution, generate_fourier_solution_esdirk, generate_fourier_solution_implicit, generate_basis_solution, generate_basis_solution_esdirk, generate_basis_solution_implicit,
       central_difference,
       minmod, ub, fl, ulpl, urpl, ulnl, urnl,
       muscl_minmod_RHS!, muscl_minmod_viscous_RHS!, generate_muscl_minmod_solution, generate_muscl_reduced,
       get_1D_energy_fft,
       get_1D_energy_custom, get_1D_energy_custom_coefficients,
       mode_extractor, get_1D_energy_upwind,
       spectral_approximation_fourier,
       rhs_advection_galerkin!, rhs_advection_diffusion_galerkin!, quadratic_nonlinear_triple_product_basis_galerkin, quadratic_nonlinear_basis_galerkin, rhs_viscous_burgers_galerkin!, rhs_inviscid_burgers_galerkin!, rhs_kdv_galerkin!, rhs_ks_galerkin!,
       generate_fourier_solution_real, generate_fourier_solution_esdirk_real, generate_fourier_solution_implicit_real,
       advection_pde_real!, advection_diffusion_pde_real!, viscous_burgers_pde_real!, quadratic_nonlinear_real, kdv_pde_real!, ks_pde_real!,
       quadratic_nonlinear_basis_pseudo, rhs_viscous_burgers_pseudo!, rhs_inviscid_burgers_pseudo!, rhs_kdv_pseudo!,rhs_ks_pseudo!,
       rhs_advection_diffusion_dirichlet!, generate_legendre_basis_dirichlet_advection_diffusion

# Export the functions for DBasis.jl
export trunk_build, basis_interpolate, dbasis_interpolate, trunk_ortho_build, build_basis,
       basis_eval, expansion_coefficients, expansion_approximation,
       basis_derivative, 
       save_basis, load_basis,
       trunk_ortho_build_expand, build_basis_expand

# Load functions
include("General.jl")
include("OperatorNN.jl")
include("PDESolve.jl")
include("DBasis.jl")

end
