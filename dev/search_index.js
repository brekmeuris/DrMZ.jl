var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = DrMZ","category":"page"},{"location":"#DrMZ","page":"Home","title":"DrMZ","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [DrMZ]","category":"page"},{"location":"#DrMZ.advection_diffusion_pde!-NTuple{4,Any}","page":"Home","title":"DrMZ.advection_diffusion_pde!","text":"advection_diffusion_pde!(duhat,uhat,p,t)\n\nRHS for the advection-diffusion equation u_t = - u_x + ν u_xx for numerical integration in Fourier space where ν is the viscosity.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.advection_pde!-NTuple{4,Any}","page":"Home","title":"DrMZ.advection_pde!","text":"advection_pde!(duhat,uhat,p,t)\n\nRHS for the advection equation u_t = - u_x for numerical integration in Fourier space.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.average_error-Tuple{Any,Any}","page":"Home","title":"DrMZ.average_error","text":"average_error(domain,error)\n\nCompute the average error using the trapezoid rule frac1T int_0^T error(t) dt.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.average_ic_error-Tuple{Any,Any}","page":"Home","title":"DrMZ.average_ic_error","text":"average_ic_error(target,prediction)\n\nCompute the two-norm relative error between the prediction and target values for an initial condition.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.backward_upwind-NTuple{4,Any}","page":"Home","title":"DrMZ.backward_upwind","text":"function backward_upwind(u_j,u_jneg,u_jnegg,nu)\n\nCompute the partial RHS (u_j^n+1 = u_j^n + backward_upwind) of the second order backward upwind scheme of Beam and Warming for the inviscid Burgers equation u_t = -u u_x (split into monotone and non-montone terms), -nu(u_j^n-u_j-1^n) - fracnu2(1-nu)(u_j^n-u_j-1^n) + fracnu2(1-nu)(u_j-1^n-u_j-2^n). nu is the CFL condition.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.backward_upwind_limited-NTuple{5,Any}","page":"Home","title":"DrMZ.backward_upwind_limited","text":"function backward_upwind_limited(u_j,u_jneg,u_jnegg,u_jpos,nu)\n\nCompute the partial RHS (u_j^n+1 = u_j^n + backward_upwind) of the second order limited backward upwind scheme of Beam and Warming for the inviscid Burgers equation u_t = -u u_x (split into monotone and non-montone terms), -nu(u_j^n-u_j-1^n) - fracnu2(1-nu)(u_j^n-u_j-1^n) + fracnu2(1-nu)(u_j-1^n-u_j-2^n). nu is the CFL condition.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.basis_OpNN-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.basis_OpNN","text":"function basis_OpNN(trunk,x_locations)\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.build_basis-NTuple{4,Any}","page":"Home","title":"DrMZ.build_basis","text":"build_basis(trunk,intial_condition,x_locations,opnn_output_width,initial_condition)\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.build_basis_redefinition-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.build_basis_redefinition","text":"build_basis_redefinition(basis_full,x_redefined,x_full)\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.build_dense_model-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.build_dense_model","text":"function build_dense_model(number_layers,neurons,activations)\n\nBuild a feedforward neural network (FFNN) consisting of number_layers of Flux dense layers for the specified number of neurons and activations.\n\nExamples\n\njulia> build_dense_model(2,[(128,128),(128,128)],[relu,relu])\nChain(Dense(128, 128, NNlib.relu), Dense(128, 128, NNlib.relu))\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.central_difference-NTuple{4,Any}","page":"Home","title":"DrMZ.central_difference","text":"function central_difference(u_j,u_jpos,u_jneg,mu)\n\nCompute the second order central difference for the viscous term of the viscous Burgers equation u_t = - u u_x + ν u_xx. mu is equal to $ \\frac{\\nu *\\Delta t}{\\Delta x^2}$.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.cheby_diff-NTuple{4,Any}","page":"Home","title":"DrMZ.cheby_diff","text":"cheby_diff(sol,N,dL)\n\nCompute the derivative using a Chebyshev differentiation matrix on the interval ab with N discretization points.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.cheby_diff_matrix-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.cheby_diff_matrix","text":"cheby_diff_matrix(N,a,b)\n\nGenerate the Chebyshev differentiation matrix for the interval ab with N discretization points.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.cheby_grid-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.cheby_grid","text":"cheby_grid(N,a,b)\n\nGenerate the grid of Chebyshev points on the interval ab with N discretization points.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.error_rel-Tuple{Any,Any}","page":"Home","title":"DrMZ.error_rel","text":"error_rel(target,prediction)\n\nCompute the relative error between the prediction and target values.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.error_se-Tuple{Any,Any}","page":"Home","title":"DrMZ.error_se","text":"error_se(target,prediction)\n\nCompute the squared error between the prediction and target values.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.exp_kernel_periodic-Tuple{Any,Any}","page":"Home","title":"DrMZ.exp_kernel_periodic","text":"function exp_kernel_periodic(x_locations;length_scale=0.5)\n\nCovariance kernel for radial basis function (GRF) and periodic IC f(sin^2(πx)).\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.fft_norm-Tuple{Any}","page":"Home","title":"DrMZ.fft_norm","text":"fft_norm(solution)\n\nCompute the FFT normalized by frac1N.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.forward_upwind-NTuple{4,Any}","page":"Home","title":"DrMZ.forward_upwind","text":"function forward_upwind(u_j,u_jpos,u_jposs,nu)\n\nCompute the partial RHS (u_j^n+1 = u_j^n + forward_upwind) of the second order forward upwind scheme of Beam and Warming for the inviscid Burgers equation u_t = -u u_x (split into monotone and non-montone terms), -nu(u_j+1^n-u_j^n) - fracnu2(nu+1)(u_j+1^n-u_j^n) + fracnu2(nu+1)(u_j+2^n-u_j+1^n). nu is the CFL condition.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.forward_upwind_limited-NTuple{5,Any}","page":"Home","title":"DrMZ.forward_upwind_limited","text":"function forward_upwind_limited(u_j,u_jpos,u_jposs,u_jneg,nu)\n\nCompute the partial RHS (u_j^n+1 = u_j^n + forward_upwind) of the second order limited forward upwind scheme of Beam and Warming for the inviscid Burgers equation u_t = -u u_x (split into monotone and non-montone terms), -nu(u_j+1^n-u_j^n) - fracnu2(nu+1)(u_j+1^n-u_j^n) + fracnu2(nu+1)(u_j+2^n-u_j+1^n). nu is the CFL condition.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.fourier_diff-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.fourier_diff","text":"fourier_diff(sol,N,dL;format=\"matrix\")\n\nCompute the derivative using a Fourier differentiation matrix (default) or the spectral derivative for periodic functions for domain length dL and with N discretization points.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.gauss_quad-NTuple{4,Any}","page":"Home","title":"DrMZ.gauss_quad","text":"function gauss_quad(a,b,func,number_points)\n\nCompute the integral using Gauss-Legendre quadrature for the interval int_a^b for a given func using a specified number_points.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.generate_basis_solution-NTuple{7,Any}","page":"Home","title":"DrMZ.generate_basis_solution","text":"generate_basis_solution(L1,L2,t_span,N,basis,initial_condition,pde_function;dt=1e-3,rtol=1e-10,atol=1e-14)\n\nGenerate the solution for a given linear pde_function and initial_conditon on a periodic domain using a N mode custom basis expansion.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.generate_basis_solution_nonlinear-NTuple{7,Any}","page":"Home","title":"DrMZ.generate_basis_solution_nonlinear","text":"generate_basis_solution_nonlinear(L1,L2,t_span,N,basis,initial_condition,pde_function;dt=1e-4,rtol=1e-10,atol=1e-14)\n\nGenerate the solution for a given non-linear pde_function and initial_conditon on a periodic domain using a N mode custom basis expansion.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.generate_bwlimitersoupwind_solution-NTuple{5,Any}","page":"Home","title":"DrMZ.generate_bwlimitersoupwind_solution","text":"function generate_bwlimitersoupwind_solution(L1,L2,t_end,N,initial_condition;dt=1e-4)\n\nGenerate second order limited upwind solution for the inviscid Burgers equation u_t = -u u_x based on the Beam and Warming scheme with a Van Leer limiter.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.generate_bwlimitersoupwind_viscous_solution-NTuple{5,Any}","page":"Home","title":"DrMZ.generate_bwlimitersoupwind_viscous_solution","text":"function generate_bwlimitersoupwind_viscous_solution(L1,L2,t_end,N,initial_condition;dt=1e-4,nu=0.1)\n\nGenerate second order solution for the viscous Burgers equation u_t = - u u_x + ν u_xx based on the Beam and Warming second order upwind with a Van Leer limiter for the convective term and a second order central differnce for the diffusive term. nu is the viscosity.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.generate_fourier_solution-NTuple{6,Any}","page":"Home","title":"DrMZ.generate_fourier_solution","text":"generate_fourier_solution(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-3,,nu=0.1,rtol=1e-10,atol=1e-14)\n\nGenerate the solution for a given pde_function and initial_condition on a periodic domain using a N mode Fourier expansion.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.generate_periodic_functions-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.generate_periodic_functions","text":"generate_periodic_functions(x_locations,number_functions,length_scale)\n\nGenerate a specified number_functions of random periodic functions using the exp_kernel_periodic function and a multivariate distribution.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.generate_periodic_train_test-NTuple{8,Any}","page":"Home","title":"DrMZ.generate_periodic_train_test","text":"generate_periodic_train_test(L1,L2,t_span,number_sensors,number_test_functions,number_train_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt=1e-3,domain=\"periodic\"))\n\nGenerate the training and testing data for a specified pde_function_handle for periodic boundary conditions using a Fourier spectral method.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.generate_sinusoidal_functions_2_parameter-Tuple{Any,Any}","page":"Home","title":"DrMZ.generate_sinusoidal_functions_2_parameter","text":"generate_sinusoidal_functions_2_parameter(x_locations,number_functions)\n\nGenerate a specified number_functions of random periodic functions for the distribution α sin(x)+β for α  -11 and β  -11.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.gradient_ratio_backward_j-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.gradient_ratio_backward_j","text":"function gradient_ratio_backward_j(u_j,u_jneg,u_jpos)\n\nCompute the backward gradient ratio at point i as required for the van_leer_limiter.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.gradient_ratio_backward_jneg-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.gradient_ratio_backward_jneg","text":"function gradient_ratio_backward_jneg(u_j,u_jneg,u_jnegg)\n\nCompute the backward gradient ratio at point i-1 as required for the van_leer_limiter.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.gradient_ratio_forward_j-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.gradient_ratio_forward_j","text":"function gradient_ratio_forward_j(u_j,u_jneg,u_jpos)\n\nCompute the forward gradient ratio at point i as required for the van_leer_limiter.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.gradient_ratio_forward_jpos-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.gradient_ratio_forward_jpos","text":"function gradient_ratio_forward_jpos(u_j,u_jpos,u_jposs)\n\nCompute the forward gradient ratio at point i+1 as required for the van_leer_limiter.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.ic_error-Tuple{Any,Any}","page":"Home","title":"DrMZ.ic_error","text":"ic_error(target,prediction)\n\nCompute the relative error between the prediction and target values for an initial condition. If the target = 0, both the prediction and target are augmented by epsilon_machine.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.ifft_norm-Tuple{Any}","page":"Home","title":"DrMZ.ifft_norm","text":"ifft_norm(solution)\n\nCompute the IFFT normalized by N.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.inviscid_burgers_pde!-NTuple{4,Any}","page":"Home","title":"DrMZ.inviscid_burgers_pde!","text":"inviscid_burgers_pde!(duhat,uhat,p,t)\n\nRHS for the inviscid Burgers equation u_t = - u u_x for numerical integration in Fourier space.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.load_data-NTuple{4,Any}","page":"Home","title":"DrMZ.load_data","text":"load_data(n_epoch,number_train_functions,number_test_functions,pde_function)\n\nLoad the trained branch and trunk neural networks along with the train_data and test_data.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.load_data_initial_conditions-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.load_data_initial_conditions","text":"load_data_initial_conditions(number_train_functions,number_test_functions)\n\nLoad the initial conditions from the train_data and test_data.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.load_data_train_test-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.load_data_train_test","text":"load_data_train_test(number_train_functions,number_test_functions,pde_function)\n\nLoad the train_data and test_data.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.load_model-Tuple{Any,Any}","page":"Home","title":"DrMZ.load_model","text":"load_model(n_epoch,pde_function)\n\nLoad the trained branch and trunk neural networks.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.loss_all-NTuple{5,Any}","page":"Home","title":"DrMZ.loss_all","text":"loss_all(branch,trunk,initial_conditon,solution_location,target_value)\n\nCompute the mean squared error (MSE) for a complete dataset.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.min_max_scaler-Tuple{Any}","page":"Home","title":"DrMZ.min_max_scaler","text":"min_max_scaler(x;dim = 2)\n\nCompute the required scaler to shift the features to a given range.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.min_max_transform-Tuple{Any,Any}","page":"Home","title":"DrMZ.min_max_transform","text":"min_max_transform(x,scale_object;min = 0,max = 1)\n\nApply min_max_scaler to shift the features.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.mse_error-Tuple{Any,Any}","page":"Home","title":"DrMZ.mse_error","text":"mse_error(target,prediction)\n\nCompute the mean squared error between the prediction and target values.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.norm_rel_error-Tuple{Any,Any}","page":"Home","title":"DrMZ.norm_rel_error","text":"norm_rel_error(target,prediction)\n\nCompute the two-norm relative error between the prediction and target values.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.opnn_advection_diffusion_pde!-NTuple{4,Any}","page":"Home","title":"DrMZ.opnn_advection_diffusion_pde!","text":"opnn_advection_diffusion_pde!(du,u,p,t)\n\nRHS for the advection-diffusion equation u_t = - u_x + ν u_xx for numerical integration in the custom basis space where ν is the viscosity.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.opnn_advection_pde!-NTuple{4,Any}","page":"Home","title":"DrMZ.opnn_advection_pde!","text":"opnn_advection_pde!(du,u,p,t)\n\nRHS for the advection equation u_t = - u_x for numerical integration in the custom basis space.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.opnn_inviscid_burgers_pde!-NTuple{4,Any}","page":"Home","title":"DrMZ.opnn_inviscid_burgers_pde!","text":"opnn_inviscid_burgers_pde!(du,u,p,t)\n\nRHS for the inviscid Burgers equation u_t = - u u_x for numerical integration in the custom basis space.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.opnn_viscous_burgers_pde!-NTuple{4,Any}","page":"Home","title":"DrMZ.opnn_viscous_burgers_pde!","text":"opnn_viscous_burgers_pde!(du,u,p,t)\n\nRHS for the viscous Burgers equation u_t = - u u_x + ν u_xx for numerical integration in the custom basis space where ν is the viscosity.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.orthonormal_check-Tuple{Any}","page":"Home","title":"DrMZ.orthonormal_check","text":"orthonormal_check(basis;tol = 1e-15)\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.periodic_fill_domain-Tuple{Any}","page":"Home","title":"DrMZ.periodic_fill_domain","text":"periodic_fill_domain(x_locations)\n\nOutput the full domain from periodic domain specified for x_locations.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.periodic_fill_solution-Tuple{Any}","page":"Home","title":"DrMZ.periodic_fill_solution","text":"periodic_fill_solution(solution)\n\nOutput the full u(tx) solution from periodic solution.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.predict-NTuple{5,Any}","page":"Home","title":"DrMZ.predict","text":"predict(branch,trunk,initial_condition,x_locations,t_values)\n\nPredict solution u(tx) at specified output locations using trained operator neural network branch and trunk.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.predict_min_max-NTuple{6,Any}","page":"Home","title":"DrMZ.predict_min_max","text":"predict_min_max(branch,trunk,initial_condition,x_locations,t_values,scale_object)\n\nUses the trained operator neural net branch and trunk to predict solution at specified output locations when using min-max normalization\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.quadratic_nonlinear-NTuple{4,Any}","page":"Home","title":"DrMZ.quadratic_nonlinear","text":"quadratic_nonlinear!(uhat,N,dL,alpha)\n\nCompute the convolution sum fracik2sum_p+q=k u_p u_q resulting from the quadratic nonlinearity of Burgers equation u u_x in Fourier space. Convolution sum is padded with the 3/2 rule for dealiasing.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.quadratic_nonlinear_opnn-Tuple{Any,Any}","page":"Home","title":"DrMZ.quadratic_nonlinear_opnn","text":"quadratic_nonlinear_opnn(uhat,nonlinear_triple)\n\nCompute the triple product sum sum_k=1^N sum_l=1^N u_k u_l sum_j=0^N-1 phi_jk phi_jl^ phi_jm^* resulting from the quadratic nonlinearity of Burgers equation u u_x in custom basis space.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.quadratic_nonlinear_opnn_pseudo-NTuple{4,Any}","page":"Home","title":"DrMZ.quadratic_nonlinear_opnn_pseudo","text":"quadratic_nonlinear_opnn_pseudo(basis,uhat,dL,N)\n\nCompute the quadratic nonlinearity of Burgers equation u u_x in custom basis space using a pseudo spectral type approach. At each step, transform solution to real space and compute the product of the real space solution and the spatial derivative of the real space solution prior to transforming back to custom basis space.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.quadratic_nonlinear_triple_product-Tuple{Any,Any}","page":"Home","title":"DrMZ.quadratic_nonlinear_triple_product","text":"quadratic_nonlinear_triple_product(basis,Dbasis)\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.reduced_initial_condition-NTuple{5,Any}","page":"Home","title":"DrMZ.reduced_initial_condition","text":"reduced_initial_condition(L1,L2,N,x_locations,initial_condition)\n\nExtract the x locations and intial condition values u(x) at a reduced number of equally spaced spatial locations.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.save_basis-NTuple{4,Any}","page":"Home","title":"DrMZ.save_basis","text":"save_basis(basis,N,n_epoch,pde_function)\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.save_data-NTuple{6,Any}","page":"Home","title":"DrMZ.save_data","text":"save_data(train_data,test_data,number_train_functions,number_test_functions,number_solution_points,pde_function)\n\nSave the train_data and test_data.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.shifted_nodes-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.shifted_nodes","text":"function shifted_nodes(a,b,xd)\n\nCompute the shifted nodes for Gauss-Legendre quadrature from int_-1^1 to $ \\int_a^b$.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.simpson-Tuple{Any,Any}","page":"Home","title":"DrMZ.simpson","text":"simpson(x_range,integrand)\n\nNumerical integration using the multi-application Simpson's and trapezoid rules.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.simpson13-NTuple{4,Any}","page":"Home","title":"DrMZ.simpson13","text":"simpson13(h,f1,f2)\n\nNumerical integration using the single-application Simpson's 1/3 rule.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.simpson38-NTuple{5,Any}","page":"Home","title":"DrMZ.simpson38","text":"simpson38(x_range,integrand)\n\nNumerical integration using the single-application Simpson's 3/8 rule.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.solution_extraction-NTuple{5,Any}","page":"Home","title":"DrMZ.solution_extraction","text":"solution_extraction(x_locations,t_values,solution,initial_condition,number_solution_points)\n\nExtract the specified number_solution_points randomly from the u(tx) solution space.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.solution_interpolation-NTuple{5,Any}","page":"Home","title":"DrMZ.solution_interpolation","text":"solution_interpolation(t_span_original,x_locations_original,t_span_interpolate,x_locations_interpolate,solution)\n\nCompute the u(tx) solution at intermediate (tx) locations using linear interpolation.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.solution_spatial_sampling-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.solution_spatial_sampling","text":"solution_spatial_sampling(x_prediction,x_target,solution)\n\nExtract the solution values u(tx) at a reduced number of equally spaced spatial locations.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.solution_temporal_sampling-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.solution_temporal_sampling","text":"solution_temporal_sampling(t_prediction,t_target,solution)\n\nExtract the solution values u(tx) at a reduced number of equally spaced temporal locations.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.spectral_approximation-Tuple{Any,Any}","page":"Home","title":"DrMZ.spectral_approximation","text":"spectral_approximation(basis,coefficients)\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.spectral_coefficients-Tuple{Any,Any}","page":"Home","title":"DrMZ.spectral_coefficients","text":"spectral_coefficients(basis,fnc)\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.spectral_matrix-Tuple{Any,Any}","page":"Home","title":"DrMZ.spectral_matrix","text":"spectral_matrix(basis,Dbasis)\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.standard_scaler-Tuple{Any}","page":"Home","title":"DrMZ.standard_scaler","text":"standard_scaler(x;dim=2)\n\nCompute the required scaler to shift the features to have zero mean and unit variance.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.standard_transform-Tuple{Any,Any}","page":"Home","title":"DrMZ.standard_transform","text":"standard_transform(x,scale_object)\n\nApply standard_scaler to shift the features.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.train_model-NTuple{6,Any}","page":"Home","title":"DrMZ.train_model","text":"train_model(branch,trunk,n_epoch,train_data;learning_rate=0.00001,save_at=2500)\n\nTrain the operator neural network using the mean squared error (MSE) and ADAM optimization for n_epochs epochs.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.trapz-Tuple{Any,Any}","page":"Home","title":"DrMZ.trapz","text":"trapz(x_range,integrand)\n\nNumerical integration using the multi-application trapezoidal rule.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.trapz1-Tuple{Any,Any,Any}","page":"Home","title":"DrMZ.trapz1","text":"trapz1(x_range,integrand)\n\nNumerical integration using the single-application trapezoidal rule.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.van_leer_limiter-Tuple{Any}","page":"Home","title":"DrMZ.van_leer_limiter","text":"function van_leer_limiter(r)\n\nCompute the Van Leer limiter Psi = fracr + r1+r where r is the gradient ratio.\n\n\n\n\n\n","category":"method"},{"location":"#DrMZ.viscous_burgers_pde!-NTuple{4,Any}","page":"Home","title":"DrMZ.viscous_burgers_pde!","text":"viscous_burgers_pde!(duhat,uhat,p,t)\n\nRHS for the viscous Burgers equation u_t = - u u_x + ν u_xx for numerical integration in Fourier space where ν is the viscosity.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
