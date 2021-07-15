ENV["JULIA_CUDA_SILENT"] = true

using DrMZ
using Flux
using Plots; pyplot()
using Plots.PlotMeasures
using Random
using Printf
using ColorSchemes
using Parameters: @with_kw
using BSON: @load
using LaTeXStrings
using LinearAlgebra
using DelimitedFiles

default(fontfamily="serif",frame=:box,grid=:hide,palette=:viridis,markeralpha=0.4,dpi=200,legendfontsize=6);#,size=(Int(1.5*600),Int(1.5*400)))
PyPlot.matplotlib.rc("mathtext",fontset="cm")

@with_kw mutable struct Args
    num_sensors::Int = 2048;
    num_train_functions::Int = 500;
    num_test_functions::Int = Int(2*num_train_functions);
    num_sol_points::Int = 100;
    L1::Float64 = 0.0;
    L2::Float64 = 2*pi;
    t_end::Float64 = 10.0;
    tspan::Tuple = (0.0,t_end);
    n_epoch::Int = 25000;
    N::Int = 32;
end

function generate_basis_results(pde_function,pde_function_handle,opnn_pde_function_handle;random_integer = "None",kws...)

    args = Args(;);

    # Load all the data
    branch, trunk, train_ic, train_loc, train_sol, test_ic, test_loc, test_sol = load_data(args.n_epoch,args.num_train_functions,args.num_test_functions,pde_function)

    train_MSE = loss_all(branch,trunk,train_ic,train_loc,train_sol);
    test_MSE = loss_all(branch,trunk,test_ic,test_loc,test_sol);
    println("Training set MSE: $train_MSE, test set MSE: $test_MSE")

    # Set up x domain
    dL = abs(args.L2 - args.L1);
    j = reduce(vcat,[0:1:args.num_sensors-1]);
    x = (dL.*j)./args.num_sensors;

    dt = 1e-3;
    t = reduce(vcat,[0:dt:args.t_end]);

    if random_integer == "none"
        rand_int = rand(1:args.num_test_functions);
        ic = test_ic[:,rand_int*args.num_sol_points];
    elseif random_integer == "exact"
        ic = 1 .+ cos.(x);
        rand_int = 0;
    else
        rand_int = random_integer;
        ic = test_ic[:,rand_int*args.num_sol_points];
    end

    # dLb = abs(1-0);
    # jb = reduce(vcat,[0:1:args.num_sensors-1]);
    # xb = (dLb.*jb)./args.num_sensors;
    # ic_b = 1 .+ cos.(2*pi*xb);

    # x_N, ic_N = reduced_initial_condition(args.L1,args.L2,args.N,x,ic);
    # opnn_output_width = Flux.outdims(trunk[end],ic_b)[1];

    # basis = zeros(size(xb,1),opnn_output_width);
    # for i in 1:opnn_output_width
    #     basis[:,i] = basis_OpNN(trunk,xb,i)
    # end

    trunk_funcs = readdlm("TFvaluesU");

    L0 = trunk_funcs[1];
    N = trunk_funcs[2];
    k = 2;
    basis = zeros(Int(N),Int(L0));
    for m = 1:Int(L0)
        basis[:,m] = trunk_funcs[k+1:k+Int(N)];
        k = k + Int(N);
    end

    # norm_basis=[norm(basis[:,i]) for i in 1:size(basis,2)];
    # basis_trunc = basis[:,findall(norm_basis.>0)];
    F = qr(basis);
    basis_full = Matrix(F.Q);
    # basis_full = F.Q*Matrix(I,size(basis,1),size(basis,1));

    # return basis, trunk_funcs, basis_full

    # Generate custom basis functions for num_sensors
    # basis_full = build_basis(trunk,x,opnn_output_width,ic);

    # Generate custom basis functions for N
    # basis_N = build_basis_redefinition(basis_full,x_N,x);

    # Generate the solution using the OpNN basis functions
    u_opnn_full = generate_basis_solution(args.L1,args.L2,args.tspan,args.num_sensors,basis_full,ic,opnn_pde_function_handle);
    # u_opnn_full_trunc = generate_basis_solution(args.L1,args.L2,args.tspan,args.num_sensors,basis_full[:,1:args.N],ic,opnn_pde_function_handle);
    # u_opnn_N = generate_basis_solution(args.L1,args.L2,args.tspan,args.N,basis_N,ic_N,opnn_pde_function_handle);

    # Extract solution points in real space from full OpNN trunc solution
    # u_opnn_full_trunc_reduced = solution_spatial_sampling(x_N,x,u_opnn_full_trunc);

    # Generate the M size Fourier solution for validation
    u_fourier_M = generate_fourier_solution(args.L1,args.L2,args.tspan,args.num_sensors,ic,pde_function_handle);

    # Extract solution points in real space from M Fourier solution
    # u_fourier_M_reduced = solution_spatial_sampling(x_N,x,u_fourier_M);

    # Compute the relative error vs M size Fourier solution
    opnn_full_rel_error = norm_rel_error(u_fourier_M,u_opnn_full);
    # opnn_full_trunc_rel_error = norm_rel_error(u_fourier_M_reduced,u_opnn_full_trunc_reduced);
    # opnn_N_rel_error = norm_rel_error(u_fourier_M_reduced,u_opnn_N);

    pltmulterror = plot(t,opnn_full_rel_error,legend=:outerbottom,foreground_color_legend = nothing,label="Error at $(args.num_sensors) locations with the $(size(basis_full,2)) custom basis vectors: average error $(@sprintf("%.4e",average_error(t,opnn_full_rel_error)))",xlabel = L"t", ylabel = "Relative Error",xlims=(t[1],t[end]))
    # plot!(pltmulterror,t,opnn_full_trunc_rel_error,label="Error at $(args.N) locations with the $(size(basis_full[:,1:args.N],2)) custom basis vectors: average error $(@sprintf("%.4e",average_error(t,opnn_full_trunc_rel_error)))")
    # plot!(pltmulterror,t,opnn_N_rel_error,label="Error at $(args.N) locations with the $(size(basis_N,2)) custom basis vectors using redefinition: average error $(@sprintf("%.4e",average_error(t,opnn_N_rel_error)))")
    display(pltmulterror)
    savefig(pltmulterror, @sprintf("evolution_error_%s_%i_%i.png",pde_function,args.num_sensors,args.N))

    pltopnn = plot(heatmap(periodic_fill_domain(x),t,periodic_fill_solution(u_opnn_full),aspect_ratio=dL/args.t_end),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = L"x", ylabel = L"t",xlims=(x[1],x[end]),ylims=(0,args.t_end))#,xlims=(x[1],x[end]),ylims=(t_data[1],t_data[end]))
    display(pltopnn)
    savefig(pltopnn, @sprintf("opnn_solution_%s_%i_%i.png",pde_function,args.num_sensors,args.N))

    # pltopnn_trunc = plot(heatmap(periodic_fill_domain(x),t,periodic_fill_solution(u_opnn_full_trunc),aspect_ratio=dL/args.t_end),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = L"x", ylabel = L"t",xlims=(x[1],x[end]),ylims=(0,args.t_end))#,xlims=(x[1],x[end]),ylims=(t_data[1],t_data[end]))
    # display(pltopnn_trunc)
    # savefig(pltopnn_trunc, @sprintf("opnn_truncation_solution_%s_%i_%i.png",pde_function,args.num_sensors,args.N))

    # u_opnn_N_smooth = solution_interpolation(t,periodic_fill_domain(x_N),t,x,periodic_fill_solution(u_opnn_N));
    #
    # pltopnn_N = plot(heatmap(periodic_fill_domain(x),t,periodic_fill_solution(u_opnn_N_smooth),aspect_ratio=dL/args.t_end),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = L"x", ylabel = L"t",xlims=(x[1],x[end]),ylims=(0,args.t_end))#,xlims=(x[1],x[end]),ylims=(t_data[1],t_data[end]))
    # display(pltopnn_N)
    # savefig(pltopnn_N, @sprintf("opnn_redefinition_solution_%s_%i_%i.png",pde_function,args.num_sensors,args.N))

    pltfourier = plot(heatmap(periodic_fill_domain(x),t,periodic_fill_solution(u_fourier_M),aspect_ratio=dL/args.t_end),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = L"x", ylabel = L"t",xlims=(x[1],x[end]),ylims=(0,args.t_end))#,xlims=(x[1],x[end]),ylims=(t_data[1],t_data[end]))
    display(pltfourier)
    savefig(pltfourier, @sprintf("fourier_solution_%s_%i_%i.png",pde_function,args.num_sensors,args.N))

    pltfinal = plot(x,u_opnn_full[end,:],label="N = $(size(basis_full,2)) custom basis vector solution",legend=:outerbottom,foreground_color_legend = nothing,xlims=(x[1],x[end]),xlabel=L"x",ylabel=L"u(x)",linewidth=1.5)
    # plot!(pltfinal,x,u_opnn_full_trunc[end,:],label="r = $(size(basis_full[:,1:args.N],2)) custom basis vector solution")#,linestyle=:dash)
    # plot!(pltfinal,periodic_fill_domain(x_N),periodic_fill_solution(u_opnn_N[end,:])[:,1],label="n = $(size(basis_N,2)) custom basis vector solution using redefinition")#,linestyle=:dash)
    plot!(pltfinal,x,u_fourier_M[end,:],label="M = $(args.num_sensors) Fourier solution",linestyle=:dash,linewidth=1)
    display(pltfinal)
    savefig(pltfinal, @sprintf("opnn_vs_fourier_solution_end_%s_%i_%i.png",pde_function,args.num_sensors,args.N))

    return basis, trunk_funcs, basis_full

end


# Change the working directory to the file location
cd(@__DIR__)

# Clear the REPL if using Juno
# clearconsole()

basis, trunk_funcs, basis_full = generate_basis_results("advection_equation",advection_pde!,opnn_advection_pde!;random_integer = "exact")
