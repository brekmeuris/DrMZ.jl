ENV["JULIA_CUDA_SILENT"] = true
# Change the working directory to the file location
cd(@__DIR__)

using DrMZ
using Parameters: @with_kw
using Flux.NNlib
using LaTeXStrings
using ColorSchemes
using Flux
using Plots; pyplot()
using Plots.PlotMeasures
using Random
using Printf

default(fontfamily="serif",frame=:box,grid=:hide,palette=:viridis,markeralpha=0.4,dpi=200,legendfontsize=6);
PyPlot.matplotlib.rc("mathtext",fontset="cm")

@with_kw mutable struct Args
    num_sensors::Int = 128;
    num_train_functions::Int = 500;
    num_test_functions::Int = Int(2*num_train_functions);
    num_sol_points::Int = 100;
    L1::Float64 = 0.0;
    L2::Float64 = 2*pi;
    tspan::Tuple = (0.0,1.0);
    n_epoch::Int = 25000;
    N::Int = num_sensors;
    branch_layers::Int = 2; # Branch depth
    branch_neurons::Array{Tuple} = [(Int(num_sensors),num_sensors),(num_sensors,N)]; # Branch layers input and output dimensions - quantity of tuples must match branch depth
    branch_activations::Array = [tanh,identity]; # Activation functions for each branch layer
    trunk_layers::Int = 3; # Trunk depth
    trunk_neurons::Array{Tuple} = [(2,num_sensors),(num_sensors,num_sensors),(num_sensors,N)]; # Trunk layers input and output dimensions - quantity of tuples must match trunk depth
    trunk_activations::Array = [tanh,tanh,tanh]; # Activation functions for each trunk layer
end

function generate_train(pde_function,pde_function_handle;kws...)

    args = Args(;);

    # Generate training and testing data
    train_data, test_data = generate_periodic_train_test(args.tspan,args.num_sensors,args.num_train_functions,args.num_test_functions,args.num_sol_points,pde_function_handle);
    save_data(train_data,test_data,args.num_train_functions,args.num_test_functions,args.num_sol_points,pde_function)

    branch = build_dense_model(args.branch_layers,args.branch_neurons,args.branch_activations);
    trunk = build_dense_model(args.trunk_layers,args.trunk_neurons,args.trunk_activations)

    # Train the operator neural network
    branch, trunk = train_model(branch,trunk,args.n_epoch,train_data,test_data,pde_function);
    save_model(branch,trunk,args.n_epoch,pde_function)

end

function generate_opnn_results(pde_function,pde_function_handle;random_integer = "none",kws...)

    args = Args(;);

    # Load all the data
    branch, trunk, train_ic, train_loc, train_sol, test_ic, test_loc, test_sol = load_data(args.n_epoch,args.num_train_functions,args.num_test_functions,pde_function)

    train_MSE = loss_all(branch,trunk,train_ic,train_loc,train_sol);
    test_MSE = loss_all(branch,trunk,test_ic,test_loc,test_sol);
    println("Training set MSE: $train_MSE, test set MSE: $test_MSE")

    dL = abs(args.L2 - args.L1);
    j = reduce(vcat,[0:1:args.num_sensors-1]);
    x = (dL.*j)./args.num_sensors;
    dt = 1e-3;
    t = reduce(vcat,[0:dt:args.tspan[2]]);

    if random_integer == "none"
        rand_int = rand(1:args.num_test_functions);
        ic = test_ic[:,rand_int*args.num_sol_points];
    elseif random_integer == "exact"
        ic = sin.(x/2).^2;
        rand_int = 0;
    else
        rand_int = random_integer;
        ic = test_ic[:,rand_int*args.num_sol_points];
    end

    # Comparisons vs random test IC
    pltrand_test = plot(x,ic,label = "IC",linewidth=:2,xlims=(x[1],x[end]),xlabel=L"x",ylabel=L"u(x)")
    plot!(pltrand_test,x,ic,label = "Sensor locations",seriestype=:scatter)
    display(pltrand_test)
    savefig(pltrand_test, @sprintf("random_ic_%s_%i.png",pde_function,rand_int))

    # M size Fourier solution for comparisons
    u_fourier_M = generate_fourier_solution(args.L1,args.L2,args.tspan,args.num_sensors,ic,pde_function_handle)[1];

    pltexact = plot(heatmap(periodic_fill_domain(x),t,periodic_fill_solution(u_fourier_M),aspect_ratio=dL/args.tspan[2]),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = L"x", ylabel = L"t",xlims=(x[1],x[end]),ylims=(0,args.tspan[2]))
    display(pltexact)
    savefig(pltexact, @sprintf("exact_fourier_solution_random_ic_%s_%i.png",pde_function,rand_int))

    # Operator neural network prediction
    u_predict = predict(branch,trunk,ic,x,t);

    pltpredict = plot(heatmap(periodic_fill_domain(x),t,periodic_fill_solution(u_predict),aspect_ratio=dL/args.tspan[2]),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = L"x", ylabel = L"t",xlims=(x[1],x[end]),ylims=(0,args.tspan[2]))
    display(pltpredict)
    savefig(pltpredict, @sprintf("opnn_solution_random_ic_epochs_%i_%s_%i.png",args.n_epoch,pde_function,rand_int))

    opnn_error = norm_rel_error(u_fourier_M,u_predict);
    opnn_error = plot(t,opnn_error,legend=:outerbottom,foreground_color_legend = nothing,label="Operator network average error $(@sprintf("%.4e",average_error(t,opnn_error)))",xlabel = L"t", ylabel = "Rel. Error",xlims=(t[1],t[end]))
    display(opnn_error)
    savefig(opnn_error, @sprintf("opnn_vs_exact_solution_random_ic_epochs_%i_%s_%i.png",args.n_epoch,pde_function,rand_int))

end

function generate_basis_function_results(pde_function;kws...)

    args = Args(;);

    # Load all the data
    branch, trunk, train_ic, train_loc, train_sol, test_ic, test_loc, test_sol = load_data(args.n_epoch,args.num_train_functions,args.num_test_functions,pde_function)

    M = Flux.outdims(trunk[end],train_ic[:,1])[1];
    x, w = gauss_legendre(args.num_sensors,0,2*pi);

    basis, S = build_basis(trunk,args.L1,args.L2,M,x,w);

    pltsing = plot((1:1:length(S)),S,label=false,seriestype=:scatter,yaxis = :log10,xlabel = L"k", ylabel = L"$\sigma_k$",xlims=(1,length(S)))
    display(pltsing)
    savefig(pltsing, @sprintf("singular_values_%s_%i_functions.png",pde_function,M))
  
    x_plot = (args.L1:0.01:args.L2);
    pltbasis = plot(x_plot,basis[1].(x_plot),legend=false,label=L"$\phi_1$",xlabel=L"x",xlims=(x[1],x[end]),linewidth=1.5)
    for i in 2:M
      plot!(pltbasis,x_plot,basis[i].(x_plot),legend=false,label=L"$\phi_{$i}$",linewidth=1.5)
    end
    display(pltbasis)
    savefig(pltbasis, @sprintf("basis_functions_%s_%i_functions.png",pde_function,M))

    ak = expansion_coefficients(basis,exp.(sin.(x)),x,w);

    pltak = plot((1:1:length(basis)),log10.(abs.(ak)),label=false,seriestype=:scatter,xlabel = L"k", ylabel = L"|$a_k$|")
    display(pltak)
    savefig(pltak, @sprintf("expansion_coefficients_%s_%i_functions_exp_sinx.png",pde_function,M))

    save_basis(basis,M,pde_function)

end

# Generate data and train the model
generate_train("advection_equation",advection_pde!)

# Generate results - enter "none", an integer value (e.g. 975), or "exact" to specify an initial condition
# If testing ability of network to extrapolate beyond the training interval, comment out the generation functions above and adjust the Arg tspan
generate_opnn_results("advection_equation",advection_pde!;random_integer = "exact")

# Generate the custom basis functions from the trained trunk functions
generate_basis_function_results("advection_equation")