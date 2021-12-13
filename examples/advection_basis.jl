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
    L1::Float64 = 0.0;
    L2::Float64 = 2*pi;
    n_epoch::Int = 50000;
    N::Int = 128;
end

function generate_basis_function_results(pde_function;kws...)

    args = Args(;);

    M = args.num_sensors;

    # Load the trained trunk
    trunk = load_trunk(args.n_epoch,pde_function)

    # Generate quadrature grid
    x, w = gauss_legendre(args.N,args.L1,args.L2);

    # Generate custom basis functions
    basis, S = build_basis(trunk,args.L1,args.L2,M,x,w);

    # Plot singular values
    pltsing = plot((1:1:length(S)),S,label=false,seriestype=:scatter,yaxis = :log10,xlabel = L"k", ylabel = L"$\sigma_k$",xlims=(1,length(S)))
    display(pltsing)
    savefig(pltsing, @sprintf("singular_values_%s_%i_functions.png",pde_function,M))
  
    # Plot custom basis functions
    x_plot = (args.L1:0.01:args.L2);
    pltbasis = plot(x_plot,basis[1].(x_plot),legend=false,label=L"$\phi_1$",xlabel=L"x",xlims=(x[1],x[end]),linewidth=1.5)
    for i in 2:M
      plot!(pltbasis,x_plot,basis[i].(x_plot),legend=false,label=L"$\phi_{$i}$",linewidth=1.5)
    end
    display(pltbasis)
    savefig(pltbasis, @sprintf("basis_functions_%s_%i_functions.png",pde_function,M))

    # Compute the expansion coefficients
    ak = expansion_coefficients(basis,exp.(sin.(x)),x,w);

    # Plot the expansion coefficients
    pltak = plot((1:1:length(basis)),abs.(ak),label=false,seriestype=:scatter,yaxis = :log10,xlabel = L"k", ylabel = L"|$a_k$|")
    display(pltak)
    savefig(pltak, @sprintf("expansion_coefficients_%s_%i_functions_exp_sinx.png",pde_function,M))

end

# Generate the custom basis functions from the trained trunk functions
generate_basis_function_results("advection_equation")