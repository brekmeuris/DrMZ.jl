ENV["JULIA_CUDA_SILENT"] = true
cd(@__DIR__)

using Plots; pyplot()
using DrMZ
using Flux
using Flux.NNlib
using LinearAlgebra
using ColorSchemes
using LaTeXStrings
using Printf
using Parameters: @with_kw
using Random
using Plots.PlotMeasures

fnt = Plots.font("sans-serif",8.0)
default(frame=:box,grid=:hide,palette=:viridis,titlefont=fnt,guidefont=fnt,tickfont=fnt,legendfont=fnt,size=(800,600),markersize=6)

@with_kw mutable struct Args
  n_epoch::Int = 50000;
  num_sensors::Int = 128;
  num_train_functions::Int = 500;
  num_test_functions::Int = Int(2*num_train_functions);
  num_sol_points::Int = 100;
  L1::Float64 = 0.0;
  L2::Float64 = 2*pi;
  tspan::Tuple = (0.0,10.0);
  dt::Float64 = 1e-3;
  N::Int = num_sensors;
  alpha::Float64 = 1.0;
  nu::Float64 = 0.1;
  modes::Int = 128;
end

function generate_DG_results(pde_function,pde_function_handle_basis,pde_function_handle_reference;kws...)

  args = Args(;);

  branch, trunk, train_ic, train_loc, train_sol, test_ic, test_loc, test_sol = load_data(args.n_epoch,args.num_train_functions,args.num_test_functions,pde_function)

  M = Flux.outdims(trunk[end],train_ic[:,1])[1];
  x, w = gauss_legendre(args.N,args.L1,args.L2);
  xf, wf = trapezoid(args.modes,args.L1,args.L2);
  W = diagm(0 => w);

  t_span = (0:args.dt:args.tspan[2]);

  basis_full, S = build_basis(trunk,args.L1,args.L2,M,x,w);
  r = findall(S .>= 1e-13)[end];
  basis = basis_full[1:r];

  plts_full = plot((1:1:length(S)),S,label=false,seriestype=:scatter,yaxis = :log10,xlabel = L"k", ylabel = L"$\sigma_k$",xlims=(1,length(S)))
  savefig(plts_full, @sprintf("singular_values_%s_%i.pdf",pde_function,r))

  c_vec_b = Int.(round.(range(1,stop=256,length=6)));
  x_plot = (args.L1:0.01:args.L2);
  pltbasis = plot(x_plot,basis_full[1].(x_plot),legend=:topright,label=L"\phi_1(x)",xlabel=L"x",xlims=(x[1],x[end]),linewidth=2,ylims=(-0.8+minimum(basis_full[1].(x_plot)),1.2+maximum(basis_full[1].(x_plot))),linecolor=ColorSchemes.viridis.colors[c_vec_b[1]])
  for i in 2:6
    plot!(pltbasis,x_plot,basis_full[i].(x_plot),legend=:topright,label=L"\phi_{%$i}(x)",linewidth=2,ylims=(-0.8+minimum(basis_full[i].(x_plot)),1.2+maximum(basis_full[i].(x_plot))),linecolor=ColorSchemes.viridis.colors[c_vec_b[i]])
  end
  savefig(pltbasis, @sprintf("basis_functions_%s_%i.pdf",pde_function,r))

  rand_vec = (2,"sine","expsine");
  mark = (:circle,:diamond,:hexagon);
  u0_vec = zeros(r,length(rand_vec));
  error_vec = zeros(size(t_span,1),length(rand_vec));
  error_vec_red = zeros(size(t_reduced,1),length(rand_vec));
  avg_error_vec = zeros(3);
  diff_vec = zeros(size(t_span,1),length(rand_vec));
  c_vec = Int.(round.(range(1,stop=256,length=length(rand_vec))));
  leg = String[];

  for j in 1:length(rand_vec)

    random_integer = rand_vec[j];

    if random_integer == 2 
      rand_int = 2;
      icf = test_ic[:,rand_int*args.num_sol_points];
      ic_fft = fft_norm(icf);
      k = (2π)/abs(args.L2-args.L1)*reduce(vcat,[0:args.num_sensors/2-1 -args.num_sensors/2:-1]);
      ic = real.(spectral_approximation_fourier(x,k,ic_fft));
      push!(leg,"Random (in-dist.)")
    elseif random_integer == "sine"
      ic = sin.(x);
      icf = sin.(xf);
      rand_int = 0;
      push!(leg,L"\sin(x)")
    elseif random_integer == "expsine"
      ic = exp.(sin.(x));
      icf = exp.(sin.(xf));
      rand_int = 1000;
      push!(leg,L"e^{\sin(x)}")
    end

    pltic = plot(x,ic,legend=false,xlims=(x[1],x[end]),xlabel=L"x",ylabel=L"u(x)",linewidth=2)
    savefig(pltic, @sprintf("u0_%s_%i.pdf",pde_function,rand_int))

    u0_vec[:,j] = expansion_coefficients(basis,ic,x,w);

    # Determine flow direction
    if args.alpha >= 0.0
      xin = args.L1;
      xout = args.L2;
    else
      xin = args.L2;
      xout = args.L1;
    end

    xin_diff = args.L1;
    xout_diff = args.L2;

    # Pre-compute derivative and boundary condition matricies
    Dbasis = basis_derivative(basis);
    basis_nodes = basis_eval(basis,x);
    Dbasis_nodes = basis_eval(Dbasis,x);
    basis_in = basis_eval(basis,xin)[:];
    basis_out = basis_eval(basis,xout)[:];

    basis_in_diff = basis_eval(basis,xin_diff)[:];
    basis_out_diff = basis_eval(basis,xout_diff)[:];

    Dmatrix = Dbasis_nodes'*W*basis_nodes;
    BC = (basis_out-basis_in)*basis_out';

    # Parameters for ODE solver
    params = [args.alpha*Dmatrix-abs(args.alpha)*BC, args.nu*(-Dmatrix + (basis_out_diff - basis_in_diff)*basis_out_diff')*(-Dmatrix + (basis_out_diff - basis_in_diff)*basis_in_diff')];

    u_basis, t_basis, u_basis_coeffs = generate_basis_solution(x,w,args.tspan,ic,basis_nodes,params,pde_function_handle_basis;dt=args.dt);

    u_fourier, fourier_coefficients, k = generate_fourier_solution(args.L1,args.L2,args.tspan,args.modes,icf,pde_function_handle_reference;dt=args.dt,alpha=args.alpha,nu=args.nu);

    pltbasis = plot(heatmap(x,t_span,u_basis,aspect_ratio=(args.L2)/args.tspan[2]),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = L"x", ylabel = L"t",xlims=(args.L1,args.L2),ylims=(0,args.tspan[2]),dpi=300,right_margin=150mm)
    savefig(pltbasis, @sprintf("basis_%s_%i_%i.png",pde_function,r,rand_int))

    pltfourier = plot(heatmap(periodic_fill_domain(xf),t_span,periodic_fill_solution(u_fourier),aspect_ratio=(args.L2)/args.tspan[2]),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = L"x", ylabel = L"t",xlims=(args.L1,args.L2),ylims=(0,args.tspan[2]),dpi=300,right_margin=150mm)
    savefig(pltfourier, @sprintf("fourier_%s_%i_%i.png",pde_function,args.modes,rand_int))

    u_fourier_shifted = zeros(size(t_span,1),size(x,1));
    for i in 1:size(t_span,1)
        u_fourier_shifted[i,:] = real.(spectral_approximation_fourier(x,k,fourier_coefficients[i][:]))
    end

    error_vec[:,j] = norm_rel_error(u_fourier_shifted,u_basis);
    avg_error_vec[j] = average_error(t_span,error_vec[:,j]);

    pltfinal = plot(x,u_basis[end,:],label="$r custom basis function solution",legend=:outerbottom,foreground_color_legend = nothing,xlims=(args.L1,args.L2),xlabel=L"x",ylabel=L"u(x)")
    plot!(pltfinal,xf,u_fourier[end,:],label="Fourier solution",linestyle=:dash)
    savefig(pltfinal, @sprintf("end_%s_%i_%i.pdf",pde_function,r,rand_int))

    endpoints = zeros(size(t_basis,1),2);
    for i in 1:size(t_basis,1)
      endpoints[i,1] = expansion_approximation(basis,u_basis_coeffs[i],args.L1)[1];
      endpoints[i,2] = expansion_approximation(basis,u_basis_coeffs[i],args.L2)[1];
      diff_vec[i,j] = abs(endpoints[i,2] - endpoints[i,1]);
    end
  
  end

  plte0 = plot((1:1:length(basis)),abs.(u0_vec[:,1]),label=L"u_0(x) = "*leg[1],legend=:best,seriestype=:scatter,yaxis = :log10,xlabel = L"l", ylabel = L"|$a_l(0)$|",markershape=mark[1],markercolor=ColorSchemes.viridis.colors[c_vec[1]],yticks=([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0],[L"10^{-12}",L"10^{-10}",L"10^{-8}",L"10^{-6}",L"10^{-4}",L"10^{-2}",L"10^{0}"]))
  pltmulterror = plot(t_span,error_vec[:,1],yaxis = :log10,label=L"u_0(x) = "*leg[1]*", avg.: $(@sprintf("%.4e",avg_error_vec[1]))",legend=:best,xlabel = L"t", ylabel = L"E(t)",xlims=(t_span[1],t_span[end]),linewidth=2,linecolor=ColorSchemes.viridis.colors[c_vec[1]])
  pltbc = plot(t_span,diff_vec[:,1],label=L"u_0(x) = "*leg[1],legend=:best,xlabel = L"t", ylabel = L"|$u(t,2 \pi)-u(t,0)$|",yformatter=:scientific,xlims=(0,t_span[end]),linewidth=2,linecolor=ColorSchemes.viridis.colors[c_vec[1]])

  for j in 2:length(rand_vec)
    plot!(plte0,(1:1:length(basis)),abs.(u0_vec[:,j]),label=L"u_0(x) = "*leg[j],seriestype=:scatter,yaxis = :log10,xlabel = L"l", ylabel = L"|$a_l(0)$|",markershape=mark[j],markercolor=ColorSchemes.viridis.colors[c_vec[j]],yticks=([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0],[L"10^{-12}",L"10^{-10}",L"10^{-8}",L"10^{-6}",L"10^{-4}",L"10^{-2}",L"10^{0}"]))
    plot!(pltmulterror,t_span,error_vec[:,j],yaxis = :log10,label=L"u_0(x) = "*leg[j]*", avg.: $(@sprintf("%.4e",avg_error_vec[j]))",legend=:best,xlabel = L"t", ylabel = L"E(t)",xlims=(t_span[1],t_span[end]),linewidth=2,linecolor=ColorSchemes.viridis.colors[c_vec[j]])
    plot!(pltbc,t_span,diff_vec[:,j],label=L"u_0(x) = "*leg[j],legend=:best,xlabel = L"t", ylabel = L"|$u(t,2 \pi)-u(t,0)$|",yformatter=:scientific,xlims=(0,t_span[end]),linewidth=2,linecolor=ColorSchemes.viridis.colors[c_vec[j]])
  end

  savefig(plte0, @sprintf("u0_coefficients_%s_%i.pdf",pde_function,r))
  savefig(pltmulterror, @sprintf("error_%s_%i.pdf",pde_function,r))
  savefig(pltbc, @sprintf("BC_error_%s_%i.pdf",pde_function,r))

  return nothing

end

generate_DG_results("advection_diffusion_equation",rhs_advection_diffusion!,advection_diffusion_pde!)