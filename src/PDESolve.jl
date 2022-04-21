"""
    advection_pde!(duhat,uhat,p,t)

RHS for the advection equation ``u_t = - u_x`` for numerical integration in Fourier space.

"""
function advection_pde!(duhat,uhat,p,t)
    N, k, dL, nu, alpha = p;
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -alpha*im*k.*uhat;
end

"""
    advection_diffusion_pde!(duhat,uhat,p,t)

RHS for the advection-diffusion equation \$u_t = - u_x + ν u_{xx}\$ for numerical integration in Fourier space where ν is the viscosity.

"""
function advection_diffusion_pde!(duhat,uhat,p,t)
    N, k, dL, nu, alpha = p;
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -alpha*im*k.*uhat + nu*(im*k).^2 .*uhat;
end

"""
    viscous_burgers_pde!(duhat,uhat,p,t)

RHS for the viscous Burgers equation \$u_t = - u u_x + ν u_{xx}\$ for numerical integration in Fourier space where ν is the viscosity.

"""
function viscous_burgers_pde!(duhat,uhat,p,t)
    N, k, dL, nu = p;
    alpha = 1.0;
    uhat[Int(N/2)+1] = 0.0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= -nu*k.^2 .*uhat - uhat_nonlinear;
end

"""
    inviscid_burgers_pde!(duhat,uhat,p,t)

RHS for the inviscid Burgers equation \$u_t = - u u_x\$ for numerical integration in Fourier space.

"""
function inviscid_burgers_pde!(duhat,uhat,p,t)
    N, k, dL, nu = p;
    alpha = 1.0;
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= -uhat_nonlinear;
end

"""
    quadratic_nonlinear!(uhat,N,dL,alpha)

Compute the convolution sum \$\\frac{ik}{2}\\sum_{p+q=k} u_p u_q\$ resulting from the quadratic nonlinearity of Burgers equation \$u u_x\$ in Fourier space. Convolution sum is padded with the 3/2 rule for dealiasing.

"""
function quadratic_nonlinear(uhat,N,dL,alpha)
    M = Int((3/2)*N); # For dealiasing -> For MZ models must be 3*N to account for unresolved modes + dealiasing
    k_M = reduce(vcat,(2*π/dL)*[0:M/2-1 -M/2:-1]); # Wavenumbers for convolution sum with padding
    u_M = zeros(Complex{Float64},Int(M));
    u_M[1:Int(N/2)] = uhat[1:Int(N/2)];
    u_M[end-Int(N/2)+2:end] = uhat[Int(N/2)+2:end];
    u_M_sum = fft_norm(ifft_norm(u_M).*ifft_norm(u_M));
    u_M_convolution = alpha*im.*k_M/2.0 .*u_M_sum;
    uhat_nonlinear  = zeros(Complex{Float64},Int(N)); # Extract N modes from convolution
    uhat_nonlinear[1:Int(N/2)] = u_M_convolution[1:Int(N/2)];
    uhat_nonlinear[end-Int(N/2)+2:end] = u_M_convolution[end-Int(N/2)+2:end];
    return uhat_nonlinear
end

"""
    kdv_explicit_pde!(duhat,uhat,p,t)

Explicit portion of the RHS for the Korteweg-de Vries equation \$u_t = - u u_x - \\nu^2 u_{xxx}\$ for numerical integration numerical integration in Fourier space.

"""
function kdv_explicit_pde!(duhat,uhat,p,t)
    N, dL, nu = p;
    alpha = 1;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0.0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= -uhat_nonlinear;
end

"""
    kdv_implicit_pde!(duhat,uhat,p,t)

Implicit portion of the RHS for the Korteweg-de Vries equation \$u_t = - u u_x - \\nu^2 u_{xxx}\$ for numerical integration numerical integration in Fourier space.

"""
function kdv_implicit_pde!(duhat,uhat,p,t)
    N, dL, nu = p;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0.0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= nu^2*im.*k.^3 .*uhat;
end

"""
    kdv_pde!(duhat,uhat,p,t)

RHS for the Korteweg-de Vries equation \$u_t = - u u_x - \\nu^2 u_{xxx}\$ for numerical integration numerical integration in Fourier space.

"""
function kdv_pde!(duhat,uhat,p,t)
    N, dL, nu = p;
    alpha = 1;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0.0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= -uhat_nonlinear + nu^2*im.*k.^3 .*uhat;
end

"""
    ks_explicit_pde!(duhat,uhat,p,t)

Explicit portion of the RHS for the Kuramoto-Sivashinsky equation \$u_t = - u u_x -u_{xx} - \\nu u_{xxxx}\$ for numerical integration numerical integration in Fourier space.

"""
function ks_explicit_pde!(duhat,uhat,p,t)
    N, dL, nu = p;
    alpha = 1.0;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0.0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= -uhat_nonlinear;
end

"""
    ks_implicit_pde!(duhat,uhat,p,t)

Implicit portion of the RHS for the Kuramoto-Sivashinsky equation \$u_t = - u u_x -u_{xx} - \\nu u_{xxxx}\$ for numerical integration numerical integration in Fourier space.

"""
function ks_implicit_pde!(duhat,uhat,p,t)
    N, dL, nu = p;
    alpha = 1.0;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0.0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= (k.^2-nu*k.^4).*uhat;
end

"""
    ks_pde!(duhat,uhat,p,t)

RHS for the Kuramoto-Sivashinsky equation \$u_t = - u u_x -u_{xx} - \\nu u_{xxxx}\$ for numerical integration numerical integration in Fourier space.

"""
function ks_pde!(duhat,uhat,p,t)
    N, dL, nu = p;
    alpha = 1.0;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0.0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= -uhat_nonlinear + (k.^2-nu*k.^4).*uhat;
end

"""
    rhs_advection!(du,u,p,t)

RHS for the advection equation ``u_t = - u_x`` for numerical integration in custom basis space using discontinuous Galerkin method.

"""
function rhs_advection!(du,u,p,t)
    du .= p*u;
  end

"""
    rhs_advection_diffusion!(du,u,p,t)

RHS for the advection-diffusion equation \$u_t = - u_x + ν u_{xx}\$ for numerical integration in custom basis space using discontinuous Galerkin method and where ν is the viscosity.

"""
function rhs_advection_diffusion!(du,u,p,t)
    p1, p2 = p
    du .= p1*u + p2*u;
end

"""
    burgers_flux(u_right,u_left)

Numerical flux for Burgers nonlinearity, \$f(u) = \\frac{1}{2}u^2\$, for use with discontinuous Galerkin method.

"""
function burgers_flux(u_right,u_left)
    return (1/2)*((u_right+abs(u_right))/2-(u_left-abs(u_left))/2)^2
end

"""
    quadratic_nonlinear_triple_product_basis(basis_nodes,Dbasis_nodes,nodes,weights)

"""
function quadratic_nonlinear_triple_product_basis(basis_nodes,Dbasis_nodes,nodes,weights)

    nonlinear_triple = zeros(size(basis_nodes,2),size(basis_nodes,2),size(basis_nodes,2))
    for k in 1:size(basis_nodes,2)
        for l in 1:size(basis_nodes,2)
            for m in 1:size(basis_nodes,2)
                inner_loop = 0.0;
                for p in 1:size(basis_nodes,1)
                    inner_loop += basis_nodes[p,k]*basis_nodes[p,l]*conj(Dbasis_nodes[p,m])*weights[p];
                end
                nonlinear_triple[k,l,m] = inner_loop;
  
            end
        end
    end
    return nonlinear_triple
end

"""
    quadratic_nonlinear_basis(u,nonlinear_triple)

"""
function quadratic_nonlinear_basis(u,nonlinear_triple)
    quadratic_nonlinear = zeros(size(u,1));
    for m in 1:size(u,1)
        inner_loop = 0.0;
        for l in 1:size(u,1)
            for k in 1:size(u,1)
                inner_loop += u[k]*u[l]*nonlinear_triple[k,l,m];
            end
        end
        quadratic_nonlinear[m] = 1/2*inner_loop;
    end
    return quadratic_nonlinear
end

"""
    rhs_viscous_burgers!(du,u,p,t)

RHS for the viscous Burgers equation \$u_t = - u u_x + ν u_{xx}\$ for numerical integration in custom basis space using Discontinuous Galerking method and where ν is the viscosity.

"""
function rhs_viscous_burgers!(du,u,p,t)
    basis_left, basis_right, diffmat, triple_product = p;
    u_approx_left = u'*basis_left;
    u_approx_right = u'*basis_right;
    flux = quadratic_nonlinear_basis(u,triple_product); # Triple product approach
    du .= flux - burgers_flux(u_approx_right,u_approx_left)*(basis_right-basis_left) + diffmat*u;
end

"""
    rhs_inviscid_burgers!(du,u,p,t)

RHS for the inviscid Burgers equation \$u_t = - u u_x\$ for numerical integration in custom basis space using discontinuous Galerkin method.

"""
function rhs_inviscid_burgers!(du,u,p,t)
    basis_left, basis_right, triple_product = p;
    u_approx_left = u'*basis_left;
    u_approx_right = u'*basis_right;
    flux = quadratic_nonlinear_basis(u,triple_product); # Triple product approach
    du .= flux - burgers_flux(u_approx_right,u_approx_left)*(basis_right-basis_left);
end

"""
    rhs_explicit_kdv!(du,u,p,t)

Explicit portion of the RHS for the Korteweg-de Vries equation \$u_t = - u u_x - \\nu^2 u_{xxx}\$ for numerical integration in custom basis space using discontinuous Galerkin method.

"""
function rhs_explicit_kdv!(du,u,p,t)
    basis_left, basis_right, diff3mat, triple_product = p;
    u_approx_left = u'*basis_left;
    u_approx_right = u'*basis_right;
    flux = quadratic_nonlinear_basis(u,triple_product); # Triple product approach
    du .= flux - burgers_flux(u_approx_right,u_approx_left)*(basis_right-basis_left);
end

"""
    rhs_implicit_kdv!(du,u,p,t)

Implicit portion of the RHS for the Korteweg-de Vries equation \$u_t = - u u_x - \\nu^2 u_{xxx}\$ for numerical integration in custom basis space using discontinuous Galerkin method.

"""
function rhs_implicit_kdv!(du,u,p,t)
    basis_left, basis_right, diff3mat, triple_product = p;
    du .= diff3mat*u;
end

"""
    rhs_kdv!(du,u,p,t)

RHS for the Korteweg-de Vries equation \$u_t = - u u_x - \\nu^2 u_{xxx}\$ for numerical integration in custom basis space using discontinuous Galerkin method.

"""
function rhs_kdv!(du,u,p,t)
    basis_left, basis_right, diff3mat, triple_product = p;
    u_approx_left = u'*basis_left;
    u_approx_right = u'*basis_right;
    flux = quadratic_nonlinear_basis(u,triple_product); # Triple product approach
    du .= flux - burgers_flux(u_approx_right,u_approx_left)*(basis_right-basis_left) + diff3mat*u;
end

"""
    rhs_explicit_ks!(du,u,p,t)

Explicit portion of the RHS for the Kuramoto-Sivashinsky equation \$u_t = - u u_x -u_{xx} - \\nu u_{xxxx}\$ for numerical integration in custom basis space using discontinuous Galerkin method.

"""
function rhs_explicit_ks!(du,u,p,t)
    basis_left, basis_right, diffmat, diff4mat, triple_product = p;
    u_approx_left = u'*basis_left;
    u_approx_right = u'*basis_right;
    flux = quadratic_nonlinear_basis(u,triple_product); # Triple product approach
    du .= flux - burgers_flux(u_approx_right,u_approx_left)*(basis_right-basis_left);
end

"""
    rhs_implicit_ks!(du,u,p,t)

Implicit portion of the RHS for the Kuramoto-Sivashinsky equation \$u_t = - u u_x -u_{xx} - \\nu u_{xxxx}\$ for numerical integration in custom basis space using discontinuous Galerkin method.

"""
function rhs_implicit_ks!(du,u,p,t)
    basis_left, basis_right, diffmat, diff4mat, triple_product = p;
    du .= diffmat*u + diff4mat*u;
end

"""
    rhs_ks!(du,u,p,t)

RHS for the Kuramoto-Sivashinsky equation \$u_t = - u u_x -u_{xx} - \\nu u_{xxxx}\$ for numerical integration in custom basis space using discontinuous Galerkin method.

"""
function rhs_ks!(du,u,p,t)
    basis_left, basis_right, diffmat, diff4mat, triple_product = p;
    u_approx_left = u'*basis_left;
    u_approx_right = u'*basis_right;
    flux = quadratic_nonlinear_basis(u,triple_product); # Triple product approach
    du .= flux - burgers_flux(u_approx_right,u_approx_left)*(basis_right-basis_left) + diffmat*u + diff4mat*u;
end

"""
    generate_fourier_solution(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-3,nu=0.1,rtol=1e-10,atol=1e-14)

Generate the solution for a given `pde_function` and `initial_condition` on a periodic domain using a `N` mode Fourier expansion and a RK45 solver.

"""
function generate_fourier_solution(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-3,nu=0.1,rtol=1e-10,atol=1e-14,alpha=1.0)
    # Transform random initial condition to Fourier domain
    uhat0 = fft_norm(initial_condition);
    dL = abs(L2-L1);

    # Generate Fourier Galerkin solution for N
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]);
    p = [N,k,dL,nu,alpha]

    # Solve the system of ODEs in Fourier domain
    prob = ODEProblem(pde_function,uhat0,tspan,p);
    sol = solve(prob,DP5(),reltol=rtol,abstol=atol,saveat = dt)

    u_sol = zeros(size(sol.t,1),N);
    for j in 1:size(sol.t,1) # Reshape output and plot
        u_sol[j,:] = real.(ifft_norm(sol.u[j]));
    end
    return u_sol, sol.u, k
end

"""
    generate_fourier_solution_esdirk(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-4,rtol=1e-8,atol=1e-12,nu=0.1)

Generate the solution for a given `pde_function` and `initial_condition` on a periodic domain using a `N` mode Fourier expansion and a ESDIRK4 solver.

"""
function generate_fourier_solution_esdirk(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-4,rtol=1e-8,atol=1e-12,nu=0.1)

    # Transform random initial condition to Fourier domain
    uhat0 = fft_norm(initial_condition);
    dL = abs(L2-L1);
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]);
  
    # Generate Fourier Galerkin solution for N
    p = [N,dL,nu];
    t_length = tspan[2]/dt+1;
  
    # Solve the system of ODEs in Fourier domain
    prob = ODEProblem(pde_function,uhat0,tspan,p);
    sol = solve(prob,Kvaerno4(autodiff=false,linsolve=LinSolveFactorize(lu!)),reltol=rtol,abstol=atol,saveat = dt)

    u_sol = zeros(size(sol.t,1),N);
    for j in 1:size(sol.t,1) # Reshape output and plot
        u_sol[j,:] = real.(ifft_norm(sol.u[j])); # u[t,x,IC]
    end
    return u_sol, sol.u, k
end

"""
    generate_fourier_solution_implicit(L1,L2,tspan,N,initial_condition,pde_function_explicit,pde_function_implicit;dt=1e-4,rtol=1e-8,atol=1e-12,nu=0.1)

Generate the solution for a given `pde_function` and `initial_condition` on a periodic domain using a `N` mode Fourier expansion and a Crank-Nicolson solver.

"""
function generate_fourier_solution_implicit(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-4,rtol=1e-8,atol=1e-12,nu=0.1)
    # Transform random initial condition to Fourier domain
    uhat0 = fft_norm(initial_condition);
    dL = abs(L2-L1);
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]);
  
    # Generate Fourier Galerkin solution for N
    p = [N,dL,nu];
    t_length = tspan[2]/dt+1;
  
    # Solve the system of ODEs in Fourier domain
    prob = ODEProblem(pde_function,uhat0,tspan,p);
    sol = solve(prob,Trapezoid(autodiff=false,linsolve=LinSolveFactorize(lu!)),reltol=rtol,abstol=atol,saveat = dt,force_dtmin=true,maxiters=1e8)

    u_sol = zeros(size(sol.t,1),N);
    for j in 1:size(sol.t,1) # Reshape output and plot
        u_sol[j,:] = real.(ifft_norm(sol.u[j])); # u[t,x,IC]
    end
    return u_sol, sol.u, k
end

"""
    generate_basis_solution(nodes,weights,tspan,initial_condition,basis,params,pde_function;dt=1e-3,rtol=1e-10,atol=1e-14,peak=1.05)

Generate the solution for a given `pde_function` and `initial_condition` on a periodic domain using a custom basis function expansion and a RK45 solver.

"""
function generate_basis_solution(nodes,weights,tspan,initial_condition,basis,params,pde_function;dt=1e-3,rtol=1e-10,atol=1e-14,peak=1.05)
    u0 = expansion_coefficients(basis,initial_condition,nodes,weights);
    e0 = get_1D_energy_custom_coefficients(u0);

    t_length = tspan[2]/dt+1;
  
    # Solve the system of ODEs
    # Standard energy criteria
    energy_criteria(u,t,integrator) = get_1D_energy_custom_coefficients(integrator.u) > peak*e0;
    # Energy criteria for inviscid Burgers
    # energy_criteria(u,t,integrator) = (get_1D_energy_custom_coefficients(integrator.u) > peak*get_1D_energy_custom_coefficients(integrator.uprev)) || (abs(integrator.u'*params[2]-integrator.u'*params[1]) > 1e-2);

    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(energy_criteria,affect!,save_positions=(true,false))
    prob = ODEProblem(pde_function,u0,tspan,params);
    sol = solve(prob,DP5(),reltol=rtol,abstol=atol,saveat = dt,callback=cb)

    u_sol = zeros(size(sol.t,1),size(nodes,1));
    for j in 1:size(sol.t,1) # Reshape output and plot
        u_sol[j,:] = expansion_approximation(basis,sol.u[j],nodes)
    end
    return u_sol, sol.t, sol.u
end

"""
    generate_basis_solution_esdirk(nodes,weights,tspan,initial_condition,basis,params,pde_function;dt=1e-4,rtol=1e-8,atol=1e-12)

Generate the solution for a given `pde_function` and `initial_condition` on a periodic domain using a custom basis function expansion and a ESDIRK4 solver.

"""
function generate_basis_solution_esdirk(nodes,weights,tspan,initial_condition,basis,params,pde_function,;dt=1e-4,rtol=1e-8,atol=1e-12)

    u0 = expansion_coefficients(basis,initial_condition,nodes,weights);
  
    t_length = tspan[2]/dt+1;
  
    # Solve the system of ODEs
    prob = ODEProblem(pde_function,u0,tspan,params);
    sol = solve(prob,Kvaerno4(autodiff=false,linsolve=LinSolveFactorize(lu!)),reltol=rtol,abstol=atol,saveat = dt)

    u_sol = zeros(size(sol.t,1),size(nodes,1));
    for j in 1:size(sol.t,1) # Reshape output and plot
        u_sol[j,:] = expansion_approximation(basis,sol.u[j],nodes)
    end
      return u_sol, sol.t, sol.u
end

"""
    generate_basis_solution_implicit(nodes,weights,tspan,initial_condition,basis,params,pde_function;dt=1e-3,rtol=1e-8,atol=1e-12)

Generate the solution for a given `pde_function` and `initial_condition` on a periodic domain using a custom basis function expansion and a Crank-Nicolson solver.

"""
function generate_basis_solution_implicit(nodes,weights,tspan,initial_condition,basis,params,pde_function;dt=1e-4,rtol=1e-8,atol=1e-12)
    u0 = expansion_coefficients(basis,initial_condition,nodes,weights);
  
    t_length = tspan[2]/dt+1;
  
    # Solve the system of ODEs
    prob = ODEProblem(pde_function,u0,tspan,params);
    sol = solve(prob,Trapezoid(autodiff=false,linsolve=LinSolveFactorize(lu!)),reltol=rtol,abstol=atol,saveat = dt,force_dtmin=true,maxiters=1e8)

    u_sol = zeros(size(sol.t,1),size(nodes,1));
    for j in 1:size(sol.t,1) # Reshape output and plot
        u_sol[j,:] = expansion_approximation(basis,sol.u[j],nodes)
    end
      return u_sol, sol.t, sol.u
  end

"""
    central_difference(u_j,u_jpos,u_jneg,mu)

Compute the second order central difference for the viscous term of the viscous Burgers equation \$u_t = - u u_x + ν u_{xx}\$. `mu` is equal to \$ \\frac{\\nu}{\\Delta x^2}\$.

"""
function central_difference(u_j,u_jpos,u_jneg,mu)
    ux = zeros(size(u_j,2));
    ux = mu.*(u_jpos-2*u_j+u_jneg);
    return ux
end

"""
    minmod(x,y)

"""
function minmod(x,y)
    return sign(x)*max(0,min(abs(x),y*sign(x)));
end

"""
    ub(ulv,urv)

"""
function ub(ulv,urv)
    if ulv != urv
        return ((1/2)*urv^2 - (1/2)*ulv^2) / (urv - ulv); # ̄u_{j+/-1/2}
    else
        return ulv
    end
end

"""
    fl(ulv,urv,ubv)

"""
function fl(ulv,urv,ubv)
    return (1/2).*((1/2).*ulv.^2 .+ (1/2).*urv.^2 .- abs.(ubv).*(urv .- ulv)); # f_{j+/-1/2}
# fln(ulv,urv,ubv) = (1/2)*(flux.(urv) - flux.(ulv) - abs.(ubv).*(ulv - urv)); # f_{j+/-1/2}
end

"""
    ulpl(ujp,uj,ujn,kappa,omega)

"""
function ulpl(ujp,uj,ujn,kappa,omega)
    return uj .+ ((1-kappa)/4).*minmod.((uj .- ujn),omega.*(ujp .- uj)) .+ ((1+kappa)/4).*minmod.((ujp .- uj),omega.*(uj .- ujn)); # u_{j+1/2}^L
end

"""
    urpl(ujpp,ujp,uj,kappa,omega)

"""
function urpl(ujpp,ujp,uj,kappa,omega)
    return ujp  .- ((1+kappa)/4).*minmod.((ujp .- uj),omega.*(ujpp .- ujp)) .- ((1-kappa)/4).*minmod.((ujpp .- ujp),omega.*(ujp .- uj)); # u_{j+1/2}^R
end

"""
    ulnl(uj,ujn,ujnn,kappa,omega)

"""
function ulnl(uj,ujn,ujnn,kappa,omega)
    return ujn .+ ((1-kappa)/4).*minmod.((ujn .- ujnn),omega.*(uj .- ujn)) .+ ((1+kappa)/4).*minmod.((uj .- ujn),omega.*(ujn .- ujnn)); # u_{j-1/2}^L
end

"""
    urnl(ujp,uj,ujn,kappa,omega)

"""
function urnl(ujp,uj,ujn,kappa,omega)
    return uj .- ((1+kappa)/4).*minmod.((uj .- ujn),omega.*(ujp .- uj)) .- ((1-kappa)/4).*minmod.((ujp .- uj),omega.*(uj .- ujn)); # u_{j-1/2}^R
end

"""
    muscl_minmod_RHS!(du,u,p,t)

"""
function muscl_minmod_RHS!(du,u,p,t)
    dx, kappa, omega, nu = p;

    ulpvp = zeros(size(u,1));
    urpvp = zeros(size(u,1));
    ulnvp = zeros(size(u,1));
    urnvp = zeros(size(u,1));

    # ulp(ujp,uj,ujn,omega)
    ulpvp[2:end-1] .= ulpl(u[3:end],u[2:end-1],u[1:end-2],kappa,omega);
    ulpvp[1] = ulpl(u[2],u[1],u[end],kappa,omega);
    ulpvp[end] = ulpl(u[1],u[end],u[end-1],kappa,omega);

    # urp(ujpp,ujp,uj,omega)
    urpvp[1:end-2] .= urpl(u[3:end],u[2:end-1],u[1:end-2],kappa,omega);
    urpvp[end-1] = urpl(u[1],u[end],u[end-1],kappa,omega);
    urpvp[end] = urpl(u[2],u[1],u[end],kappa,omega);

    ubpp = ub.(ulpvp,urpvp);
    fpp = fl(ulpvp,urpvp,ubpp);

    # uln(uj,ujn,ujnn,omega)
    ulnvp[3:end] .= ulnl(u[3:end],u[2:end-1],u[1:end-2],kappa,omega);
    ulnvp[2] = ulnl(u[2],u[1],u[end],kappa,omega);
    ulnvp[1] = ulnl(u[1],u[end],u[end-1],kappa,omega);

    # urn(ujp,uj,ujn,omega)
    urnvp[2:end-1] .= urnl(u[3:end],u[2:end-1],u[1:end-2],kappa,omega);
    urnvp[1] = urnl(u[2],u[1],u[end],kappa,omega);
    urnvp[end] = urnl(u[1],u[end],u[end-1],kappa,omega);

    ubnp = ub.(ulnvp,urnvp);
    fnp = fl(ulnvp,urnvp,ubnp);

    du .= -(1/dx).*(fpp .- fnp);
end

"""
    muscl_minmod_viscous_RHS!(du,u,p,t)

"""
function muscl_minmod_viscous_RHS!(du,u,p,t)
    dx, kappa, omega, nu = p;

    mu = nu/(dx)^2;

    ulpvp = zeros(size(u,1));
    urpvp = zeros(size(u,1));
    ulnvp = zeros(size(u,1));
    urnvp = zeros(size(u,1));
    ux_viscous = zeros(size(u,1));

    # ulp(ujp,uj,ujn,omega)
    ulpvp[2:end-1] .= ulpl(u[3:end],u[2:end-1],u[1:end-2],kappa,omega);
    ulpvp[1] = ulpl(u[2],u[1],u[end],kappa,omega);
    ulpvp[end] = ulpl(u[1],u[end],u[end-1],kappa,omega);

    # urp(ujpp,ujp,uj,omega)
    urpvp[1:end-2] .= urpl(u[3:end],u[2:end-1],u[1:end-2],kappa,omega);
    urpvp[end-1] = urpl(u[1],u[end],u[end-1],kappa,omega);
    urpvp[end] = urpl(u[2],u[1],u[end],kappa,omega);

    ubpp = ub.(ulpvp,urpvp);
    fpp = fl(ulpvp,urpvp,ubpp);

    # uln(uj,ujn,ujnn,omega)
    ulnvp[3:end] .= ulnl(u[3:end],u[2:end-1],u[1:end-2],kappa,omega);
    ulnvp[2] = ulnl(u[2],u[1],u[end],kappa,omega);
    ulnvp[1] = ulnl(u[1],u[end],u[end-1],kappa,omega);

    # urn(ujp,uj,ujn,omega)
    urnvp[2:end-1] .= urnl(u[3:end],u[2:end-1],u[1:end-2],kappa,omega);
    urnvp[1] = urnl(u[2],u[1],u[end],kappa,omega);
    urnvp[end] = urnl(u[1],u[end],u[end-1],kappa,omega);

    ux_viscous[1] = central_difference(u[1],u[2],u[end],mu);
    ux_viscous[2:end-1] = central_difference(u[2:end-1],u[3:end],u[1:end-2],mu);
    ux_viscous[end] = central_difference(u[end],u[1],u[end-1],mu);

    ubnp = ub.(ulnvp,urnvp);
    fnp = fl(ulnvp,urnvp,ubnp);

    du .= -(1/dx).*(fpp .- fnp) .+ ux_viscous;
end

"""
    generate_muscl_minmod_solution(L1,L2,t_end,N,u0,pde_function_handle;dt=1e-4,kappa=-1)

"""
function generate_muscl_minmod_solution(L1,L2,t_end,N,u0,pde_function_handle;dt=1e-4,kappa=-1,nu=0.1)

    omega = ((3-kappa)/(1-kappa))

    dL = abs(L2-L1);
    # Set up periodic domain
    j = reduce(vcat,[0:1:N-1]);
    x = (dL.*j)./N;
    dx = x[2]-x[1];
    t = reduce(vcat,[0:dt:t_end]);

    p = [dx,kappa,omega,nu]
    t_span = (0,t_end);

    prob = ODEProblem(pde_function_handle,u0,t_span,p);
    sol = solve(prob,BS3(),reltol=1e-6,abstol=1e-8,saveat = dt);
   
    u_sol = zeros(size(sol.t,1),size(x,1));
    for i in 1:size(sol.t,1)
        u_sol[i,:] = sol.u[i];
    end

    return u_sol

end

"""
    generate_muscl_reduced(L1,L2,t_end,dt,M,N,ic_func,pde_function_handle;kappa=-1,nu=0.1)

"""
# function generate_muscl_reduced(L1,L2,t_end,dt,M,N,ic_func,pde_function_handle;kappa=-1,nu=0.1)
function generate_muscl_reduced(L1,L2,t_end,dt,M,N,ic,pde_function_handle;kappa=-1,nu=0.1)
    dL = abs(L2 - L1);
    j = reduce(vcat,[0:1:N-1]);
    x = (dL.*j)./N;
    j_full = reduce(vcat,[0:1:M-1]);
    x_full = (dL.*j_full)./M;
    # ic = ic_func.(x_full);
    u_full = generate_muscl_minmod_solution(L1,L2,t_end,M,ic,pde_function_handle;dt=dt,kappa=kappa,nu=nu)
    u_reduced = solution_spatial_sampling(x,x_full,u_full);
    return u_full, u_reduced
end

"""
    get_1D_energy_fft(u_solution)

Compute the energy in the Fourier domain using the scaling of \$ \\frac{1}{N} \$. Note: this does not include the 2π multiplier found in Parseval's identity for Fourier series and computes \$ \\frac{1}{2} \\sum \\vert \\hat{u}_k \\vert^2 \$.

"""
function get_1D_energy_fft(u_solution) # To Do: Add and extraction of specific mode step
    energy = zeros(size(u_solution,1))
    for i in 1:size(u_solution,1)
        u_hat = fft_norm(u_solution[i,:]);
        energy[i] = (1/2)*real((u_hat'*u_hat));
    end
    return energy
end

"""
    get_1D_energy_custom(basis,u_solution,L1,L2,weights;multiplier=1/(4*pi))

Compute the energy in the custom basis domain: \$ \\frac{1}{2} \\sum \\vert \\a_k \\vert^2 \$. Multiplier defaults to \$ \\frac{1}{4\\pi} \$ to match the Fourier calculation.

"""
function get_1D_energy_custom(basis,u_solution,nodes,weights;multiplier=1/(4*pi))
    energy = zeros(size(u_solution,1));
    for i in 1:size(u_solution,1)
        a_hat = expansion_coefficients(basis,u_solution[i,:],nodes,weights);
        energy[i] = get_1D_energy_custom_coefficients(a_hat,zeta=multiplier);
    end
    return energy
end

"""
    get_1D_energy_custom_coefficients(u_coefficients;zeta=1/(4*pi))

Compute the energy in the custom basis domain: \$ \\zeta \\sum \\vert \\a_k \\vert^2 \$. \$ \\zeta \$ defaults to \$ \\frac{1}{4\\pi} \$ to match the Fourier calculation.

"""
function get_1D_energy_custom_coefficients(u_coefficients;zeta=1/(4*pi))
        # energy[i] = (1/2)*(a_hat'*a_hat);
        energy = zeta*(u_coefficients'*u_coefficients);
    return energy
end

"""
    mode_extractor(uhat,N)

"""
function mode_extractor(uhat,N)
    uhat_N = zeros(typeof(uhat[1]),N);
    uhat_N[1:Int(N/2)] = uhat[1:Int(N/2)];
    uhat_N[end-Int(N/2)+2:end] = uhat[end-Int(N/2)+2:end];
    return uhat_N
end

"""
    get_1D_energy_upwind(u_solution_full,u_solution,N)

"""
function get_1D_energy_upwind(u_solution_full,u_solution,N)
    energy = zeros(size(u_solution,1))
    for i in 1:size(u_solution,1)
        u_upwind_fft = mode_extractor(fft_norm(u_solution_full[i,:]),N);
        energy[i] = (1/2)*real(u_upwind_fft'*u_upwind_fft);
    end
    return energy
end

"""
    spectral_approximation_fourier(x_locations,k,coefficients)

"""
function spectral_approximation_fourier(x_locations,k,coefficients)
    approximation = zeros(size(x_locations,1));
    for i = 1:length(k)
        approximation += coefficients[i]*exp.(im*k[i]*(x_locations));
    end
    return approximation
end
