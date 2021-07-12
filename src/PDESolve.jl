"""
    advection_pde!(duhat,uhat,p,t)

RHS for the advection equation ``u_t = - u_x`` for numerical integration in Fourier space.

"""
function advection_pde!(duhat,uhat,p,t)
    N, k, dL, nu = p;
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -im*k.*uhat;
end

"""
    advection_diffusion_pde!(duhat,uhat,p,t)

RHS for the advection-diffusion equation \$u_t = - u_x + ν u_{xx}\$ for numerical integration in Fourier space where ν is the viscosity.

"""
function advection_diffusion_pde!(duhat,uhat,p,t)
    N, k, dL, nu = p;
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -im*k.*uhat + nu*(im*k).^2 .*uhat;
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
    duhat .= -nu*k.^2 .*uhat .- uhat_nonlinear;
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
    opnn_advection_pde!(du,u,p,t)

RHS for the advection equation ``u_t = - u_x`` for numerical integration in the custom basis space.

"""
function opnn_advection_pde!(du,u,p,t)
    Dmatrix, D2matrix, nu = p;
    du .= -Dmatrix*u;
end

"""
    opnn_advection_diffusion_pde!(du,u,p,t)

RHS for the advection-diffusion equation \$u_t = - u_x + ν u_{xx}\$ for numerical integration in the custom basis space where ν is the viscosity.

"""
function opnn_advection_diffusion_pde!(du,u,p,t)
    Dmatrix, D2matrix, nu = p;
    du .= -Dmatrix*u .+ nu*D2matrix*u;
end

"""
    opnn_viscous_burgers_pde!(du,u,p,t)

RHS for the viscous Burgers equation \$u_t = - u u_x + ν u_{xx}\$ for numerical integration in the custom basis space where ν is the viscosity.

"""
function opnn_viscous_burgers_pde!(du,u,p,t)
    D2matrix, basis, dL, N, nu = p;
    u_nonlinear = quadratic_nonlinear_opnn_pseudo(basis,u,dL,N);
    du .= nu*D2matrix*u .- u_nonlinear;
end

"""
    opnn_inviscid_burgers_pde!(du,u,p,t)

RHS for the inviscid Burgers equation \$u_t = - u u_x\$ for numerical integration in the custom basis space.

"""
function opnn_inviscid_burgers_pde!(du,u,p,t)
    D2matrix, basis, dL, N, nu = p;
    u_nonlinear = quadratic_nonlinear_opnn_pseudo(basis,u,dL,N);
    du .= -u_nonlinear;

end

"""
    quadratic_nonlinear_opnn(uhat,nonlinear_triple)

Compute the triple product sum \$\\sum_{k=1}^N \\sum_{l=1}^N u_k u_l \\sum_{j=0}^{N-1} \\phi_{jk} \\phi_{jl}^{'} \\phi_{jm}^{*}\$ resulting from the quadratic nonlinearity of Burgers equation \$u u_x\$ in custom basis space.

"""
function quadratic_nonlinear_opnn(uhat,nonlinear_triple)
    quadratic_nonlinear = zeros(size(uhat,1));
    for m in 1:size(uhat,1)
        inner_loop = 0.0;
        for l in 1:size(uhat,1)
            for k in 1:size(uhat,1)
                inner_loop += uhat[k]*uhat[l]*nonlinear_triple[k,l,m];
            end
        end
        quadratic_nonlinear[m] = inner_loop;
    end
    return quadratic_nonlinear
end

"""
    quadratic_nonlinear_opnn_pseudo(basis,uhat,dL,N)

Compute the quadratic nonlinearity of Burgers equation \$u u_x\$ in custom basis space using a pseudo spectral type approach. At each step, transform solution to real space and compute the product of the real space solution and the spatial derivative of the real space solution prior to transforming back to custom basis space.

"""
function quadratic_nonlinear_opnn_pseudo(basis,uhat,dL,N);
    u = spectral_approximation(basis,uhat);
    du = fourier_diff(u,N,dL);
    udu = u.*du;
    uduhat = spectral_coefficients(basis,udu);
    return uduhat
end

"""
    generate_fourier_solution(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-3,nu=0.1,rtol=1e-10,atol=1e-14)

Generate the solution for a given `pde_function` and `initial_condition` on a periodic domain using a `N` mode Fourier expansion.

"""
function generate_fourier_solution(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-3,nu=0.1,rtol=1e-10,atol=1e-14)
    # Transform random initial condition to Fourier domain
    uhat0 = fft_norm(initial_condition);
    dL = abs(L2-L1);

    # Generate Fourier Galerkin solution for N
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]);
    p = [N,k,dL,nu]
    t_length = Int(round(tspan[2]/dt)+1);

    # Solve the system of ODEs in Fourier domain
    prob = ODEProblem(pde_function,uhat0,tspan,p);
    sol = solve(prob,DP5(),reltol=rtol,abstol=atol,saveat = dt)

    u_sol = zeros(t_length,N);
    for j in 1:size(sol.t,1) # Reshape output and plot
        u_sol[j,:] = real.(ifft_norm(sol.u[j]));
    end
    return u_sol
end

"""
    generate_basis_solution(L1,L2,t_span,N,basis,initial_condition,pde_function;dt=1e-3,nu=0.1,rtol=1e-10,atol=1e-14)

Generate the solution for a given linear `pde_function` and `initial_conditon` on a periodic domain using a `N` mode custom basis expansion.

"""
function generate_basis_solution(L1,L2,t_span,N,basis,initial_condition,pde_function;dt=1e-3,nu=0.1,rtol=1e-10,atol=1e-14)
    u0 = spectral_coefficients(basis,initial_condition);
    dL = abs(L2-L1);

    Dbasis = fourier_diff(basis,N,dL);
    Dmatrix = spectral_matrix(basis,Dbasis);

    D2basis = fourier_diff(Dbasis,N,dL);
    D2matrix = spectral_matrix(basis,D2basis);

    p = [Dmatrix,D2matrix,nu];

    prob = ODEProblem(pde_function,u0,t_span,p);
    sol = solve(prob,DP5(),reltol=rtol,abstol=atol,saveat = dt);

    u_sol = zeros(size(sol.t,1),size(basis,1));
    for i in 1:size(sol.t,1)
        u_sol[i,:] = spectral_approximation(basis,sol.u[i]);
    end
    return u_sol
end

"""
    generate_basis_solution_nonlinear(L1,L2,t_span,N,basis,initial_condition,pde_function;dt=1e-4,nu=0.1,rtol=1e-10,atol=1e-14)

Generate the solution for a given non-linear `pde_function` and `initial_conditon` on a periodic domain using a `N` mode custom basis expansion.

"""
function generate_basis_solution_nonlinear(L1,L2,t_span,N,basis,initial_condition,pde_function;dt=1e-4,nu=0.1,rtol=1e-10,atol=1e-14)
    u0 = spectral_coefficients(basis,initial_condition);
    dL = abs(L2-L1);

    Dbasis = fourier_diff(basis,N,dL);
    D2basis = fourier_diff(Dbasis,N,dL);#
    D2matrix = spectral_matrix(basis,D2basis);
    p = [D2matrix,basis,dL,N,nu];

    prob = ODEProblem(pde_function,u0,t_span,p);
    sol = solve(prob,DP5(),reltol=rtol,abstol=atol,saveat = dt);

    u_sol = zeros(size(sol.t,1),size(basis,1));
    for i in 1:size(sol.t,1)
        u_sol[i,:] = spectral_approximation(basis,sol.u[i]);
    end
    return u_sol
end

"""
    function central_difference(u_j,u_jpos,u_jneg,mu)

Compute the second order central difference for the viscous term of the viscous Burgers equation \$u_t = - u u_x + ν u_{xx}\$. `mu` is equal to \$ \\frac{\\nu \\Delta t}{\\Delta x^2}\$.

"""
function central_difference(u_j,u_jpos,u_jneg,mu)
    ux = zeros(size(u_j,2));
    ux = mu.*(u_jpos-2*u_j+u_jneg);
    return ux
end

"""
    function backward_upwind(u_j,u_jneg,u_jnegg,nu)

Compute the partial RHS (\$u_j^{n+1} = u_j^n\$ + `backward_upwind`) of the second order backward upwind scheme of Beam and Warming for the inviscid Burgers equation ``u_t = -u u_x`` (split into monotone and non-montone terms), \$-\\nu(u_j^n-u_{j-1}^n) - \\frac{\\nu}{2}(1-\\nu)(u_j^n-u_{j-1}^n) + \\frac{\\nu}{2}(1-\\nu)(u_{j-1}^n-u_{j-2}^n)\$. `nu` is the CFL condition.

"""
function backward_upwind(u_j,u_jneg,u_jnegg,nu)
    ux = zeros(size(u_j,2));
    ux = -nu.*(u_j-u_jneg)-(nu/2).*(1 .-nu).*(u_j-u_jneg)+(nu/2).*(1 .-nu).*(u_jneg-u_jnegg);
    return ux
end

"""
    function forward_upwind(u_j,u_jpos,u_jposs,nu)

Compute the partial RHS (\$u_j^{n+1} = u_j^n\$ + `forward_upwind`) of the second order forward upwind scheme of Beam and Warming for the inviscid Burgers equation ``u_t = -u u_x`` (split into monotone and non-montone terms), \$-\\nu(u_{j+1}^n-u_{j}^n) - \\frac{\\nu}{2}(\\nu+1)(u_{j+1}^n-u_{j}^n) + \\frac{\\nu}{2}(\\nu+1)(u_{j+2}^n-u_{j+1}^n)\$. `nu` is the CFL condition.

"""
function forward_upwind(u_j,u_jpos,u_jposs,nu)
    ux = zeros(size(u_j,2));
    ux = -nu.*(u_jpos-u_j)-(nu/2).*(nu .+1).*(u_jpos-u_j)+(nu/2).*(nu .+1).*(u_jposs-u_jpos);
    return ux
end

"""
    function van_leer_limiter(r)

Compute the Van Leer limiter \$\\Psi = \\frac{r + |r|}{1+r}\$ where ``r`` is the gradient ratio.

"""
function van_leer_limiter(r)
    return (r .+abs.(r))./((1 .+r) .+eps())
end

"""
    function gradient_ratio_backward_j(u_j,u_jneg,u_jpos)

Compute the backward gradient ratio at point \$i\$ as required for the [`van_leer_limiter`](@ref).

"""
function gradient_ratio_backward_j(u_j,u_jneg,u_jpos)
    return (u_jpos .-u_j)./(u_j .-u_jneg);
end

"""
    function gradient_ratio_backward_jneg(u_j,u_jneg,u_jnegg)

Compute the backward gradient ratio at point \$i-1\$ as required for the [`van_leer_limiter`](@ref).

"""
function gradient_ratio_backward_jneg(u_j,u_jneg,u_jnegg)
    return (u_j .-u_jneg)./(u_jneg .-u_jnegg);
end

"""
    function gradient_ratio_forward_j(u_j,u_jneg,u_jpos)

Compute the forward gradient ratio at point \$i\$ as required for the [`van_leer_limiter`](@ref).

"""
function gradient_ratio_forward_j(u_j,u_jneg,u_jpos)
    return (u_j .-u_jneg)./(u_jpos .-u_j);
end

"""
    function gradient_ratio_forward_jpos(u_j,u_jpos,u_jposs)

Compute the forward gradient ratio at point \$i+1\$ as required for the [`van_leer_limiter`](@ref).

"""
function gradient_ratio_forward_jpos(u_j,u_jpos,u_jposs)
    return (u_jpos .-u_j)./(u_jposs .-u_jpos);
end

"""
    function backward_upwind_limited(u_j,u_jneg,u_jnegg,u_jpos,nu)

Compute the partial RHS (\$u_j^{n+1} = u_j^n\$ + `backward_upwind`) of the second order limited backward upwind scheme of Beam and Warming for the inviscid Burgers equation ``u_t = -u u_x`` (split into monotone and non-montone terms), \$-\\nu(u_j^n-u_{j-1}^n) - \\frac{\\nu}{2}(1-\\nu)(u_j^n-u_{j-1}^n) + \\frac{\\nu}{2}(1-\\nu)(u_{j-1}^n-u_{j-2}^n)\$. `nu` is the CFL condition.

"""
function backward_upwind_limited(u_j,u_jneg,u_jnegg,u_jpos,nu)

    r_j = gradient_ratio_backward_j(u_j,u_jneg,u_jpos);
    r_jneg = gradient_ratio_backward_jneg(u_j,u_jneg,u_jnegg);
    psi_j = van_leer_limiter(r_j);
    psi_jneg = van_leer_limiter(r_jneg);

    # psi_j = 1;
    # psi_jneg = 1;

    ux = zeros(size(u_j,2))
    ux = -nu.*(u_j-u_jneg)-(nu/2).*(1 .-nu).*psi_j.*(u_j-u_jneg)+(nu/2).*(1 .-nu).*(psi_jneg).*(u_jneg-u_jnegg);
    return ux
end

"""
    function forward_upwind_limited(u_j,u_jpos,u_jposs,u_jneg,nu)

Compute the partial RHS (\$u_j^{n+1} = u_j^n\$ + `forward_upwind`) of the second order limited forward upwind scheme of Beam and Warming for the inviscid Burgers equation ``u_t = -u u_x`` (split into monotone and non-montone terms), \$-\\nu(u_{j+1}^n-u_{j}^n) - \\frac{\\nu}{2}(\\nu+1)(u_{j+1}^n-u_{j}^n) + \\frac{\\nu}{2}(\\nu+1)(u_{j+2}^n-u_{j+1}^n)\$. `nu` is the CFL condition.

"""
function forward_upwind_limited(u_j,u_jpos,u_jposs,u_jneg,nu)

    r_j = gradient_ratio_forward_j(u_j,u_jneg,u_jpos);
    r_jpos = gradient_ratio_forward_jpos(u_j,u_jpos,u_jposs);
    psi_j = van_leer_limiter(r_j);
    psi_jpos = van_leer_limiter(r_jpos);

    # psi_j = 1;
    # psi_jpos = 1;

    ux = zeros(size(u_j,2));
    ux = -nu.*(u_jpos-u_j)-(nu/2).*(nu .+1).*psi_j.*(u_jpos-u_j)+(nu/2).*(nu .+1).*(psi_jpos).*(u_jposs-u_jpos);
    return ux
end

"""
    function generate_bwlimitersoupwind_solution(L1,L2,t_end,N,initial_condition;dt=1e-4)

Generate second order limited upwind solution for the inviscid Burgers equation ``u_t = -u u_x`` based on the Beam and Warming scheme with a Van Leer limiter.

"""
function generate_bwlimitersoupwind_solution(L1,L2,t_end,N,initial_condition;dt=1e-4)
    dL = abs(L2-L1);
    # Set up periodic domain
    j = reduce(vcat,[0:1:N-1]);
    x = (dL.*j)./N;
    dx = x[2]-x[1];
    t = reduce(vcat,[0:dt:t_end]);
    u_sol = zeros(size(t,1),N);
    u_sol[1,:] = initial_condition;
    for i in 1:(size(t,1)-1) # Loop over time domain
        # Check CFL condition
        dt_dx = abs(maximum(abs.(u_sol[i,:]))*dt/dx);
        if dt_dx > 1
            println("Failed CFL Test!")
            return nothing;
        end

        # Preallocate arrays for each loop
        uneg = zeros(size(u_sol,2));
        upos = zeros(size(u_sol,2));
        uxneg = zeros(size(u_sol,2));
        uxpos = zeros(size(u_sol,2));

        # Solve one step of upwind solution
        upos = max.(u_sol[i,:],zeros(size(u_sol,2)));
        uneg = min.(u_sol[i,:],zeros(size(u_sol,2)));
        nupos = upos*(dt/dx); # CFL condition for each u, u > 0 - also controls which differencing scheme is used
        nuneg = uneg*(dt/dx); # CFL condition for each u, u < 0 - also controls which differencing scheme is used

        uxneg[1] = backward_upwind_limited(u_sol[i,1],u_sol[i,end],u_sol[i,end-1],u_sol[i,2],nupos[1]);
        uxneg[2] = backward_upwind_limited(u_sol[i,2],u_sol[i,1],u_sol[i,end],u_sol[i,3],nupos[2]);
        uxneg[3:end] = backward_upwind_limited(u_sol[i,3:end],u_sol[i,2:end-1],u_sol[i,1:end-2],vcat(u_sol[i,4:end],u_sol[i,1]),nupos[3:end]);

        uxpos[end] = forward_upwind_limited(u_sol[i,end],u_sol[i,1],u_sol[i,2],u_sol[i,end-1],nuneg[end]);
        uxpos[end-1] = forward_upwind_limited(u_sol[i,end-1],u_sol[i,end],u_sol[i,1],u_sol[i,end-2],nuneg[end-1]);
        uxpos[1:end-2] = forward_upwind_limited(u_sol[i,1:end-2],u_sol[i,2:end-1],u_sol[i,3:end],vcat(u_sol[i,end],u_sol[i,1:end-3]),nuneg[1:end-2]);

        u_sol[i+1,:] = u_sol[i,:] .+ (uxneg .+ uxpos);

    end
    return u_sol
end

"""
    function generate_bwlimitersoupwind_viscous_solution(L1,L2,t_end,N,initial_condition;dt=1e-4,nu=0.1)

Generate second order solution for the viscous Burgers equation \$u_t = - u u_x + ν u_{xx}\$ based on the Beam and Warming second order upwind with a Van Leer limiter for the convective term and a second order central differnce for the diffusive term. `nu` is the viscosity.

"""
function generate_bwlimitersoupwind_viscous_solution(L1,L2,t_end,N,initial_condition;dt=1e-4,nu=0.1)
    dL = abs(L2-L1);
    # Set up periodic domain
    j = reduce(vcat,[0:1:N-1]);
    x = (dL.*j)./N;
    dx = x[2]-x[1];
    t = reduce(vcat,[0:dt:t_end]);
    u_sol = zeros(size(t,1),N);
    u_sol[1,:] = initial_condition;
    for i in 1:(size(t,1)-1) # Loop over time domain
        # Check CFL condition
        dt_dx = abs(maximum(abs.(u_sol[i,:]))*dt/dx);
        if dt_dx > 1
            println("Failed CFL Test!")
            return nothing;
        end

        # Preallocate arrays for each loop
        uneg = zeros(size(u_sol,2));
        upos = zeros(size(u_sol,2));
        uxneg = zeros(size(u_sol,2));
        uxpos = zeros(size(u_sol,2));
        ux_viscous = zeros(size(u_sol,2));

        # Solve one step of upwind solution
        upos = max.(u_sol[i,:],zeros(size(u_sol,2)));
        uneg = min.(u_sol[i,:],zeros(size(u_sol,2)));
        nupos = upos*(dt/dx); # CFL condition for each u, u > 0 - also controls which differencing scheme is used
        nuneg = uneg*(dt/dx); # CFL condition for each u, u < 0 - also controls which differencing scheme is used
        mu = (nu*dt)/(dx)^2;

        uxneg[1] = backward_upwind_limited(u_sol[i,1],u_sol[i,end],u_sol[i,end-1],u_sol[i,2],nupos[1]);
        uxneg[2] = backward_upwind_limited(u_sol[i,2],u_sol[i,1],u_sol[i,end],u_sol[i,3],nupos[2]);
        uxneg[3:end] = backward_upwind_limited(u_sol[i,3:end],u_sol[i,2:end-1],u_sol[i,1:end-2],vcat(u_sol[i,4:end],u_sol[i,1]),nupos[3:end]);

        uxpos[end] = forward_upwind_limited(u_sol[i,end],u_sol[i,1],u_sol[i,2],u_sol[i,end-1],nuneg[end]);
        uxpos[end-1] = forward_upwind_limited(u_sol[i,end-1],u_sol[i,end],u_sol[i,1],u_sol[i,end-2],nuneg[end-1]);
        uxpos[1:end-2] = forward_upwind_limited(u_sol[i,1:end-2],u_sol[i,2:end-1],u_sol[i,3:end],vcat(u_sol[i,end],u_sol[i,1:end-3]),nuneg[1:end-2]);

        ux_viscous[1] = central_difference(u_sol[i,1],u_sol[i,2],u_sol[i,end],mu);
        ux_viscous[2:end-1] = central_difference(u_sol[i,2:end-1],u_sol[i,3:end],u_sol[i,1:end-2],mu);
        ux_viscous[end] = central_difference(u_sol[i,end],u_sol[i,1],u_sol[i,end-1],mu);

        u_sol[i+1,:] = u_sol[i,:] .+ (uxneg .+ uxpos) .+ ux_viscous;
    end
    return u_sol
end

"""
    minmod(x,y)

"""
function minmod(x,y)
    return sign(x)*max(0,min(abs(x),y*sign(x)));
end

"""
    function ub(ulv,urv)

"""
function ub(ulv,urv)
    if ulv != urv
        return ((1/2)*urv^2 - (1/2)*ulv^2) / (urv - ulv); # ̄u_{j+/-1/2}
    else
        return ulv
    end
end

"""
    function fl(ulv,urv,ubv)

"""
function fl(ulv,urv,ubv)
    return (1/2).*((1/2).*ulv.^2 .+ (1/2).*urv.^2 .- abs.(ubv).*(urv .- ulv)); # f_{j+/-1/2}
# fln(ulv,urv,ubv) = (1/2)*(flux.(urv) - flux.(ulv) - abs.(ubv).*(ulv - urv)); # f_{j+/-1/2}
end

"""
    function ulpl(ujp,uj,ujn,kappa,omega)

"""
function ulpl(ujp,uj,ujn,kappa,omega)
    return uj .+ ((1-kappa)/4).*minmod.((uj .- ujn),omega.*(ujp .- uj)) .+ ((1+kappa)/4).*minmod.((ujp .- uj),omega.*(uj .- ujn)); # u_{j+1/2}^L
end

"""
    function urpl(ujpp,ujp,uj,kappa,omega)

"""
function urpl(ujpp,ujp,uj,kappa,omega)
    return ujp  .- ((1+kappa)/4).*minmod.((ujp .- uj),omega.*(ujpp .- ujp)) .- ((1-kappa)/4).*minmod.((ujpp .- ujp),omega.*(ujp .- uj)); # u_{j+1/2}^R
end

"""
    function ulnl(uj,ujn,ujnn,kappa,omega)

"""
function ulnl(uj,ujn,ujnn,kappa,omega)
    return ujn .+ ((1-kappa)/4).*minmod.((ujn .- ujnn),omega.*(uj .- ujn)) .+ ((1+kappa)/4).*minmod.((uj .- ujn),omega.*(ujn .- ujnn)); # u_{j-1/2}^L
end

"""
    function urnl(ujp,uj,ujn,kappa,omega)

"""
function urnl(ujp,uj,ujn,kappa,omega)
    return uj .- ((1+kappa)/4).*minmod.((uj .- ujn),omega.*(ujp .- uj)) .- ((1-kappa)/4).*minmod.((ujp .- uj),omega.*(uj .- ujn)); # u_{j-1/2}^R
end

"""
    function muscl_minmod_RHS!(du,u,p,t)

"""
function muscl_minmod_RHS!(du,u,p,t)
    dx, kappa, omega = p;

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
    function generate_muscl_minmod_solution(L1,L2,t_end,N,u0;dt=1e-4,kappa=-1)

"""
function generate_muscl_minmod_solution(L1,L2,t_end,N,u0;dt=1e-4,kappa=-1)

    omega = ((3-kappa)/(1-kappa))

    dL = abs(L2-L1);
    # Set up periodic domain
    j = reduce(vcat,[0:1:N-1]);
    x = (dL.*j)./N;
    dx = x[2]-x[1];
    t = reduce(vcat,[0:dt:t_end]);

    p = [dx,kappa,omega]
    t_span = (0,t_end);

    prob = ODEProblem(muscl_minmod_RHS!,u0,t_span,p);
    sol = solve(prob,BS3(),reltol=1e-6,abstol=1e-8,saveat = dt);

    u_sol = zeros(size(sol.t,1),size(x,1));
    for i in 1:size(sol.t,1)
        u_sol[i,:] = sol.u[i];
    end

    return u_sol

end

"""
    function get_1D_energy_fft(u_solution)

Compute the energy in the Fourier domain using the scaling of \$ \\frac{1}{N} \$.

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
    function get_1D_energy_basis(basis,u_solution)

Compute the energy in the custom basis space. The expansion coefficients are scaled by \$ \\frac{1}{\\sqrt(N)} \$ for comparison to the non-unitary Fourier coefficients.

"""
function get_1D_energy_basis(basis,u_solution) # To Do: Add and extraction of specific mode step
    energy = zeros(size(u_solution,1))
    for i in 1:size(u_solution,1)
        u_hat = (1/sqrt(size(basis,2)))*spectral_coefficients(basis,u_solution[i,:]);
        energy[i] = (1/2)*real((u_hat'*u_hat));
    end
    return energy
end
