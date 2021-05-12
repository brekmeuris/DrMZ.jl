"""
    advection_pde!(duhat,uhat,p,t)

RHS for the advection equation ``u_t = - u_x`` for numerical integration in Fourier space.

"""
function advection_pde!(duhat,uhat,p,t)
    N, k, nu = p;
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -im*k.*uhat;
end

"""
    advection_diffusion_pde!(duhat,uhat,p,t)

RHS for the advection-diffusion equation \$u_t = - u_x + ν u_{xx}\$ for numerical integration in Fourier space where ν is the viscosity.

"""
function advection_diffusion_pde!(duhat,uhat,p,t)
    N, k, nu = p;
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -im*k.*uhat + nu*(im*k).^2 .*uhat;
end

"""
    viscous_burgers_pde!(duhat,uhat,p,t)

RHS for the viscous Burgers equation \$u_t = - u u_x + ν u_{xx}\$ for numerical integration in Fourier space where ν is the viscosity.

"""
function viscous_burgers_pde!(duhat,uhat,p,t)
    N, k, nu = p;
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
    N, k, nu = p;
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
    generate_fourier_solution(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-3,,nu=0.1,rtol=1e-10,atol=1e-14)

Generate the solution for a given `pde_function` and `initial_condition` on a periodic domain using a `N` mode Fourier expansion.

"""
function generate_fourier_solution(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-3,nu=0.1,rtol=1e-10,atol=1e-14)
    # Transform random initial condition to Fourier domain
    uhat0 = fft_norm(initial_condition);
    dL = abs(L2-L1);

    # Generate Fourier Galerkin solution for N
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]);
    p = [N,k,nu]
    t_length = tspan[2]/dt+1;

    # Solve the system of ODEs in Fourier domain
    prob = ODEProblem(pde_function,uhat0,tspan,p);
    sol = solve(prob,DP5(),reltol=rtol,abstol=atol,saveat = dt)

    u_sol = zeros(Int(t_length),N);
    for j in 1:size(sol.t,1) # Reshape output and plot
        u_sol[j,:] = real.(ifft_norm(sol.u[j]));
    end
    return u_sol
end

"""
    generate_basis_solution(L1,L2,t_span,N,basis,initial_condition,pde_function;dt=1e-3,rtol=1e-10,atol=1e-14)

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
    generate_basis_solution_nonlinear(L1,L2,t_span,N,basis,initial_condition,pde_function;dt=1e-4,rtol=1e-10,atol=1e-14)

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

Compute the second order central difference for the viscous term of the viscous Burgers equation \$u_t = - u u_x + ν u_{xx}\$. `mu` is equal to \$ \\frac{\\nu *\\Delta t}{\\Delta x^2}\$.

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
    return (r .+abs.(r))./(1 .+r .+eps())
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
        dt_dx = abs(maximum(u_sol[i,:])*dt/dx);
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
    dt_dx = abs(dt/dx);
    return u_sol, dt_dx
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
        dt_dx = abs(maximum(u_sol[i,:])*dt/dx);
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
    dt_dx = abs(dt/dx);
    return u_sol, dt_dx
end
