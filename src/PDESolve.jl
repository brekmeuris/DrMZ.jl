"""
    advection_pde!(duhat,uhat,k,t)

RHS for the advection equation for numerical integration in Fourier space.

input: dû/dt, û, N, delta L, t_span

output: dû/dt

"""
function advection_pde!(duhat,uhat,p,t)
    N, dL = p;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -im*k.*uhat;
end

"""
    fourier_diff(sol,N,dL)


"""
function fourier_diff(sol,N,dL;format="matrix")
    if format == "matrix"
        h = 2*pi/N;
        col = vcat(0, 0.5*(-1).^(1:N-1).*cot.((1:N-1)*h/2));
        row = vcat(col[1], col[N:-1:2]);
        diff_matrix = Toeplitz(col,row);
        diff_sol = (2*pi/dL)*diff_matrix*sol; # Make dx calc abs...
    else
        k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
        sol_k = fft(sol);
        sol_k[Int(N/2)+1] = 0;
        diff_sol = real.(ifft(im*k.*sol_k));
    end
    return diff_sol
end

"""
    cheby_grid(N,a,b)

"""
function cheby_grid(N,a,b)
    x = ((b+a)/2).+((b-a)/2)*cos.(pi*(0:N)/N);
    return x
end

"""
    cheby_diff_matrix(N,a,b)

"""
function cheby_diff_matrix(N,a,b)
    if N == 0
        D = 0;
        x = 1;
        return D, x
    else
        x = ((b+a)/2).+((b-a)/2)*cos.(pi*(0:N)/N);
        c = vcat(2, ones(N-1,1), 2).*(-1).^(0:N);
        X = repeat(x,1,N+1);
        dx = X-X';
        D = (c*(1 ./c)')./(dx+I);
        D = D - diagm(0 => sum(D,dims = 2)[:]);
        return D; x
    end
end

"""
    cheby_diff(sol,N,dL)


"""
function cheby_diff(sol,N,L1,L2)
    D, x = cheby_diff_matrix(N,L1,L2);
    diff_sol = D*sol; # Make dx calc abs...
    return diff_sol
end

"""
    opnn_advection_pde!(du,u,p,t_span)

"""
# function opnn_advection_pde!(du,u,p,t_span)
#     basis, N, dL = p;
#     uapprox = spectral_approximation(basis,u)
#     duapprox = fourier_diff(uapprox,N,dL)
#     du .= -spectral_coefficients(basis,duapprox)
# end
function opnn_advection_pde!(du,u,p,t)
    Dmatrix, D2matrix = p;
    du .= -Dmatrix*u;
end

# """
#     opnn_advection_pde_full!(du,u,p,t_span)
#
# """
# function opnn_advection_pde_full!(du,u,p,t_span)
#     basis, N, dL = p;
#     uapprox = spectral_approximation(basis,u);
#     duapprox = zeros(N);
#     duapprox[1:(N-1)] = fourier_diff(uapprox[1:(N-1)],(N-1),dL);
#     duapprox[N] = duapprox[1];
#     du .= -spectral_coefficients(basis,duapprox);
# end

"""
    advection_diffusion_pde!(duhat,uhat,p,t_span)

"""
function advection_diffusion_pde!(duhat,uhat,p,t)
    N, dL = p;
    D = 0.1; #0.015
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -im*k.*uhat + D*(im*k).^2 .*uhat;
end

"""
    opnn_advection_diffusion_pde!(du,u,p,t_span)

"""
function opnn_advection_diffusion_pde!(du,u,p,t)
    Dmatrix, D2matrix = p;
    D = 0.1; #0.015
    # uapprox = spectral_approximation(basis,u);
    # duapprox = fourier_diff(uapprox,N,dL);
    # duapprox2 = fourier_diff(duapprox,N,dL);
    # du .= -spectral_coefficients(basis,duapprox) + D*spectral_coefficients(basis,duapprox2);
    du .= -Dmatrix*u .+ D*D2matrix*u;
end

# """
#     opnn_advection_diffusion_pde_full!(du,u,p,t_span)
#
# """
# function opnn_advection_diffusion_pde_full!(du,u,p,t_span)
#     basis, N, dL = p;
#     D = 0.015;
#     uapprox = spectral_approximation(basis,u);
#     duapprox = zeros(N);
#     duapprox2 = zeros(N);
#     duapprox[1:(N-1)] = fourier_diff(uapprox[1:(N-1)],(N-1),dL);
#     duapprox2[1:(N-1)] = fourier_diff(duapprox[1:(N-1)],(N-1),dL);
#     duapprox[N] = duapprox[1];
#     duapprox2[N] = duapprox2[1];
#     du .= -spectral_coefficients(basis,duapprox) + D*spectral_coefficients(basis,duapprox2);
# end

"""

"""
function generate_fourier_solution(L1,L2,tspan,N,initial_condition,pde_function;dt=1e-3,rtol=1e-10,atol=1e-14)
    # Transform random initial condition to Fourier domain
    # uhat0 = fft(initial_condition);
    uhat0 = fft_norm(initial_condition);
    dL = abs(L2-L1);

    # Generate Fourier Galerkin solution for N
    p = [N,dL];
    t_length = tspan[2]/dt+1;

    # Solve the system of ODEs in Fourier domain
    prob = ODEProblem(pde_function,uhat0,tspan,p);
    sol = solve(prob,DP5(),reltol=rtol,abstol=atol,saveat = dt)

    u_sol = zeros(Int(t_length),N);
    for j in 1:size(sol.t,1) # Reshape output and plot
        # u_sol[j,:] = real.(ifft(sol.u[j])); # u[t,x,IC]
        u_sol[j,:] = real.(ifft_norm(sol.u[j])); # u[t,x,IC]
    end
    return u_sol
end

"""

"""
function generate_fourier_solution_stiff(L1,L2,tspan,N,initial_condition,pde_linear,pde_function_explicit;dt_fixed=1e-3,rtol=1e-10,atol=1e-14)
    # Transform random initial condition to Fourier domain
    # uhat0 = fft(initial_condition);
    uhat0 = fft_norm(initial_condition);
    dL = abs(L2-L1);

    # Generate Fourier Galerkin solution for N
    p = [N,dL];
    t_length = tspan[2]/dt_fixed+1;

    # Solve the system of ODEs in Fourier domain
    prob = SplitODEProblem(pde_linear,pde_function_explicit,uhat0,tspan,p);
    sol = solve(prob,ETDRK4(),reltol=rtol,abstol=atol,dt = dt_fixed,saveat = dt_fixed)

    u_sol = zeros(Int(t_length),N);
    for j in 1:size(sol.t,1) # Reshape output and plot
        # u_sol[j,:] = real.(ifft(sol.u[j])); # u[t,x,IC]
        u_sol[j,:] = real.(ifft_norm(sol.u[j])); # u[t,x,IC]
    end
    return u_sol
end

"""
    quadratic_nonlinear!(uhat)

"""
function quadratic_nonlinear(uhat,N,dL,alpha)
    # N = Int(size(uhat,1)); # Instead of passing in N?
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
    kdv_pde!(duhat,uhat,p,t_span)

"""
function kdv_pde!(duhat,uhat,p,t)
    N, dL = p;
    alpha = 1.0;
    epsilon = (0.1)^2;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0.0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= epsilon*im.*k.^3 .*uhat .- uhat_nonlinear;
end

"""
    kdv_implicit_pde!(duhat,uhat,p,t_span)

"""
function kdv_implicit_pde!(duhat,uhat,p,t)
    N, dL = p;
    alpha = 1.0;
    epsilon = (0.1)^2;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry;
    duhat .= epsilon*im.*k.^3 .*uhat;
end

"""
    kdv_explicit_pde!(duhat,uhat,p,t_span)

"""
function kdv_explicit_pde!(duhat,uhat,p,t)
    N, dL = p;
    alpha = 1.0;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= -uhat_nonlinear;
end

"""
    viscous_burgers_explicit_pde!(duhat,uhat,p,t_span)

"""
function viscous_burgers_explicit_pde!(duhat,uhat,p,t)
    N, dL = p;
    alpha = 1.0;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= -uhat_nonlinear;
end

"""
    inviscid_burgers_pde!(duhat,uhat,p,t_span)

"""
function inviscid_burgers_pde!(duhat,uhat,p,t)
    N, dL = p;
    alpha = 1.0;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    uhat_nonlinear = quadratic_nonlinear(uhat,N,dL,alpha);
    duhat .= -uhat_nonlinear;
end
