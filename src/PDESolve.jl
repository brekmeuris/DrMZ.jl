"""
    advection_pde!(duhat,uhat,k,t)

RHS for the advection equation for numerical integration in Fourier space.

input: dû/dt, û, N, delta L, t_span

output: dû/dt

"""
function advection_pde!(duhat,uhat,p,t_span)
    N, dL = p;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -im*k.*uhat;
end

"""
    fourier_diff(sol,N,dL)


"""
function fourier_diff(sol,N,dL)
    h = 2*pi/N;
    col = vcat(0, 0.5*(-1).^(1:N-1).*cot.((1:N-1)*h/2));
    row = vcat(col[1], col[N:-1:2]);
    diff_matrix = Toeplitz(col,row);
    diff_sol = (2*pi/dL)*diff_matrix*sol; # Make dx calc abs...
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
function opnn_advection_pde!(du,u,p,t_span)
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
function advection_diffusion_pde!(duhat,uhat,p,t_span)
    N, dL = p;
    D = 0.015;
    k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
    uhat[Int(N/2)+1] = 0; # Set the most negative mode to zero to prevent an asymmetry
    duhat .= -im*k.*uhat + D*(im*k).^2 .*uhat;
end

"""
    opnn_advection_diffusion_pde!(du,u,p,t_span)

"""
function opnn_advection_diffusion_pde!(du,u,p,t_span)
    Dmatrix, D2matrix = p;
    D = 0.015;
    # uapprox = spectral_approximation(basis,u);
    # duapprox = fourier_diff(uapprox,N,dL);
    # duapprox2 = fourier_diff(duapprox,N,dL);
    # du .= -spectral_coefficients(basis,duapprox) + D*spectral_coefficients(basis,duapprox2);
    du .= -Dmatrix*u + D*D2matrix*u;
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
