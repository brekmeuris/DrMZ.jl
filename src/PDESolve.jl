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
    opnn_advection_pde!(du,u,p,t_span)

"""
function opnn_advection_pde!(du,u,p,t_span)
    basis, N, dL = p;
    uapprox = spectral_approximation(basis,u)
    duapprox = fourier_diff(uapprox,N,dL)
    du .= -spectral_coefficients(basis,duapprox)
end
