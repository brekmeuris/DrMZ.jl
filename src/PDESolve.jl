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
