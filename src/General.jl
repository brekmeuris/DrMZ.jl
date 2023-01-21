"""
    error_se(target,prediction)

Compute the squared error between the `prediction` and `target` values.

"""
function error_se(target,prediction)
    error_se = zeros(size(target,1),size(target,2));
    for j in 1:size(target,1)
        for i in 1:size(target,2)
            error_se[j,i] = (prediction[j,i] - target[j,i])^2;
        end
    end
    return error_se
end

"""
    error_rel(target,prediction)

Compute the relative error between the `prediction` and `target` values.

"""
function error_rel(target,prediction)
    error_rel = zeros(size(target,1),size(target,2));
    for j in 1:size(target,1)
        for i in 1:size(target,2)
            error_test_rel[j,i] = (prediction[j,i] - target[j,i])/target[j,i];
        end
    end
    return error_rel
end

"""
    mse_error(target,prediction)

Compute the mean squared error between the `prediction` and `target` values.

"""
function mse_error(target,prediction)
    error_test_mse = 0.0;
    for j in 1:size(target,1)
        for i in 1:size(target,2)
            error_test_mse += (prediction[j,i] - target[j,i])^2;
        end
    end
    return (1/(size(target,1)*size(target,2)))*error_test_mse
end

"""
    norm_rel_error(target,prediction)

Compute the Euclidean two-norm relative error between the `prediction` and `target` values.

"""
function norm_rel_error(target,prediction)
    if size(target) != size(prediction)
        error("Target size and prediction size do not match!")
    end
    error = zeros(size(target,1))
    for i in 1:size(target,1)
        error[i] = norm(prediction[i,:]-target[i,:])/norm(target[i,:]);
    end
    return error
end

"""
    function norm_rel_error_continuous(target,prediction,weights)

Compute the continuous two-norm relative error between the `prediction` and `target` values.

"""
function norm_rel_error_continuous(target,prediction,weights)
  error = zeros(size(target,1))
  W = diagm(0=>weights);
  for i in 1:size(target,1)
    diff = prediction[i,:]-target[i,:];
    error[i] = sqrt(diff'*W*diff)/sqrt(target[i,:]'*W*target[i,:]);
  end
  return error
end

"""
    norm_infinity_error(target,prediction)

Compute the two-norm relative error between the `prediction` and `target` values.

"""
function norm_infinity_error(target,prediction)
  error = zeros(size(target,1))
  for i in 1:size(target,1)
    diff = prediction[i,:]-target[i,:];
    error[i] = maximum(abs.(diff),dims=1)[1];
  end
  return error
end

"""
    average_error(domain,error)

Compute the average error using the trapezoid rule \$\\frac{1}{T} \\int_0^T error(t) dt\$.

"""
function average_error(domain,error)
    return 1/(domain[end]-domain[1])*trapz(domain,error)
end

"""
    ic_error(target,prediction)

Compute the relative error between the `prediction` and `target` values for an initial condition. If the `target` = 0, the `target` in denomenator is augmented by \$\\epsilon_{machine}\$.

"""
function ic_error(target,prediction)
    error = zeros(size(target,1));
    for i in 1:size(target,1)
        error[i] = (prediction[i]-target[i])/(target[i] + eps());
    end
    return error
end

"""
    average_ic_error(target,prediction)

Compute the two-norm relative error between the `prediction` and `target` values for an initial condition.

"""
function average_ic_error(target,prediction)
    return norm(prediction-target)/norm(target)
end

"""
    periodic_fill_domain(x_locations)

Output the full domain from periodic domain specified for `x_locations`.

"""
function periodic_fill_domain(x_locations)
    x_locations_full = vcat(x_locations,(x_locations[end]+(x_locations[end]-x_locations[end-1])))
    return x_locations_full
end

"""
    periodic_fill_solution(solution)

Output the full ``u(t,x)`` solution from periodic `solution`.

"""
function periodic_fill_solution(solution)
    solution_full = zeros(size(solution,1),(size(solution,2)+1));
    for i in 1:size(solution,1)
        solution_full[i,1:end-1] = solution[i,:];
        solution_full[i,end] = solution[i,1];
    end
    return solution_full
end

"""
    solution_interpolation(t_span_original,x_locations_original,t_span_interpolate,x_locations_interpolate,solution)

Compute the ``u(t,x)`` solution at intermediate ``(t,x)`` locations using linear interpolation.

"""
function solution_interpolation(t_span_original,x_locations_original,t_span_interpolate,x_locations_interpolate,solution)
    solution_interpolate_fit = interpolate((t_span_original,x_locations_original),solution,Gridded(Linear()));
    solution_interpolate = zeros(size(t_span_interpolate,1),size(x_locations_interpolate,1));
    for i in 1:size(t_span_interpolate,1)
        for j in 1:size(x_locations_interpolate,1)
            solution_interpolate[i,j] = solution_interpolate_fit(t_span_interpolate[i],x_locations_interpolate[j]);
        end
    end
    return solution_interpolate
end

"""
    reduced_initial_condition(L1,L2,x_reduced,x_locations,initial_condition)

Extract the ``x`` locations and intial condition values ``u(x)`` at a reduced number of equally spaced spatial locations.

"""
function reduced_initial_condition(L1,L2,x_reduced,x_locations,initial_condition)
    # dL = abs(L2-L1);
    # j = reduce(vcat,[0:1:N-1]);
    # x_reduced = (dL.*j)./N;
    ind = [findall(x_reduced[i].==x_locations)[1] for i in 1:length(x_reduced)];
    ic_reduced = zeros(size(ind,1));
    for i in 1:size(ind,1)
        ic_reduced[i] = initial_condition[ind[i]];
    end
    return ic_reduced
end

"""
    fourier_diff(sol,N,dL;format="matrix")

Compute the derivative using a Fourier differentiation matrix (default) or the spectral derivative for periodic functions for domain length `dL` and with `N` discretization points.

"""
function fourier_diff(sol,N,dL;format="matrix")
    if format == "matrix"
        h = 2*pi/N;
        col = vcat(0, 0.5*(-1).^(1:N-1).*cot.((1:N-1)*h/2));
        row = vcat(col[1], col[N:-1:2]);
        diff_matrix = Toeplitz(col,row);
        diff_sol = (2*pi/dL)*diff_matrix*sol;
    elseif format == "spectral"
        k = reduce(vcat,(2*π/dL)*[0:N/2-1 -N/2:-1]); # Wavenumbers
        sol_k = fft(sol);
        sol_k[Int(N/2)+1] = 0;
        diff_sol = real.(ifft(im*k.*sol_k));
    end
    return diff_sol
end

"""
    trapz(x_range,integrand)

Numerical integration using the multi-application trapezoidal rule.

"""
function trapz(x_range,integrand)
    int = 0.0;
    for i = 2:length(x_range)
        int += ((integrand[i-1] + integrand[i])/2)*(x_range[i] - x_range[i-1]);
    end
    return int
end

"""
    solution_spatial_sampling(x_prediction,x_target,solution)

Extract the solution values ``u(t,x)`` at a reduced number of equally spaced spatial locations.

"""
function solution_spatial_sampling(x_prediction,x_target,solution)
    ind = [findall(x_prediction[i].==x_target)[1] for i in 1:length(x_prediction)];
    solution_reduced = zeros(size(solution,1),size(ind,1));
    for i in 1:size(ind,1)
        solution_reduced[:,i] = solution[:,ind[i]]
    end
    return solution_reduced
end

"""
    solution_temporal_sampling(t_prediction,t_target,solution)

Extract the solution values ``u(t,x)`` at a reduced number of equally spaced temporal locations.

"""
function solution_temporal_sampling(t_prediction,t_target,solution)
    ind = [findall(t_prediction[i].==t_target)[1] for i in 1:length(t_prediction)];
    solution_reduced = zeros(size(ind,1),size(solution,2));
    for i in 1:size(ind,1)
        solution_reduced[i,:] = solution[ind[i],:]
    end
    return solution_reduced
end

"""
    fft_norm(solution)

Compute the FFT normalized by \$\\frac{1}{N}\$.

"""
function fft_norm(solution)
    N = size(solution,1);
    return (1/N)*fft(solution)
end

"""
    ifft_norm(solution)

Compute the IFFT normalized by \$N\$.

"""
function ifft_norm(solution)
    N = size(solution,1);
    return N*ifft(solution)
end

"""
    gauss_legendre(N,L1,L2)

Compute the nodes and weights for Gauss-Legendre quadrature with `N` discretization points and on the interval ``[L1,L2]``.

"""
function gauss_legendre(N,L1,L2)
    nodes, weights = gausslegendre(N);
    return ((L1+L2)/2).+((L2-L1)/2)*nodes, ((L2-L1)/2)*weights;
end

"""
    clenshaw_curtis(N,L1,L2)

Compute the nodes and weights for Clenshaw-Curtis quadrature with `N` discretization points and on the interval ``[L1,L2]``.

"""
function clenshaw_curtis(N,L1,L2)
    theta = pi*(0:N)/N;
    x = -cos.(theta);
    w = zeros(size(x,1));
    ii = 2:N;
    v = ones(N-1);
    if mod(N,2) == 0
        w[1] = 1/(N^2-1);
        w[N+1] = w[1];
        for k in 1:Int(N/2)-1
            v = v - 2*cos.(2*k*theta[ii])/(4*k^2-1);
        end
        v = v - cos.(N*theta[ii])/(N^2-1);
    else
        w[1] = 1/(N^2);
        w[N+1] = w[1];
        for k in 1:Int((N-1)/2)
            v = v - 2*cos.(2*k*theta[ii])/(4*k^2-1);
        end
    end
    w[ii] = 2*v/N;
    return ((L1+L2)/2).+((L2-L1)/2)*x, (L2-L1)/2*w
end

"""
    cheby_grid(N,L1,L2)

Generate the grid of Chebyshev points on the interval ``[L1,L2]`` with `N` discretization points.

"""
function cheby_grid(N,L1,L2)
    x = ((L1+L2)/2).+((L2-L1)/2)*(-cos.(pi*(0:N)/N));
    return x
end

"""
    cheby_diff_matrix(N,L1,L2)

Generate the Chebyshev differentiation matrix for the interval ``[L1,L2]`` with `N` discretization points.

"""
function cheby_diff_matrix(N,L1,L2)
    if N == 0
        D = 0;
        x = 1;
        return D, x
    else
        x = ((L1+L2)/2).+((L2-L1)/2)*(-cos.(pi*(0:N)/N));
        c = vcat(2, ones(N-1,1), 2).*(-1).^(0:N);
        X = repeat(x,1,N+1);
        dx = X-X';
        D = (c*(1 ./c)')./(dx+I);
        D = D - diagm(0 => sum(D,dims = 2)[:]);
        return D, x
    end
end

"""
    cheby_diff(sol,N,L1,L2)

Compute the derivative of using a Chebyshev differentiation matrix on the interval ``[L1,L2]`` with `N` discretization points.

"""
function cheby_diff(sol,N,L1,L2)
    D, x = cheby_diff_matrix(N,L1,L2);
    diff_sol = D*sol;
    return diff_sol
end

"""
    trapezoid(N,L1,L2)

Compute the nodes and weights for trapezoid rule with `N` discretization points and on the interval ``[0,L2]``. TO DO: Revise for [L1,L2] support.

"""
function trapezoid(N,L1,L2)
    if L1<0
        error("Currently set up only for interval [0,L2]")
    end
    dL = abs(L2-L1);
    j = reduce(vcat,[0:1:N-1]);
    x = (dL.*j)./N;
    wi = ((L2-L1)/N);
    w = repeat([wi],N);
    return x, w
end

"""
    orthonormal_check(basis,weights;tol = 1e-12)

"""
function orthonormal_check(basis,weights;tol = 1e-12)
    W = diagm(0 => weights);
    for i = 1:size(basis,2)
        for j = 1:size(basis,2)
            if i == j
                if abs(sqrt(basis[:,i]'*W*basis[:,i]) - 1) > tol
                    error("Not orthonormal to $tol... $i")
                end
            elseif i != j
                if abs(basis[:,j]'*W*basis[:,i]) > tol
                    error("Not orthogonal to $tol... $i vs $j \n Review the singular value spectrum...")
                end
            end
        end
    end
end

"""
    legendre_norm(x,L1,L2,n)

Compute the `n`-th orthonormal, shifted Legendre polynomial

"""
function legendre_norm(x,L1,L2,n)
    return ((2*n+1)/(L2-L1))^(1/2)*Pl(((2*x-L1-L2)/(L2-L1)),n)
end

"""
    dlegendre_norm(x,L1,L2,n)

Compute the 1st derivative of the `n`-th shifted Legendre polynomial 

"""
function dlegendre_norm(x,L1,L2,n)
    return (2/(2*n+1))*((2*n+1)/(L2-L1))^(3/2)*dnPl(((2*x-L1-L2)/(L2-L1)),n,1)
end

"""
    legendre_norm_basis_build(nmax,L1,L2)

"""
function legendre_norm_basis_build(nmax,L1,L2)
    basis = []
    for i in 1:nmax
        push!(basis,(y)->legendre_norm(y,L1,L2,i-1))
    end
    return basis
end

"""
    dlegendre_norm_basis_build(nmax,L1,L2)

"""
function dlegendre_norm_basis_build(nmax,L1,L2)
    Dbasis = []
    for i in 1:nmax
        push!(Dbasis,(y)->dlegendre_norm(y,L1,L2,i-1))
    end
    return Dbasis
end

"""
    linear_reg(x,y)

"""
function linear_reg(x,y)
    X = [ones(size(x,1)) x];
    return (X'*X)\X' * y
  end