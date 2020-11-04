"""
    function basis_OpNN(branch,trunk,initial_condition,x_locations)

"""
function basis_OpNN(branch,trunk,initial_condition,x_locations,n)
    basis = zeros(size(x_locations,1)); # make size of ic...
    for i in 1:size(x_locations,1);
        # basis[i] = transpose(branch(initial_condition)[n])*trunk(vcat(0,x_locations[i]))[n];
        basis[i] = branch(initial_condition)[n]'*trunk(vcat(0,x_locations[i]))[n];
    end
    return basis
end

"""
    build_basis(branch,trunk,intial_condition,x_locations;tol = 1e-15)

"""
function build_basis(branch,trunk,initial_condition,x_locations;tol = 1e-15)

    basis = zeros(length(x_locations),size(initial_condition,1));
    for i in 1:size(initial_condition,1)
        basis[:,i] = basis_OpNN(branch,trunk,initial_condition,x_locations,i)
    end

    norm_basis = [norm(basis[:,i]) for i in 1:size(basis,2)];
    norm_sort = reverse(sortperm(norm_basis));
    sorted_basis = zeros(size(basis,1),size(basis,2));
    for i in 1:size(basis,2)
        indsort = norm_sort[i];
        sorted_basis[:,i] = basis[:,indsort];
    end

    F = qr(sorted_basis);
    orthonormal_basis_full = F.Q*Matrix(I, size(sorted_basis)...)
    orthonormal_basis = orthonormal_basis_full[:,1:length(x_locations)]

    # Orthogonality and... orthonormal checks
    for i = 1:size(orthonormal_basis,2)
        for j = 1:size(orthonormal_basis,2)
            if i == j
                if abs(norm(orthonormal_basis[:,i]) - 1) > tol
                    println("Not orthonormal to $tol... $i")
                    return nothing
                    break
                end
            elseif i != j
                # if abs(orthonormal_basis[:,i]'*conj(orthonormal_basis[:,j])) > tol# || (efuncs_orthonorm[:,i]'*conj(efuncs_orthonorm[:,j]) < eps())
                if abs(orthonormal_basis[:,i]'*orthonormal_basis[:,j]) > tol# || (efuncs_orthonorm[:,i]'*conj(efuncs_orthonorm[:,j]) < eps())
                    println("Not orthogonal to $tol... $i vs $j")
                    return nothing
                    break
                end
            end
        end
    end

    return orthonormal_basis, sorted_basis, basis
end

"""
    spectral_coefficients(basis,fnc)

"""
function spectral_coefficients(basis,fnc)
    coefficients = zeros(size(basis,2));
    for i = 1:size(basis,2)
        # coeffs[i] = fnc'*conj(eigen[:,i]);
        # coeffs[i] = dot(fnc,eigen[:,i]);
        # coeffs[i] = dot(fnc,eigen[:,i])/dot(eigen[:,i],eigen[:,i]); #fix this... Conjugate
        # coefficients[i] = (fnc'*conj(basis[:,i]))/(basis[:,i]'*conj(basis[:,i])); #Could this hurt stability?
        coefficients[i] = (fnc'*basis[:,i])/(basis[:,i]'*basis[:,i]); #Could this hurt stability?
        # coeffs[i] = (fnc'*conj(eigen[:,i]))/norm(eigen[:,i]); #Could this hurt stability?
    end
    return coefficients
end

"""
    spectral_approximation(basis,coefficients)

"""
function spectral_approximation(basis,coefficients)
    approximation = zeros(size(basis,1));
    for i = 1:size(basis,2)
        approximation += coefficients[i]*basis[:,i];
    end
    return approximation
end

"""
    generate_opnn_solution(L1,L2,t_span,N,basis,initial_condition)

"""
function generate_opnn_basis_solution(L1,L2,t_span,N,basis,initial_condition;dt=1e-3)
    u0 = spectral_coefficients(basis,initial_condition);
    dL = abs(L2-L1);
    p = [basis,N,dL];
    prob = ODEProblem(opnn_advection_pde!,u0,t_span,p);
    sol = solve(prob,DP5(),reltol=1e-6,abstol=1e-8,saveat = dt)#,RK4(),reltol=1e-10,abstol=1e-10) # ODE45 like solver, saveat = 0.5
    u_sol = zeros(size(sol.t,1),size(basis,1));
    for i in 1:size(sol.t,1)
        u_sol[i,:] = spectral_approximation(basis,sol.u[i]);
    end
    return u_sol
end
