"""
    function basis_OpNN(trunk,x_locations)

"""
function basis_OpNN(trunk,x_locations,n)
    basis = zeros(size(x_locations,1));
    for i in 1:size(x_locations,1);
        basis[i] = trunk(vcat(0.0,x_locations[i]))[n];
    end
    return basis
end

"""
    orthonormal_check(basis;tol = 1e-15)

"""
function orthonormal_check(basis;tol = 1e-15)
    for i = 1:size(basis,2)
        for j = 1:size(basis,2)
            if i == j
                if abs(norm(basis[:,i]) - 1) > tol
                    println("Not orthonormal to $tol... $i")
                end
            elseif i != j
                # if abs(orthonormal_basis[:,i]'*conj(orthonormal_basis[:,j])) > tol# || (efuncs_orthonorm[:,i]'*conj(efuncs_orthonorm[:,j]) < eps())
                if abs(basis[:,j]'*basis[:,i]) > tol# || (efuncs_orthonorm[:,i]'*conj(efuncs_orthonorm[:,j]) < eps())
                    println("Not orthogonal to $tol... $i vs $j")
                end
            end
        end
    end
end

"""
    build_basis(trunk,intial_condition,x_locations,opnn_output_width,initial_condition)

"""
function build_basis(trunk,x_locations,opnn_output_width,initial_condition)
    basis = zeros(size(x_locations,1),opnn_output_width);
    for i in 1:opnn_output_width
        basis[:,i] = basis_OpNN(trunk,x_locations,i)
    end

    # norm_basis = [norm(basis[:,i]) for i in 1:size(basis,2)];
    # norm_sort = reverse(sortperm(norm_basis));
    # sorted_basis = zeros(size(basis,1),size(basis,2));
    # for i in 1:size(basis,2)
    #     indsort = norm_sort[i];
    #     sorted_basis[:,i] = basis[:,indsort];
    # end

    svd_full = svd(basis,full=true);
    # svd_full = Matrix(qr(sorted_basis).Q);
    ic_norm = initial_condition/norm(initial_condition,2);
    ic_svd_full = hcat(ic_norm,svd_full.U[:,1:end-1]);
    # ic_svd_full = hcat(ic_norm,svd_full[:,1:end-1]);

    orthonormal_basis = Matrix(qr(ic_svd_full).Q);
    orthonormal_check(orthonormal_basis);
    return orthonormal_basis
end

"""
    build_basis_redefinition(basis_full,x_redefined,x_full)

"""
function build_basis_redefinition(basis_full,x_redefined,x_full)
    basis = transpose(solution_spatial_sampling(x_redefined,x_full,transpose(basis_full)));
    orthonormal_basis = Matrix(qr(basis).Q);
    orthonormal_check(orthonormal_basis);
    return orthonormal_basis
end

"""
    spectral_coefficients(basis,fnc)

"""
function spectral_coefficients(basis,fnc)
    N = size(basis,2);
    coefficients = zeros(typeof(basis[1]),size(basis,2));
    for i = 1:size(basis,2)
        # coefficients[i] = (basis[:,i]'*fnc)/(basis[:,i]'*basis[:,i]);
        inner_loop = 0.0;
        for j = 1:size(basis,1)
            inner_loop += fnc[j]*conj(basis[j,i]);
        end
        coefficients[i] = inner_loop/(basis[:,i]'*basis[:,i]);
    end
    # coefficients = (basis'*fnc); # Uses less memory but not consistently facter cpu times
    return coefficients
end

"""
    spectral_approximation(basis,coefficients)

"""
function spectral_approximation(basis,coefficients)
    N = size(basis,2);
    approximation = zeros(typeof(basis[1]),size(basis,1));
    for i = 1:size(basis,2)
        approximation += coefficients[i]*basis[:,i];
    end
    # approximation = basis*coefficients; # Uses less memory but not consistently faster cpu times
    return approximation
end

"""
    spectral_matrix(basis,Dbasis)

"""
function spectral_matrix(basis,Dbasis)
    Dmatrix = (basis'*Dbasis)/(basis'*basis); # To Do: Check speed vs second_derivative_product
    return Dmatrix
end

"""
    quadratic_nonlinear_triple_product(basis,Dbasis)

"""
function quadratic_nonlinear_triple_product(basis,Dbasis)
    nonlinear_triple = zeros(size(basis,2),size(basis,2),size(basis,2);)
    for k in 1:size(basis,2)
        for l in 1:size(basis,2)
            for m in 1:size(basis,2)
                inner_loop = 0.0;
                for p in 1:size(basis,1)
                    inner_loop += basis[p,k]*Dbasis[p,l]*conj(basis[p,m]);
                end
                nonlinear_triple[k,l,m] = inner_loop/(basis[:,m]'*basis[:,m]);
            end
        end
    end
    return nonlinear_triple
end

"""
    save_basis(basis,N,n_epoch,pde_function)

"""
function save_basis(basis,N,n_epoch,pde_function)
    @save @sprintf("basis_%i_N_epochs_%i_%s.bson",N,n_epoch,pde_function) basis
end
