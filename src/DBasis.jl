"""
    function basis_OpNN(trunk,x_locations)

"""
function basis_OpNN(trunk,x_locations,n)
    basis = zeros(size(x_locations,1));
    for i in 1:size(x_locations,1);
        basis[i] = trunk(vcat(0,x_locations[i]))[n];
    end
    return basis
end

"""
    orthonormal_check(basis;tol = 1e-15)

Are the breaks really needed...?

"""
function orthonormal_check(basis;tol = 1e-15)
    for i = 1:size(basis,2)
        for j = 1:size(basis,2)
            if i == j
                if abs(norm(basis[:,i]) - 1) > tol
                    println("Not orthonormal to $tol... $i")
                    # return nothing
                    # break
                end
            elseif i != j
                # if abs(orthonormal_basis[:,i]'*conj(orthonormal_basis[:,j])) > tol# || (efuncs_orthonorm[:,i]'*conj(efuncs_orthonorm[:,j]) < eps())
                if abs(basis[:,j]'*basis[:,i]) > tol# || (efuncs_orthonorm[:,i]'*conj(efuncs_orthonorm[:,j]) < eps())
                    println("Not orthogonal to $tol... $i vs $j")
                    # return nothing
                    # break
                end
            end
        end
    end
end

"""
    build_basis(trunk,intial_condition,x_locations,opnn_output_width)

"""
function build_basis(trunk,x_locations,opnn_output_width)

    basis = zeros(size(x_locations,1),opnn_output_width);
    for i in 1:opnn_output_width
        basis[:,i] = basis_OpNN(trunk,x_locations,i)
    end

    F = qr(basis);
    orthonormal_basis_full = F.Q*Matrix(I, size(basis)...);
    orthonormal_basis = orthonormal_basis_full[:,1:length(x_locations)];

    norm_basis = [norm(basis[:,i]) for i in 1:size(basis,2)];
    norm_sort = reverse(sortperm(norm_basis));
    sorted_basis = zeros(size(basis,1),size(basis,2));
    for i in 1:size(basis,2)
        indsort = norm_sort[i];
        sorted_basis[:,i] = basis[:,indsort];
    end

    FS = qr(sorted_basis);
    orthonormal_sorted_basis_full = FS.Q*Matrix(I, size(sorted_basis)...);
    orthonormal_sorted_basis = orthonormal_sorted_basis_full[:,1:length(x_locations)];

    # Orthonormal checks
    orthonormal_check(orthonormal_basis);
    orthonormal_check(orthonormal_sorted_basis);

    return orthonormal_sorted_basis, orthonormal_basis, sorted_basis, basis
    # return orthonormal_basis, basis
end

"""
    build_basis_factors(trunk,intial_condition,x_locations,opnn_output_width)

"""
function build_basis_factors(trunk,x_locations,opnn_output_width;sorted="false",pivot_val=false)

    basis = zeros(size(x_locations,1),opnn_output_width);
    for i in 1:opnn_output_width
        basis[:,i] = basis_OpNN(trunk,x_locations,i)
    end

    if sorted == "true"
        norm_basis = [norm(basis[:,i],2) for i in 1:size(basis,2)];
        norm_sort = reverse(sortperm(norm_basis));
        sorted_basis = zeros(size(basis,1),size(basis,2));
        for i in 1:size(basis,2)
            indsort = norm_sort[i];
            sorted_basis[:,i] = basis[:,indsort];
        end
    else
        sorted_basis = deepcopy(basis);
    end

    # Test various factorizations
    # QR factorization
    Fqr = qr(sorted_basis,Val(pivot_val));
    orthonormal_basis_qr = Fqr.Q*Matrix(I, size(sorted_basis,1), size(sorted_basis,1));
    # orthonormal_basis_qr = Matrix(Fqr.Q);

    # LQ factorization
    Flq = lq(sorted_basis);
    orthonormal_basis_lq = transpose(Flq.Q*Matrix(I,opnn_output_width,opnn_output_width));
    # orthonormal_basis_lq = transpose(Matrix(Flq.Q));


    # QR^T factorization
    Fqrt = qr(transpose(sorted_basis),Val(pivot_val));
    orthonormal_basis_qrt = Fqrt.Q*Matrix(I,opnn_output_width,opnn_output_width);

    # Check for orthonormality
    orthonormal_check(orthonormal_basis_qr);
    orthonormal_check(orthonormal_basis_lq);
    orthonormal_check(orthonormal_basis_qrt);

    return sorted_basis, orthonormal_basis_qr, orthonormal_basis_lq, orthonormal_basis_qrt, Fqr, Flq, Fqrt
end

"""
    spectral_coefficients(basis,fnc)

"""
function spectral_coefficients(basis,fnc)
    coefficients = zeros(typeof(basis[1]),size(basis,2));
    for i = 1:size(basis,2)
        if norm(basis[:,i]) > eps()
            # coefficients[i] = (basis[:,i]'*fnc)/(basis[:,i]'*basis[:,i]);
            inner_loop = 0.0;
            for j = 1:size(basis,1)
                inner_loop += fnc[j]*conj(basis[j,i]);
            end
            coefficients[i] = inner_loop/(basis[:,i]'*basis[:,i]);
        else
            coefficients[i] = 0.0;
        end
    end
    # coefficients = (basis'*fnc); # Uses less memory but not consistently facter cpu times
    return coefficients
end

"""
    spectral_approximation(basis,coefficients)

"""
function spectral_approximation(basis,coefficients)
    approximation = zeros(typeof(basis[1]),size(basis,1)); # To Do: Extend usage of type of to other function which preallocate an array of zeros
    for i = 1:size(basis,2)
        approximation += coefficients[i]*basis[:,i]; # Do we need to scale this? 1/dL???
    end
    # approximation = basis*coefficients; # Uses less memory but not consistently faster cpu times
    return approximation
end

"""
    spectral_matrix(basis,Dbasis)

"""
function spectral_matrix(basis,Dbasis)
    Dmatrix = (basis'*Dbasis)/(basis'*basis);
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
                    inner_loop += basis[p,k]*Dbasis[p,l]*conj(basis[p,m]); # Faster and less memory intensive than using sum
                end
                nonlinear_triple[k,l,m] = inner_loop/(basis[:,m]'*basis[:,m]);
                # nonlinear_triple[k,l,m] = sum(Dbasis[:,k].*basis[:,l].*conj(basis[:,m]));
            end
        end
    end
    return nonlinear_triple
end

"""
    second_derivative_product(basis,DDbasis)

"""
function second_derivative_product(basis,D2basis)
    double_product = zeros(size(basis,2),size(basis,2))
    for k in 1:size(basis,2)
        for m in 1:size(basis,2)
            inner_loop = 0.0;
            for p in 1:size(basis,1)
                inner_loop += D2basis[p,k]*conj(basis[p,m]); # Faster and less memory intensive than using sum
            end
            double_product[k,m] = inner_loop/(basis[:,m]'*basis[:,m]);
        end
    end
    return double_product
end

"""
    save_basis(basis,N,n_epoch,pde_function)

FINISH!!!

"""
function save_basis(basis,N,n_epoch,pde_function)
    @save @sprintf("basis_%i_N_epochs_%i_%s.bson",N,n_epoch,pde_function) basis
end
