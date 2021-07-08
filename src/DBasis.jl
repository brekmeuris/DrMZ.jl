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

    ic_norm = initial_condition/norm(initial_condition,2);

    norm_basis = [norm(basis[:,i]) for i in 1:size(basis,2)];
    norm_sort = reverse(sortperm(norm_basis));
    sorted_basis = zeros(size(basis,1),size(basis,2));
    for i in 1:size(basis,2)
        indsort = norm_sort[i];
        sorted_basis[:,i] = basis[:,indsort];
    end

    svd_full = Matrix(qr(sorted_basis).Q);
    ic_svd_full = hcat(ic_norm,svd_full[:,1:end-1]);

    # svd_full = svd(basis,full=true);
    # ic_svd_full = hcat(ic_norm,svd_full.U[:,1:end-1]);


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
    trunk_build(trunk,L1,L2,N,M;number_points=1000)

"""
function trunk_build(trunk,L1,L2,N,M;number_points=1000)
    norm_trunk = [sqrt(gauss_quad(L1,L2,(x)->(trunk(vcat(0,x))[i].*conj(trunk(vcat(0,x))[i])),number_points)) for i in 1:M];
    norm_sort = reverse(sortperm(norm_trunk));
    return Dict(i => function (x) return trunk(vcat(0,x))[norm_sort[i]] end for i in 1:N)
end

"""
    trunk_ortho_build(A,trunk_sort)

"""
function trunk_ortho_build(A,trunk_sort)
    trunk_ortho = [];
    N = length(trunk_sort);
    for i in 1:N
        # push!(legendre_ortho, function (x) return reduce(+,[A[j,i]*legendre[j](x) for j in 1:i]) end);
        push!(trunk_ortho, (x) -> sum([A[j,i]*trunk_sort[j](x) for j in 1:i]));
        # push!(legendre_ortho, @inline function (x) return sum([A[j,i]*legendre[j](x) for j in 1:i]) end);

    end
    return trunk_ortho
end

"""
    build_basis_functions(trunk,L1,L2,N,M,ic_func;number_points=1000,precision=2^7)

"""
function build_basis_functions(trunk,L1,L2,N,M,ic_func;number_points=1000,precision=2^7)
    setprecision(BigFloat,precision)
    trunk_sort = trunk_build(trunk,L1,L2,N,M);
    nodes, weights = gausslegendre(number_points);
    snodes = shifted_nodes(L1,L2,nodes);
    sweights = (L2-L1)/2*weights;

    W = diagm(0 => BigFloat.(sweights));

    A = zeros(BigFloat,size(snodes,1),N);
    for i in 1:size(snodes,1)
        for k in 1:N
            A[i,k] = BigFloat(trunk_sort[k](BigFloat(snodes[i])));
        end
    end

    B = transpose(A)*W*A;
    C = cholesky(Hermitian(B));
    ortho_trunk_inv = Float64.(inv(C.U));
    ortho_trunk = Float64.(\(C.U,I));
    condC = Float64(cond(C.U));

    ortho_trunk_func = trunk_ortho_build(ortho_trunk,trunk_sort)

    custom = [];
    push!(custom,ic_func);
    for i in 1:N-1
        push!(custom,ortho_trunk_func[i]);
    end

    Ac = zeros(BigFloat,size(snodes,1),N);
    for i in 1:size(snodes,1)
        for k in 1:N
            Ac[i,k] = BigFloat(custom[k](BigFloat(snodes[i])));
        end
    end

    Bc = transpose(Ac)*W*Ac;
    Cc = cholesky(Hermitian(Bc));
    ortho_custom_inv = Float64.(inv(Cc.U));
    ortho_custom = Float64.(\(Cc.U,I));
    condCc = Float64(cond(Cc.U));

    ortho_custom_func = trunk_ortho_build(ortho_custom,custom)

    ip_check = zeros(N,N);
    for i in 1:N
        for j in 1:N
            ip_check[i,j] = gauss_quad(L1,L2,(x)->(ortho_custom_func[i](x).*conj(ortho_custom_func[j](x))),number_points);
        end
    end

    return ortho_custom_func, ip_check
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
    spectral_coefficients_functions(L1,L2,basis,fnc;number_points=1000)

"""
function spectral_coefficients_functions(L1,L2,basis,fnc;number_points=1000)
    coefficients = zeros(length(basis)); # Add back typeof...
    for i = 1:length(basis)
        # coefficients[i] = gauss_quad(L1,L2,(x)->(fnc(x).*conj(basis[i](x))),number_points)/gauss_quad(L1,L2,(x)->(basis[i](x).*conj(basis[i](x))),number_points);
        coefficients[i] = gauss_quad(L1,L2,(x)->(fnc(x)*conj(basis[i](x))),number_points)/gauss_quad(L1,L2,(x)->(basis[i](x)*conj(basis[i](x))),number_points);
    end
    return coefficients
end

"""
    spectral_coefficients_fourier_functions(L1,L2,k,fnc;number_points=1000)

"""
function spectral_coefficients_fourier_functions(L1,L2,k,fnc;number_points=1000)
    coefficients = zeros(Complex{Float64},length(k)); # Add back typeof...
    for i = 1:length(k)
        coefficients[i] = gauss_quad(L1,L2,(x)->(fnc(x)*exp(-im*k[i]*x)),number_points)/gauss_quad(L1,L2,(x)->(exp(im*k[i]*x)*exp(-im*k[i]*x)),number_points);
    end
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
    spectral_approximation_functions(x_locations,basis,coefficients;number_points=1000)

"""
function spectral_approximation_functions(x_locations,basis,coefficients)
    approximation = zeros(size(x_locations,1)); # Add back typeof...
    for i = 1:length(basis)
        approximation += coefficients[i]*basis[i].(x_locations);
    end
    return approximation
end

"""
    spectral_approximation_fourier_functions(x_locations,k,coefficients;number_points=1000)

"""
function spectral_approximation_fourier_functions(x_locations,k,coefficients)
    approximation = zeros(size(x_locations,1)); # Add back typeof...
    for i = 1:length(k)
        approximation += coefficients[i]*exp.(im*k[i]*(x_locations));
    end
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


"""
    first_derivative_product(basis,Dbasis)

"""
function first_derivative_product(basis,Dbasis)
    single_product = zeros(size(basis,2),size(basis,2))
    for k in 1:size(basis,2)
        for m in 1:size(basis,2)
            inner_loop = 0.0;
            for p in 1:size(basis,1)
                inner_loop += Dbasis[p,k]*conj(basis[p,m]); # Faster and less memory intensive than using sum
            end
            single_product[m,k] = inner_loop/(basis[:,m]'*basis[:,m]);
        end
    end
    return single_product
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
            double_product[m,k] = inner_loop/(basis[:,m]'*basis[:,m]);
        end
    end
    return double_product
end
