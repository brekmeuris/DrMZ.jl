"""
    trunk_build(trunk,M)

Generate a dictionary of trunk functions evaluated at ``t = 0``

"""
function trunk_build(trunk,M)
    return Dict(i => function (x) return trunk(vcat(0,x))[i] end for i in 1:M)
end

"""
    trunk_ortho_build(utilde,L1,L2,nodes,weights)

Generate a dictionary of orthonormal custom basis functions which use an orthogonal polynomial expansion for evaluation away from the specified quadrature nodes.

"""
function trunk_ortho_build(utilde,L1,L2,nodes,weights)
    trunk_ortho = [];
    W = diagm(0 => weights);
    N = size(utilde,2);
    for i in 1:N
        coeffs = zeros(size(utilde,1));
        coeffs[1] = utilde[:,i]'*W*legendre_norm.(nodes,L1,L2,0);
        for j in 1:(size(utilde,1)-1)
            coeffs[j+1] = utilde[:,i]'*W*legendre_norm.(nodes,L1,L2,j)
        end
        push!(trunk_ortho, (x) -> coeffs'*legendre_norm_collect.(x,L1,L2,(size(utilde,1)-1)));
    end
    return trunk_ortho
end

"""
    build_basis(trunk,L1,L2,M,nodes,weights)

Generate a dictionary of orthonormal custom basis functions based on a specified quadrature rule. An orthogonal polynomial expansion is utilized for the evaluation at points away from the specified quadrature nodes.     

"""
function build_basis(trunk,L1,L2,M,nodes,weights)
    trunk_func = trunk_build(trunk,M);

    W = diagm(0 => weights);

    A = zeros(size(nodes,1),M);
    for i in 1:size(nodes,1)
        for k in 1:M
            A[i,k] = trunk_func[k](nodes[i]);
        end
    end

    B = W^(1/2)*A;
    F = svd(B);
    utilde = W^(-1/2)*F.U;

    ortho_trunk_func = trunk_ortho_build(utilde,L1,L2,nodes,weights);

    # Check if the functions are orthonormal to a specified precision at the specified nodes
    orthonormal_check(basis_eval(ortho_trunk_func,nodes),weights)

    return ortho_trunk_func, F.S
end

"""
    basis_eval(basis,nodes)

Evaluate the basis functions for a specified spatial grid or at a specified point. Evaluation for a specified grid returns an ``N x M`` matrix with ``N`` respresenting the number of nodes and ``M`` representing the number of custom basis functions. Evalaution for a specified point returns a ``1 x M`` vector.

"""
function basis_eval(basis,nodes::Array)
    apply(f,x) = f(x);
    return permutedims(apply.(basis,permutedims(nodes)));
end
function basis_eval(basis,node)
    apply(f,x) = f(x);
    return permutedims(apply.(basis,node));
end

"""
    expansion_coefficients(basis,fnc,nodes,weights)

Compute the expansion coefficients...     

"""
function expansion_coefficients(basis::Array{Any,1},fnc::Function,nodes,weights)
    return basis_eval(basis,nodes)'*diagm(0 => weights)*fnc.(nodes)
end
function expansion_coefficients(basis::Array{Any,1},fnc,nodes,weights)
    return basis_eval(basis,nodes)'*diagm(0 => weights)*fnc
end
function expansion_coefficients(basis::Array{Float64,2},fnc::Function,nodes,weights)
    return basis'*diagm(0 => weights)*fnc.(nodes)#./diag(basis'*diagm(0=>weights)*basis)
end
function expansion_coefficients(basis::Array{Float64,2},fnc,nodes,weights)
    return basis'*diagm(0 => weights)*fnc#./diag(basis'*diagm(0=>weights)*basis)
end

"""
    expansion_approximation(basis,coefficients,nodes)

Compute the expansion approximation...

"""
function expansion_approximation(basis::Array{Any,1},coefficients,nodes)
    return basis_eval(basis,nodes)*coefficients
end
function expansion_approximation(basis::Array{Float64,2},coefficients,nodes)
    return basis*coefficients
end

"""
    basis_derivative(basis,nodes)

Compute the derivatives of each basis function using auto differentiation.

"""
function basis_derivative(basis)
  Dbasis = []
  for i in 1:length(basis);
    push!(Dbasis,(x)->ForwardDiff.derivative.(basis[i],x))
  end
  return Dbasis
end

"""
    save_basis(basis,M,pde_function)

"""
function save_basis(basis,M,pde_function)
    @save @sprintf("basis_%i_M_%s.bson",M,pde_function) basis
end

"""
    load_basis(basis,M,pde_function)

"""
function load_basis(M,pde_function)
    @load @sprintf("basis_%i_M_%s.bson",M,pde_function) basis
    return basis
end


