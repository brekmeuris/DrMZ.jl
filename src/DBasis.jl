"""
    trunk_build(trunk,M,dtsample)

Generate a dictionary of trunk functions evaluated at ``t = 0`` or at a range of times spanning (0:dtsample:1)

"""
function trunk_build(trunk,M,dtsample)
    basis_dict = [];
    if dtsample == nothing
        t_array = (0.0)
    else
        t_array = (0.0:dtsample:1.0)
    end
    for i in 1:M
        for j in 1:length(t_array)
            push!(basis_dict, (x)->trunk(vcat(t_array[j],x))[i])
        end
    end
    return basis_dict
end

"""
    basis_interpolate(coeffs,x,L1,L2)

"""
function basis_interpolate(coeffs,x,L1,L2)
    return sum([coeffs[k]*legendre_norm.(x,L1,L2,k-1) for k in 1:(size(coeffs,1))])
end

"""
    dbasis_interpolate(coeffs,x,L1,L2,m)

"""
function dbasis_interpolate(coeffs,x,L1,L2)
    return sum([coeffs[k]*dlegendre_norm.(x,L1,L2,k-1) for k in 1:(size(coeffs,1))])
end

"""
    trunk_ortho_build(utilde,L1,L2,nodes,weights,L)

Generate a dictionary of orthonormal custom basis functions which use an orthogonal polynomial expansion for evaluation away from the specified quadrature nodes.

"""
function trunk_ortho_build(utilde,L1,L2,nodes,weights,L)
    trunk_ortho = [];
    dtrunk_ortho = [];
    W = diagm(0 => weights);
    N = size(utilde,2);
    for i in 1:N
        coeffs = zeros(L+1);
        coeffs[1] = legendre_norm.(nodes,L1,L2,0)'*W*utilde[:,i];
        for j in 1:L
            coeffs[j+1] = legendre_norm.(nodes,L1,L2,j)'*W*utilde[:,i];
        end
        push!(trunk_ortho, (x) -> basis_interpolate(coeffs,x,L1,L2))
        push!(dtrunk_ortho, (x) -> dbasis_interpolate(coeffs,x,L1,L2))
    end
    return trunk_ortho, dtrunk_ortho
end

"""
    build_basis(trunk,L1,L2,M,nodes,weights,L;cutoff=1e-13,tol=1e-12)

Generate a dictionary of orthonormal custom basis functions based on a specified quadrature rule. An orthogonal polynomial expansion is utilized for the evaluation at points away from the specified quadrature nodes.     

"""
function build_basis(trunk,L1,L2,M,nodes,weights,L;cutoff=1e-13,tol=1e-12,dtsample=nothing)
    trunk_func = trunk_build(trunk,M,dtsample);

    W = diagm(0 => weights);

    A = zeros(size(nodes,1),length(trunk_func));
    for i in 1:size(nodes,1)
        for k in 1:length(trunk_func)
            A[i,k] = trunk_func[k](nodes[i]);
        end
    end

    B = W^(1/2)*A;
    F = svd(B);
    utilde = W^(-1/2)*F.U;

    r = findall(F.S .>= cutoff)[end];
    ortho_trunk_func, dortho_trunk_func = trunk_ortho_build(utilde[:,1:r],L1,L2,nodes,weights,L);

    # Check if the functions are orthonormal to a specified precision at the specified nodes
    orthonormal_check(basis_eval(ortho_trunk_func[1:r],nodes),weights;tol=tol)

    return ortho_trunk_func[1:r], dortho_trunk_func[1:r], F.S, utilde[1:r]
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
    return basis'*diagm(0 => weights)*fnc.(nodes)
end
function expansion_coefficients(basis::Array{Float64,2},fnc,nodes,weights)
    return basis'*diagm(0 => weights)*fnc
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

"""
    trunk_ortho_build_expand(utilde,nodes;p=17)

Generate a dictionary of periodic, orthonormal custom basis functions which use B-splines for evaluation away from the specified quadrature nodes.

"""
function trunk_ortho_build_expand(utilde,nodes;p=17)
    trunk_ortho = [];
    M = size(utilde,2);
    for i in 1:M
        n_full = periodic_fill_domain(nodes);
        u_full = periodic_fill_solution(utilde[:,i]')[:];
        period = n_full[end] - n_full[1];
        n_full_pad = vcat(n_full[end-6*p:end-1] .- period,n_full,n_full[1+1:1+6*p] .+ period);
        u_full_pad = vcat(u_full[end-6*p:end-1],u_full,u_full[1+1:1+6*p]);
        b = BSplineBasis(p+1,n_full_pad);
        push!(trunk_ortho, BSplines.interpolate(b,n_full_pad,u_full_pad));
    end
    return trunk_ortho
end

"""
    build_basis_expand(trunk,L1,L2,M,nodes,weights;cutoff=1e-13,tol=1e-12)

Generate a dictionary of periodic, orthonormal custom basis functions based on the trapezoid rule. B-splines are utilized for the evaluation at points away from the specified quadrature nodes.     

To Do: Add time-sampling capabilities...

"""
function build_basis_expand(trunk,L1,L2,M,nodes,weights;cutoff=1e-13,tol=1e-12)

    W = diagm(0 => weights);
    x_expand = feature_expansion_set_x(L1,L2,2,nodes');
  
    t_array = zeros(size(x_expand,2));
    x_t = vcat(t_array',x_expand);
  
    A = zeros(size(nodes,1),M);
    for i in 1:size(nodes,1)
        for k in 1:M
            A[i,k] = trunk(x_t[:,i])[k];
        end
    end
  
    B = W^(1/2)*A;
    F = svd(B);
    utilde = W^(-1/2)*F.U;
  
    ortho_trunk_func = trunk_ortho_build_expand(utilde,nodes);
    r = findall(F.S .>= cutoff)[end];

    # Check if the functions are orthonormal to a specified precision at the specified nodes
    orthonormal_check(basis_eval(ortho_trunk_func[1:r],nodes),weights;tol=tol)

    return ortho_trunk_func[1:r], F.S, utilde

end