"""

"""
function basis_branch(branch,trunk,initial_condition,x_locations,n)
    basis_n = zeros(size(x_locations,1)); # make size of ic...
    for i in 1:size(locations,1);
        basis_n[i] = transpose(branch(initial_condition)[n])*trunk(vcat(0,x_locations[i]))[n];
    end
    return basis_n
end


# branch,trunk,initial_condition,x_locations,t_values
