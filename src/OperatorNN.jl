"""
    predict(branch,trunk,initial_condition,x_locations,t_values)

Uses the trained operator neural net branch and trunk to predict solution at specified output locations.

inputs: branch, trunk, initial condition, x locations, t values

output: u(x,t)

"""
function predict(branch,trunk,initial_condition,x_locations,t_values)
    u = zeros(size(trunk,1),size(x_locations,1));
    bkt = transpose(bk(initial_condition));
    for i in 1:size(trunk,1)
        for j in 1:size(x_locations,1)
            u[i,j] = bkt*tk(vcat(t_values[i],x_locations[j]))
        end
    end
    return u
end

"""
    loss_all(branch,trunk,initial_conditon,solution_location,target_value)






"""
function loss_all(branch,trunk,initial_conditon,solution_location,target_value)
    yhat = zeros(1,size(target_value,2));
    for i in 1:size(target_value,2)
        yhat[i] = transpose(branch(initial_condition[:,i]))*trunk(solution_location[:,i]);
    end
    error = (1/size(target_value,2))*sum((yhat.-target_value).^2,)
    return error
end
