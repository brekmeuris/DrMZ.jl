"""
    predict(branch,trunk,initial_condition,x_locations,t_values)

Uses the trained operator neural net branch and trunk to predict solution at specified output locations.

inputs: branch, trunk, initial condition, x locations, t values

output: u(x,t)

"""
function predict(branch,trunk,initial_condition,x_locations,t_values)
    u = zeros(size(t_values,1),size(x_locations,1));
    bkt = transpose(branch(initial_condition));
    for i in 1:size(t_values,1)
        for j in 1:size(x_locations,1)
            u[i,j] = bkt*trunk(vcat(t_values[i],x_locations[j]))
        end
    end
    return u
end

"""
    loss_all(branch,trunk,initial_conditon,solution_location,target_value)

Computes the MSE for a complete dataset, Flux.mse does not seem to compute the correct MSE when applied to multiple instances. Use Flux.mse for actual training purposes and this function when you want to quantify the performance of the trained network for all of the training or testing data.

input: branch, trunk, initial condition, solution locations, target value

output: error

"""
function loss_all(branch,trunk,initial_condition,solution_location,target_value)
    yhat = zeros(1,size(target_value,2));
    for i in 1:size(target_value,2)
        yhat[i] = transpose(branch(initial_condition[:,i]))*trunk(solution_location[:,i]);
    end
    error = (1/size(target_value,2))*sum((yhat.-target_value).^2,)
    return error
end
