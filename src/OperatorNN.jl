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





"""
function loss_all(x,y,z,b,t)
    ŷ = zeros(1,size(z,2));
    for i in 1:size(z,2)
        ŷ[i] = transpose(b(x[:,i]))*t(y[:,i]);
    end
    error = (1/size(z,2))*sum((ŷ.-z).^2,)
    # error = sum((ŷ.-z).^2)
    # error = mean((ŷ.-z).^2);
    return error
end
