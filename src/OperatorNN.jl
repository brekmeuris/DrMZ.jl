"""
    predict(branch,trunk,initial_condition,x_locations,t_values)

Uses the trained operator neural net branch and trunk to predict solution at specified output locations.

input: branch, trunk, initial condition, x locations, t values

output: u(x,t)

"""
function predict(branch,trunk,initial_condition,x_locations,t_values)
    u = zeros(size(t_values,1),size(x_locations,1));
    bkt = transpose(branch(initial_condition));
    for i in 1:size(t_values,1)
        for j in 1:size(x_locations,1)
            u[i,j] = bkt*trunk(vcat(t_values[i],x_locations[j]));
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
    error = (1/size(target_value,2))*sum((yhat.-target_value).^2,);
    return error
end

"""
    function build_branch_model(input_size,neurons;activation=relu)

Builds the branch FFNN with 3 layers for a given input size and number of neurons

input: input vector size, number of neurons

output: branch (Flux dense layer)

"""
function build_branch_model(input_size,neurons;activation=relu)
    branch = Chain(Dense(input_size,neurons,activation),
        Dense(neurons,neurons,activation),
        Dense(neurons,neurons))|>f64; # Pipe it to be Float64
    return branch
end

"""
    function build_trunk_model(input_size,neurons;activation=relu)

Builds the trunk FFNN with 4 layers for a given input size and number of neurons

input: input vector size, number of neurons

output: trunk (Flux dense layer)

"""
function build_trunk_model(input_size,neurons,activation=relu)
    trunk = Chain(Dense(input_size,neurons,activation),
        Dense(neurons,neurons,activation),
        Dense(neurons,neurons,activation),
        Dense(neurons,neurons,activation))|>f64; # Pipe it to be Float64
    return trunk
end

"""
    train_model(branch,trunk,n_epochs,training_data;learning_rate=0.00001)



"""
function train_model(branch,trunk,n_epoch,training_data;learning_rate=0.00001)
    loss(x,y,z) = Flux.mse(transpose(branch(x))*trunk(y),z);
    par = Flux.params(branch,trunk);
    opt = ADAM(learning_rate);
    loss_all_train = Array{Float64}(undef,n_epoch+1,1);
    loss_all_train[1] = loss_all(branch,trunk,train_data.data[1],train_data.data[2],train_data.data[3])
    @showprogress 1 "Training the model..." for i in 1:n_epoch
        Flux.train!(loss,par,train_data,opt);
        loss_all_train[i+1] = loss_all(branch,trunk,train_data.data[1],train_data.data[2],train_data.data[3])
    end
    return branch,trunk,loss_all_train
end
