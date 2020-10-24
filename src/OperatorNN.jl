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
    train_model(branch,trunk,n_epoch,train_data;learning_rate=0.00001)

Trains the operator neural network using the mean squared error and ADAM optimization

input: branch, trunk, number of training epochs, training data

output: trained branch, trained trunk, MSE loss for each epoch

"""
function train_model(branch,trunk,n_epoch,train_data;learning_rate=0.00001)
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

"""
    function exp_kernel_periodic(x_locations;length_scale=0.5)

Covariance kernel for radial basis function (GRF) and periodic IC f(sin^2(πx))

input: x locations, length scale

output: Σ covariance matrix

"""
function exp_kernel_periodic(x_locations,length_scale)
    out = zeros(size(x_locations,1),size(x_locations,1));
    out = [-(1/(2*length_scale^2))*norm(sin(π*x_locations[i])^2 - sin(π*x_locations[j])^2)^2 for i in 1:size(x_locations,1), j in 1:size(x_locations,1)]
    return exp.(out)
end

"""
    generate_periodic_functions(x_locations;length_scale=0.5)

Generates a specified number of random periodic functions using the exp_kernel_periodic function and a multivariate distribution

input: x locations, number of functions

output: random periodic functions

"""
function generate_periodic_functions(x_locations,number_functions;length_scale=0.5)
    sigma = exp_kernel_periodic(x_locations,length_scale); # Covariance
    mu = zeros(size(x_locations,1)); # Zero mean
    # Force the covariance matrix to be positive definite, by construction it is "approximately" Hermitian but Julia is strict
    mineig = eigmin(sigma);
    sigma -= mineig * I;
    d = MvNormal(mu,Symmetric(sigma)); # Multivariate distribution
    return rand(d,Int(number_functions))
end

"""

"""
function solution_extraction(t,x,u_sol,u_initial,num_extracted_points)
    if num_extracted_points >= size(u_sol,1)*size(u_sol,2)
        println("Invalid number of test points for the given dataset size!")
        return nothing
    else
        # Create a grid of the t,x data in the form of an array of tuples for accessing below
        t_x_grid = Array{Tuple{Float64,Float64}}(undef,(size(u_sol,1), size(u_sol,2)));
        for i in 1:size(u_sol,1)
            for j in 1:size(u_sol,2)
                t_x_grid[i,j] = (t[i],x[j]);
            end
        end
        lin_ind = reduce(vcat,size(u_sol[:]));
        shuffled_indices = randperm(lin_ind) # Randomly shuffle the linear indices corresponding to the t,x data
        indices = shuffled_indices[1:num_extracted_points] # From shuffled data, extract the number of test points for training
        initial = repeat(reshape(u_initial,(length(u_initial),1)),1,num_extracted_points);
        sol_location = zeros(2,num_extracted_points);
        for i in 1:2
            for j in 1:num_extracted_points
                sol_location[i,j] = t_x_grid[indices[j]][i]
            end
        end
        sol = reshape(u_sol[indices],(1,num_extracted_points));
        return initial, sol_location, sol
    end
end
