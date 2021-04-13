"""
    predict(branch,trunk,initial_condition,x_locations,t_values)

Uses the trained operator neural net branch and trunk to predict solution at specified output locations.

input: branch, trunk, initial condition, x locations, t values

output: u(x,t)

"""
function predict(branch,trunk,initial_condition,x_locations,t_values)
    u = zeros(size(t_values,1),size(x_locations,1));
    bk = branch(initial_condition)';
    for i in 1:size(t_values,1)
        for j in 1:size(x_locations,1)
            u[i,j] = bk*trunk(vcat(t_values[i],x_locations[j]));
        end
    end
    return u
end

"""
    predict_min_max(branch,trunk,initial_condition,x_locations,t_values,scale_object)

Uses the trained operator neural net branch and trunk to predict solution at specified output locations when using min-max normalization

input: branch, trunk, initial condition, x locations, t values

output: u(x,t)

"""
function predict_min_max(branch,trunk,initial_condition,x_locations,t_values,scale_object)
    u = zeros(size(t_values,1),size(x_locations,1));
    bk = branch(initial_condition)';
    for i in 1:size(t_values,1)
        for j in 1:size(x_locations,1)
            u[i,j] = bk*trunk(min_max_transform(vcat(t_values[i],x_locations[j]),scale_object));
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
        yhat[i] = branch(initial_condition[:,i])'*trunk(solution_location[:,i]);
    end
    return (1/size(target_value,2))*sum((yhat.-target_value).^2);
end

"""
    function build_dense_model(number_layers,neurons,activations)

Builds a FFNN of Flux dense layers

input: number of layers, array consisting of tuples (layer input size, layer output size), and array of activation functions

output: Flux model consisting of chained dense layers piped to output Float64

"""
function build_dense_model(number_layers,neurons,activations)
    layers = [Dense(neurons[i][1],neurons[i][2],activations[i]) for i in 1:number_layers];
    return Chain(layers...)|>f64
end

"""
    train_model(branch,trunk,n_epoch,train_data;learning_rate=0.00001)

Trains the operator neural network using the mean squared error and ADAM optimization

input: branch, trunk, number of training epochs, training data # UPDATE

output: trained branch, trained trunk, MSE loss for each epoch # UPDATE

To Do : ADD A PARAMETER TO APPLY NORMALIZATION OR SCALING...

"""
function train_model(branch,trunk,n_epoch,train_data,test_data,pde_function;learning_rate=1e-5)
    loss(x,y,z) = Flux.mse(branch(x)'*trunk(y),z)
    par = Flux.params(branch,trunk);
    opt = ADAM(learning_rate);
    loss_all_train = Array{Float64}(undef,n_epoch+1,1);
    loss_all_test = Array{Float64}(undef,n_epoch+1,1);
    loss_all_train[1] = loss_all(branch,trunk,train_data.data[1],train_data.data[2],train_data.data[3]); # To Do update to trend test error as well
    loss_all_test[1] = loss_all(branch,trunk,test_data.data[1],test_data.data[2],test_data.data[3]); # To Do update to trend test error as well
    @showprogress 1 "Training the model..." for i in 1:n_epoch
        Flux.train!(loss,par,train_data,opt);
        loss_all_train[i+1] = loss_all(branch,trunk,train_data.data[1],train_data.data[2],train_data.data[3]);
        loss_all_test[i+1] = loss_all(branch,trunk,test_data.data[1],test_data.data[2],test_data.data[3]);
        if i%2500 == 0
            save_model(branch,trunk,i,loss_all_train,loss_all_test,pde_function)
            train_MSE = loss_all(branch,trunk,train_data.data[1],train_data.data[2],train_data.data[3]);
            test_MSE = loss_all(branch,trunk,test_data.data[1],test_data.data[2],test_data.data[3]);
            println("Train MSE $train_MSE")
            println("Test MSE $test_MSE")
        end
    end
    return branch,trunk,loss_all_train,loss_all_test
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
function generate_periodic_functions(x_locations,number_functions,length_scale)
    sigma = exp_kernel_periodic(x_locations,length_scale); # Covariance
    mu = zeros(size(x_locations,1)); # Zero mean
    # Force the covariance matrix to be positive definite, by construction it is "approximately" Hermitian but Julia is strict
    mineig = eigmin(sigma);
    sigma -= mineig * I;
    d = MvNormal(mu,Symmetric(sigma)); # Multivariate distribution
    return rand(d,Int(number_functions))
end

"""
    generate_sinusoidal_functions_2_parameter(x_locations,number_functions)

"""
function generate_sinusoidal_functions_2_parameter(x_locations,number_functions)
    values = rand(-1.0:(2.0/(number_functions*10)):1.0,(number_functions,2)); # alpha, beta parameters
    initial_base = vcat(values,[0.0 0.0]);
    if size(unique(initial_base,dims=2),1) != size(initial_base,1)
        println("Duplicate multiplier sets or generating function included in data!")
        return nothing
    end
    initial_conditions = zeros(size(x_locations,1),number_functions);
    for i in 1:number_functions
        initial_conditions[:,i] = values[i,1]*sin.(x_locations).+values[i,2]; # αsin(2πx)+β
    end
    return initial_conditions,
end

"""
    solution_extraction(x_locations,t_values,solution,initial_condition,number_solution_points)

FINISH!!!

"""
function solution_extraction(x_locations,t_values,solution,initial_condition,number_solution_points)
    if number_solution_points >= size(solution,1)*size(solution,2)
        println("Invalid number of test points for the given dataset size!")
        return nothing
    else
        # Create a grid of the t,x data in the form of an array of tuples for accessing below
        t_x_grid = Array{Tuple{Float64,Float64}}(undef,(size(solution,1), size(solution,2)));
        for i in 1:size(solution,1)
            for j in 1:size(solution,2)
                t_x_grid[i,j] = (t_values[i],x_locations[j]);
            end
        end
        lin_ind = reduce(vcat,size(solution[:]));
        shuffled_indices = randperm(lin_ind) # Randomly shuffle the linear indices corresponding to the t,x data
        indices = shuffled_indices[1:number_solution_points] # From shuffled data, extract the number of test points for training
        initial = repeat(reshape(initial_condition,(length(initial_condition),1)),1,number_solution_points);
        sol_location = zeros(2,number_solution_points);
        for i in 1:2
            for j in 1:number_solution_points
                sol_location[i,j] = t_x_grid[indices[j]][i]
            end
        end
        sol = reshape(solution[indices],(1,number_solution_points));
        return initial, sol_location, sol
    end
end

"""
    generate_periodic_train_test(L1,L2,t_span,number_sensors,number_test_functions,number_train_functions,number_solution _points;length_scale=0.5,batch=number_solution_points,dt=1e-3)

FINISH!!!

"""
function generate_periodic_train_test(L1,L2,t_span,number_sensors,number_train_functions,number_test_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt_size=1e-3,domain="periodic")

    if domain == "periodic"
        x_full = range(L1,stop = L2,length = number_sensors+1); # Full domain for periodic number of sensors
        random_ics = generate_periodic_functions(x_full,Int(number_train_functions + number_test_functions),length_scale)
        dL = abs(L2-L1);
        # Set up x domain and wave vector for spectral solution
        j = reduce(vcat,[0:1:number_sensors-1]);
        x = (dL.*j)./number_sensors;
        k = reduce(vcat,(2*π/dL)*[0:number_sensors/2-1 -number_sensors/2:-1]);
    elseif domain == "full"
        x_full = range(L1,stop = L2,length = number_sensors); # Full domain for periodic number of sensors
        random_ics = generate_periodic_functions(x_full,Int(number_train_functions + number_test_functions),length_scale)
        dL = abs(L2-L1);
        # Set up x domain and wave vector for spectral solution
        j = reduce(vcat,[0:1:number_sensors-2]);
        x = (dL.*j)./Int(number_sensors-1);
        k = reduce(vcat,(2*π/dL)*[0:Int(number_sensors-1)/2-1 -Int(number_sensors-1)/2:-1]);
    end

    # Generate the dataset using spectral method
    t_length = t_span[2]/dt_size + 1;
    t = range(t_span[1],stop = t_span[2], length = Int(t_length));
    train_ic = zeros(number_sensors,number_solution_points,number_train_functions);
    train_loc = zeros(2,number_solution_points,number_train_functions);
    train_target = zeros(1,number_solution_points,number_train_functions);
    test_ic = zeros(number_sensors,number_solution_points,number_test_functions);
    test_loc = zeros(2,number_solution_points,number_test_functions);
    test_target = zeros(1,number_solution_points,number_test_functions);

    if domain == "periodic"
        @showprogress 1 "Building training dataset..." for i in 1:number_train_functions
            interp_train_sample = random_ics[1:end-1,i];
            u_train = generate_fourier_solution(L1,L2,t_span,number_sensors,interp_train_sample,pde_function_handle,dt=dt_size);
            train_ic[:,:,i], train_loc[:,:,i], train_target[:,:,i] = solution_extraction(x,t,u_train,interp_train_sample,number_solution_points);
        end
        @showprogress 1 "Building testing dataset..." for i in 1:number_test_functions
            interp_test_sample = random_ics[1:end-1,Int(number_train_functions+i)];
            u_test = generate_fourier_solution(L1,L2,t_span,number_sensors,interp_test_sample,pde_function_handle,dt=dt_size);
            test_ic[:,:,i], test_loc[:,:,i], test_target[:,:,i] = solution_extraction(x,t,u_test,interp_test_sample,number_solution_points);
        end
    elseif domain == "full"
        @showprogress 1 "Building training dataset..." for i in 1:number_train_functions
            interp_train_sample_fourier = random_ics[1:end-1,i];
            u_train_fourier = generate_fourier_solution(L1,L2,t_span,Int(number_sensors-1),interp_train_sample_fourier,pde_function_handle,dt=dt_size);
            interp_train_sample = random_ics[:,i];
            u_train = periodic_fill_solution(u_train_fourier);
            train_ic[:,:,i], train_loc[:,:,i], train_target[:,:,i] = solution_extraction(x_full,t,u_train,interp_train_sample,number_solution_points);
        end
        @showprogress 1 "Building testing dataset..." for i in 1:number_test_functions
            interp_test_sample_fourier = random_ics[1:end-1,Int(number_train_functions+i)];
            u_test_fourier = generate_fourier_solution(L1,L2,t_span,Int(number_sensors-1),interp_test_sample_fourier,pde_function_handle,dt=dt_size);
            interp_test_sample = random_ics[:,Int(number_train_functions+i)];
            u_test = periodic_fill_solution(u_test_fourier);
            test_ic[:,:,i], test_loc[:,:,i], test_target[:,:,i] = solution_extraction(x_full,t,u_test,interp_test_sample,number_solution_points);
        end
    end

    # Combine data sets from each function
    opnn_train_ic = reshape(hcat(train_ic...),(number_sensors,Int(number_solution_points*number_train_functions)));
    opnn_train_loc = reshape(hcat(train_loc...),(2,Int(number_solution_points*number_train_functions)));
    opnn_train_target = reshape(hcat(train_target...),(1,Int(number_solution_points*number_train_functions)));
    opnn_test_ic = reshape(hcat(test_ic...),(number_sensors,Int(number_solution_points*number_test_functions)));
    opnn_test_loc = reshape(hcat(test_loc...),(2,Int(number_solution_points*number_test_functions)));
    opnn_test_target = reshape(hcat(test_target...),(1,Int(number_solution_points*number_test_functions)));

    train_data = DataLoader(opnn_train_ic, opnn_train_loc, opnn_train_target, batchsize=batch);
    test_data = DataLoader(opnn_test_ic, opnn_test_loc, opnn_test_target, batchsize=batch);
    return train_data, test_data
end

"""
    save_model(branch,trunk,n_epoch,loss_all_train,loss_all_test,pde_function)

FINISH!!!

"""
function save_model(branch,trunk,n_epoch,loss_all_train,loss_all_test,pde_function)
    @save @sprintf("branch_epochs_%i_%s.bson",n_epoch,pde_function) branch
    @save @sprintf("trunk_epochs_%i_%s.bson",n_epoch,pde_function) trunk
    @save @sprintf("train_loss_epochs_%i_%s.bson",n_epoch,pde_function) loss_all_train
    @save @sprintf("test_loss_epochs_%i_%s.bson",n_epoch,pde_function) loss_all_test
end

"""
    load_model(branch,trunk,n_epoch)

FINISH!!!

"""
function load_model(n_epoch,pde_function)
    @load @sprintf("branch_epochs_%i_%s.bson",n_epoch,pde_function) branch
    @load @sprintf("trunk_epochs_%i_%s.bson",n_epoch,pde_function) trunk
    return branch, trunk
end

"""
    save_data(train_data,test_data,u_test,u_train,n_epoch,number_solution_points,loss_all_train)

FINISH!!!

"""
function save_data(train_data,test_data,n_epoch,number_train_functions,number_test_functions,number_solution_points,pde_function)
    train_ic = train_data.data[1];
    train_loc = train_data.data[2];
    train_sol = train_data.data[3];
    test_ic = test_data.data[1];
    test_loc = test_data.data[2];
    test_sol = test_data.data[3];
    @save @sprintf("train_ic_data_%i_%s.bson",number_train_functions,pde_function) train_ic
    @save @sprintf("test_ic_data_%i_%s.bson",number_test_functions,pde_function) test_ic
    @save @sprintf("train_loc_data_%i_%s.bson",number_train_functions,pde_function) train_loc
    @save @sprintf("test_loc_data_%i_%s.bson",number_test_functions,pde_function) test_loc
    @save @sprintf("train_target_data_%i_%s.bson",number_train_functions,pde_function) train_sol
    @save @sprintf("test_target_data_%i_%s.bson",number_test_functions,pde_function) test_sol
end

"""
    load_data(n_epoch,number_train_functions,number_test_functions)

FINISH!!!

"""
function load_data(n_epoch,number_train_functions,number_test_functions,pde_function)
    @load @sprintf("branch_epochs_%i_%s.bson",n_epoch,pde_function) branch
    @load @sprintf("trunk_epochs_%i_%s.bson",n_epoch,pde_function) trunk
    @load @sprintf("train_ic_data_%i_%s.bson",number_train_functions,pde_function) train_ic
    @load @sprintf("train_loc_data_%i_%s.bson",number_train_functions,pde_function) train_loc
    @load @sprintf("train_target_data_%i_%s.bson",number_train_functions,pde_function) train_sol
    @load @sprintf("test_ic_data_%i_%s.bson",number_test_functions,pde_function) test_ic
    @load @sprintf("test_loc_data_%i_%s.bson",number_test_functions,pde_function) test_loc
    @load @sprintf("test_target_data_%i_%s.bson",number_test_functions,pde_function) test_sol
    return branch, trunk, train_ic, train_loc, train_sol, test_ic, test_loc, test_sol#
end

"""
    load_data_train_test(number_train_functions,number_test_functions,pde_function)

FINISH!!!

"""
function load_data_train_test(number_train_functions,number_test_functions,pde_function)
    @load @sprintf("train_ic_data_%i_%s.bson",number_train_functions,pde_function) train_ic
    @load @sprintf("train_loc_data_%i_%s.bson",number_train_functions,pde_function) train_loc
    @load @sprintf("train_target_data_%i_%s.bson",number_train_functions,pde_function) train_sol
    @load @sprintf("test_ic_data_%i_%s.bson",number_test_functions,pde_function) test_ic
    @load @sprintf("test_loc_data_%i_%s.bson",number_test_functions,pde_function) test_loc
    @load @sprintf("test_target_data_%i_%s.bson",number_test_functions,pde_function) test_sol
    return train_ic, train_loc, train_sol, test_ic, test_loc, test_sol
end

"""
    load_data_initial_conditions(number_train_functions,number_test_functions)

FINISH!!!

"""
function load_data_initial_conditions(number_train_functions,number_test_functions,pde_function)
    @load @sprintf("train_ic_data_%i_%s.bson",number_train_functions,pde_function) train_ic
    @load @sprintf("test_ic_data_%i_%s.bson",number_test_functions,pde_function) test_ic
    return train_ic, test_ic
end

"""
    min_max_scaler(x;dim = 2)

"""
function min_max_scaler(x;dim = 2)
    scaler = 1.0 ./ (maximum(x,dims=dim) .- minimum(x,dims=dim));
    min = -minimum(x,dims=dim).*scaler;
    return scaler, min
end

"""
    min_max_transform(x,scale_object;min = 0,max = 1)

"""
function min_max_transform(x,scale_object;min = 0,max = 1)
    x_scaled = deepcopy(x);
    x_scaled .*= scale_object[1][:];
    x_scaled .+= scale_object[2][:];
    x_scaled .*= (max-min);
    x_scaled .+= min;
    return x_scaled
end

"""
    standard_scaler(x;dim=2)

"""
function standard_scaler(x;dim=2)
    x_mean = mean(x,dims=dim);
    x_std = std(x,dims=dim,corrected=false)
    return x_mean, x_std
end

"""
    standard_transform(x,scale_object)

"""
function standard_transform(x,scale_object)
    x_scaled = deepcopy(x);
    x_scaled .-= scale_object[1][:];
    x_scaled ./= scale_object[2][:];
end
