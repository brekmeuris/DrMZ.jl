"""
    predict(branch,trunk,initial_condition,x_locations,t_values)

Uses the trained operator neural net branch and trunk to predict solution at specified output locations.

input: branch, trunk, initial condition, x locations, t values

output: u(x,t)

"""
function predict(branch,trunk,initial_condition,x_locations,t_values)
    u = zeros(size(t_values,1),size(x_locations,1));
    # bkt = transpose(branch(initial_condition));
    bkt = branch(initial_condition)';
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
        # yhat[i] = transpose(branch(initial_condition[:,i]))*trunk(solution_location[:,i]);
        yhat[i] = branch(initial_condition[:,i])'*trunk(solution_location[:,i]);
    end
    return (1/size(target_value,2))*sum((yhat.-target_value).^2,);
end

"""
    function build_branch_model(input_size,neurons;activation=relu)

Builds the branch FFNN with 3 layers for a given input size and number of neurons

input: input vector size, number of neurons

output: branch (Flux dense layer)

"""
function build_branch_model(input_size,neurons;activation=relu)
    return Chain(Dense(input_size,neurons,activation),
        Dense(neurons,neurons,activation),
        Dense(neurons,neurons))|>f64; # Pipe it to be Float64
end

"""
    function build_trunk_model(input_size,neurons;activation=relu)

Builds the trunk FFNN with 4 layers for a given input size and number of neurons

input: input vector size, number of neurons

output: trunk (Flux dense layer)

"""
function build_trunk_model(input_size,neurons,activation=relu)
    return Chain(Dense(input_size,neurons,activation),
        Dense(neurons,neurons,activation),
        Dense(neurons,neurons,activation),
        Dense(neurons,neurons,activation))|>f64; # Pipe it to be Float64
end

"""
    train_model(branch,trunk,n_epoch,train_data;learning_rate=0.00001)

Trains the operator neural network using the mean squared error and ADAM optimization

input: branch, trunk, number of training epochs, training data

output: trained branch, trained trunk, MSE loss for each epoch

"""
function train_model(branch,trunk,n_epoch,train_data;learning_rate=0.00001)
    # loss(x,y,z) = Flux.mse(transpose(branch(x))*trunk(y),z);
    loss(x,y,z) = Flux.mse(branch(x)'*trunk(y),z)
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
    solution_extraction(t,x,u_sol,u_initial,num_extracted_points) CLEAN UP VARIABLES...

FINISH

predict(branch,trunk,initial_condition,x_locations,t_values)

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

FINISH!!! THIS SHOULD ALSO PASS IN RHS AND ANY PARAMETERS NEEDED FOR ODE SOLVE
CREATE ALTERNATIVE FUNCTION THAT IS GENERATE_STANDARD_TRAIN_TEST FOR FUNCTIONS OVER FULL DOMAIN

EDIT THIS SO THAT WHAT IS OUTPUT IS OVER FULL DOMAIN.

"""
function generate_periodic_train_test(L1,L2,t_span,number_sensors,number_train_functions,number_test_functions,number_solution_points;length_scale=0.5,batch=number_solution_points,dt=1e-3)

    x_full = range(L1,stop = L2,length = number_sensors+1); # Full domain
    random_ics = generate_periodic_functions(x_full,Int(number_train_functions + number_test_functions),length_scale)
    dL = abs(L2-L1);

    # Set up x domain and wave vector for spectral solution
    j = reduce(vcat,[0:1:number_sensors-1]);
    x = (dL.*j)./number_sensors;
    k = reduce(vcat,(2*π/dL)*[0:number_sensors/2-1 -number_sensors/2:-1]);

    # Generate the dataset using spectral method
    interp_train_sample = zeros(size(x,1),number_train_functions);
    interp_test_sample = zeros(size(x,1),number_test_functions);
    t_length = t_span[2]/dt + 1;
    t = range(t_span[1],stop = t_span[2], length = Int(t_length));
    u_train = zeros(Int(t_length),number_sensors,number_train_functions);
    u_test = zeros(Int(t_length),number_sensors,number_test_functions);
    train_ic = zeros(number_sensors,number_solution_points,number_train_functions);
    train_loc = zeros(2,number_solution_points,number_train_functions);
    train_target = zeros(1,number_solution_points,number_train_functions);
    test_ic = zeros(number_sensors,number_solution_points,number_test_functions);
    test_loc = zeros(2,number_solution_points,number_test_functions);
    test_target = zeros(1,number_solution_points,number_test_functions);

    p = [number_sensors,dL];

    @showprogress 1 "Building training dataset..." for i in 1:number_train_functions

        interp_train_sample[:,i] = random_ics[1:end-1,i];

        # Transform random initial condition to Fourier domain
        uhat0 = fft(interp_train_sample[:,i]);

        # Solve the system of ODEs in Fourier domain
        prob = ODEProblem(advection_pde!,uhat0,t_span,p);
        sol = solve(prob,DP5(),reltol=1e-6,abstol=1e-8,saveat = dt)

        for j in 1:size(sol.t,1) # Reshape output and plot
            u_train[j,:,i] = real.(ifft(sol.u[j])); # u[t,x,IC]
        end

        train_ic[:,:,i], train_loc[:,:,i], train_target[:,:,i] = solution_extraction(x,t,u_train[:,:,i],interp_train_sample[:,i],number_solution_points);
    end

    @showprogress 1 "Building testing dataset..." for i in 1:number_test_functions

        interp_test_sample[:,i] = random_ics[1:end-1,Int(number_train_functions+i)];

        # Transform random initial condition to Fourier domain
        uhat0 = fft(interp_test_sample[:,i]); # Transform random initial condition to Fourier domain

        # Solve the system of ODEs in Fourier domain
        prob = ODEProblem(advection_pde!,uhat0,t_span,p);
        sol = solve(prob,DP5(),reltol=1e-6,abstol=1e-8,saveat = dt)

        for j in 1:size(sol.t,1) # Reshape output and plot
            u_test[j,:,i] = real.(ifft(sol.u[j])); # u[t,x,IC]
        end

        test_ic[:,:,i], test_loc[:,:,i], test_target[:,:,i] = solution_extraction(x,t,u_test[:,:,i],interp_test_sample[:,i],number_solution_points);
    end

    # Combine data sets from each function
    opnn_train_ic = reshape(hcat(train_ic...),(number_sensors,Int(number_solution_points*number_train_functions)));
    opnn_train_loc = reshape(hcat(train_loc...),(2,Int(number_solution_points*number_train_functions)));
    opnn_train_target = reshape(hcat(train_target...),(1,Int(number_solution_points*number_train_functions)));
    opnn_test_ic = reshape(hcat(test_ic...),(number_sensors,Int(number_solution_points*number_test_functions)));
    opnn_test_loc = reshape(hcat(test_loc...),(2,Int(number_solution_points*number_test_functions)));
    opnn_test_target = reshape(hcat(test_target...),(1,Int(number_solution_points*number_test_functions)));

    train_data = DataLoader(opnn_train_ic, opnn_train_loc, opnn_train_target, batchsize = batch);
    test_data = DataLoader(opnn_test_ic, opnn_test_loc, opnn_test_target, batchsize = batch);
    return train_data, test_data, u_train, u_test, x, t # THIS X WILL NEED TO CHANGE ONCE WE EXPORT THE FULL X DOMAIN...
end

"""
    save_model(branch,trunk,n_epoch)

FINISH!!!

"""
function save_model(branch,trunk,n_epoch)
    @save @sprintf("branch_epochs_%i.bson",n_epoch) branch
    @save @sprintf("trunk_epochs_%i.bson",n_epoch) trunk
end

"""
    save_data(train_data,test_data,u_test,u_train,n_epoch,number_solution_points,loss_all_train)

FINISH!!!

"""
function save_data(train_data,test_data,u_test,u_train,n_epoch,number_solution_points,loss_all_train)
    number_train_functions = Int(size(train_data.data[1],2)/number_solution_points);
    number_test_functions = Int(size(test_data.data[1],2)/number_solution_points);
    @save @sprintf("u_sol_train_functions_%i.bson",number_train_functions) u_train
    @save @sprintf("u_sol_test_functions_%i.bson",number_test_functions) u_test
    train_ic = train_data.data[1];
    train_loc = train_data.data[2];
    train_sol = train_data.data[3];
    test_ic = test_data.data[1];
    test_loc = test_data.data[2];
    test_sol = test_data.data[3];
    @save @sprintf("train_ic_data_%i.bson",number_train_functions) train_ic
    @save @sprintf("test_ic_data_%i.bson",number_test_functions) test_ic
    @save @sprintf("train_loc_data_%i.bson",number_train_functions) train_loc
    @save @sprintf("test_loc_data_%i.bson",number_test_functions) test_loc
    @save @sprintf("train_target_data_%i.bson",number_train_functions) train_sol
    @save @sprintf("test_target_data_%i.bson",number_test_functions) test_sol
    @save @sprintf("train_loss_epochs_%i.bson",n_epoch) loss_all_train
end

"""
    load_data(n_epoch,number_train_functions,number_test_functions)

FINISH!!!

"""
function load_data(n_epoch,number_train_functions,number_test_functions)

end
