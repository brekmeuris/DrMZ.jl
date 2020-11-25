"""

"""
function generate_opnn_results(pde_function;random_integer = "None",kws...)

    args = Args(;);

    # Load all the data
    branch, trunk, train_ic, train_loc, train_sol, test_ic, test_loc, test_sol, u_test = load_data(args.n_epoch,args.num_train_functions,args.num_test_functions,pde_function)

    train_MSE = loss_all(branch,trunk,train_ic,train_loc,train_sol);
    test_MSE = loss_all(branch,trunk,test_ic,test_loc,test_sol);
    println("Training set MSE: $train_MSE, test set MSE: $test_MSE")

    # Set up x domain and wave vector for spectral solution
    dL = abs(args.L2 - args.L1);
    j = reduce(vcat,[0:1:args.num_sensors-1]);
    x = (dL.*j)./args.num_sensors;
    ic_exact = sin.(pi*x).^2;
    dt = 1e-3;
    t = reduce(vcat,[args.L1:dt:args.L2]);

    if random_integer == "None"
        rand_int = rand(1:args.num_test_functions);
    else
        rand_int = random_integer;
    end

    # Comparisons vs random test IC
    pltrand_test = plot(x,test_ic[:,rand_int*args.num_sol_points],label = "Random IC",palette=:viridis,frame=:box,linewidth=:2,xlims=(x[1],x[end]),xlabel="x",ylabel="u(x)")
    plot!(pltrand_test,x,test_ic[:,rand_int*args.num_sol_points],label = "Sensor locations",seriestype=:scatter,palette=:viridis,frame=:box,markeralpha = 0.4)
    display(pltrand_test)

    savefig(pltrand_test, @sprintf("random_ic_%s.png",pde_function))

    pltexact = plot(heatmap(periodic_fill_domain(x),t,periodic_fill_solution(u_test[:,:,rand_int])),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = "x", ylabel = "t",right_margin = 8mm)#,xlims=(x[1],x[end]),ylims=(t_data[1],t_data[end]))
    display(pltexact)

    savefig(pltexact, @sprintf("exact_fourier_solution_random_ic_%s.png",pde_function))

    u_predict = predict(branch,trunk,test_ic[:,rand_int*args.num_sol_points],x,t);
    pltpredict = plot(heatmap(periodic_fill_domain(x),t,periodic_fill_solution(u_predict)),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = "x", ylabel = "t",right_margin = 8mm)#,xlims=(x[1],x[end]),ylims=(t_data[1],t_data[end]))
    display(pltpredict)

    savefig(pltpredict, @sprintf("opnn_solution_random_ic_epochs_%i_%s.png",args.n_epoch,pde_function))

    # anim = @animate for i in 1:size(u_test[:,:,rand_int],1)
    #     plot(x,[u_test[i,:,rand_int],u_predict[i,:]],label = ["Fourier solution" "Neural network prediction"],linewidth=:2,palette=:viridis,frame=:box,xlims=(x[1],x[end]));
    # end
    # gif(anim, @sprintf("anim_fourier_vs_opnn_%s.gif",pde_function), fps = 30);

    opnn_error = error_test_sse(u_test[:,:,rand_int],u_predict);
    plterror = plot(heatmap(x,t,opnn_error),fillcolor=cgrad(ColorSchemes.viridis.colors),label=false,xlabel = "x", ylabel = "t",right_margin = 8mm)#,xlims=(x[1],x[end]),ylims=(t_data[1],t_data[end]))
    display(plterror)

    savefig(plterror, @sprintf("opnn_vs_exact_solution_random_ic_sqerror_epochs_%i_%s.png",args.n_epoch,pde_function))

end
