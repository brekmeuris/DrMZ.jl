"""
    periodic_fill_domain(x_locations)

"""
function periodic_fill_domain(x_locations)
    x_locations_full = vcat(x_locations,(x_locations[end]+(x_locations[end]-x_locations[end-1])))
    return x_locations_full
end

"""
    periodic_fill_solution(solution)

"""
function periodic_fill_solution(solution)
    solution_full = zeros(size(solution,1),(size(solution,2)+1));
    for i in 1:size(solution,1)
        solution_full[i,1:end-1] = solution[i,:];
        solution_full[i,end] = solution[i,1];
    end
    return solution_full
end

"""
    error_test_sse(target,prediction)

"""
function error_test_sse(target,prediction)
    error_test_se = zeros(size(target,1),size(target,2));
    for j in 1:size(target,1)
        for i in 1:size(target,2)
            error_test_se[j,i] = (prediction[j,i] - target[j,i]).^2;
        end
    end
    return error_test_se
end

"""
    error_test_rel(target,prediction)

"""
function error_test_rel(target,prediction)
    error_test_rel = zeros(size(target,1),size(target,2));
    for j in 1:size(target,1)
        for i in 1:size(target,2)
            error_test_rel[j,i] = (prediction[j,i] - target[j,i])/target[j,i];
        end
    end
    return error_test_rel
end

"""
    solution_interpolation(t_span_original,x_locations_original,t_span_interpolate,x_locations_interpolate,solution)

"""
function solution_interpolation(t_span_original,x_locations_original,t_span_interpolate,x_locations_interpolate,solution)
    solution_interpolate_fit = LinearInterpolation((t_span_original,x_locations_original), solution);
    solution_interpolate = zeros(size(t_span_interpolate,1),size(x_locations_interpolate,1));
    for i in 1:size(t_span_interpolate,1)
        for j in 1:size(x_locations_interpolate,1)
            solution_interpolate[i,j] = solution_interpolate_fit(t_span_interpolate[i],x_locations_interpolate[j]);
        end
    end
    return solution_interpolate
end

"""
    reduced_initial_condition(L1,L2,N,x_locations,initial_condition)

"""
function reduced_initial_condition(L1,L2,N,x_locations,initial_condition)
    dL = abs(L2-L1);
    j = reduce(vcat,[0:1:N-1]);
    x_interpolate = (dL.*j)./N;
    ic_interpolate_fit = LinearInterpolation(x_locations,initial_condition)
    ic_interpolate = [ic_interpolate_fit(i) for i in x_interpolate];
    return x_interpolate, ic_interpolate
end

"""
    reduced_initial_condition_full(L1,L2,N,x_locations,initial_condition)

"""
function reduced_initial_condition_full(L1,L2,N,x_locations,initial_condition)
    dL = abs(L2-L1);
    j = reduce(vcat,[0:1:(N-1)-1]);
    x_interpolate = zeros(N);
    x_interpolate[1:(N-1)] = (dL.*j)./(N-1);
    x_interpolate[N] = L2;
    ic_interpolate_fit = LinearInterpolation(x_locations,initial_condition)
    ic_interpolate = [ic_interpolate_fit(i) for i in x_interpolate];
    return x_interpolate, ic_interpolate
end

"""
    mse_error(target,prediction)

"""
function mse_error(target,prediction)
    return (1/size(target[:],1))*sum((prediction[:].-target[:]).^2,);
end

"""
    trapz(x_range,integrand)

Assumes constant step size...

"""
function trapz(x_range,integrand)
    x1 = x_range[2]-x_range[1];
    x2 = x_range[end]-x_range[end-1];
    if x1 != x2
        return nothing
    end
    int = 0;
    for i = 1:length(x_range)-1
        int += (x_range[i+1] - x_range[i])*(integrand[i] + integrand[i+1]);
    end
    return (1/2)*int
end
