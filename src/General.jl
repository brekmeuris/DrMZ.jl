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
            # error_test_rel[j,i] = (prediction[j,i] - target[j,i])/target[j,i];
            error_test_rel[j,i] = (prediction[j,i]-target[j,i])^2/(target[j,i])^2;
        end
    end
    return error_test_rel
end

"""
    solution_interpolation(t_span_original,x_locations_original,t_span_interpolate,x_locations_interpolate,solution)

"""
function solution_interpolation(t_span_original,x_locations_original,t_span_interpolate,x_locations_interpolate,solution)
    # solution_interpolate_fit = LinearInterpolation((t_span_original,x_locations_original), solution);
    solution_interpolate_fit = interpolate((t_span_original,x_locations_original),solution,Gridded(Linear()));
    # solution_interpolate_fit = interpolate(solution,BSpline(Quadratic(Line(OnCell()))));
    # solution_interpolate_fit = CubicSplineInterpolation((t_span_original,x_locations_original),solution[i,:]);
    solution_interpolate = zeros(size(t_span_interpolate,1),size(x_locations_interpolate,1));
    for i in 1:size(t_span_interpolate,1)
        # solution_interpolate_fit = CubicSplineInterpolation(x_locations_original,solution[i,:]);
        for j in 1:size(x_locations_interpolate,1)
            solution_interpolate[i,j] = solution_interpolate_fit(t_span_interpolate[i],x_locations_interpolate[j]);
            # solution_interpolate[i,j] = solution_interpolate_fit(x_locations_interpolate[j]);
        end
    end
    return solution_interpolate
end
# function solution_interpolation(t_span_original,x_locations_original,t_span_interpolate,x_locations_interpolate,solution)
#     # solution_interpolate_fit = LinearInterpolation((t_span_original,x_locations_original), solution);
#     solution_interpolate_fit = interpolate((t_span_original,x_locations_original),solution,Gridded(Linear()));
#     # solution_interpolate_fit = interpolate(solution,BSpline(Quadratic(Line(OnCell()))));
#     solution_interpolate = zeros(size(t_span_interpolate,1),size(x_locations_interpolate,1));
#     for i in 1:size(t_span_interpolate,1)
#         for j in 1:size(x_locations_interpolate,1)
#             solution_interpolate[i,j] = solution_interpolate_fit(t_span_interpolate[i],x_locations_interpolate[j]);
#         end
#     end
#     return solution_interpolate
# end

"""
    reduced_initial_condition(L1,L2,N,x_locations,initial_condition)

"""
function reduced_initial_condition(L1,L2,N,x_locations,initial_condition)
    dL = abs(L2-L1);
    j = reduce(vcat,[0:1:N-1]);
    x_reduced = (dL.*j)./N;
    ind = [findall(x_reduced[i].==x_locations)[1] for i in 1:length(x_reduced)];
    ic_reduced = zeros(size(ind,1));
    for i in 1:size(ind,1)
        ic_reduced[i] = initial_condition[ind[i]];
    end
    # ic_interpolate_fit = LinearInterpolation(x_locations,initial_condition)
    # ic_interpolate = [ic_interpolate_fit(i) for i in x_reduced];
    return x_reduced, ic_reduced#, ic_interpolate
end

# """
#     reduced_initial_condition_full(L1,L2,N,x_locations,initial_condition)
#
# """
# function reduced_initial_condition_full(L1,L2,N,x_locations,initial_condition)
#     dL = abs(L2-L1);
#     j = reduce(vcat,[0:1:(N-1)-1]);
#     x_interpolate = zeros(N);
#     x_interpolate[1:(N-1)] = (dL.*j)./(N-1);
#     x_interpolate[N] = L2;
#     ic_interpolate_fit = LinearInterpolation(x_locations,initial_condition)
#     ic_interpolate = [ic_interpolate_fit(i) for i in x_interpolate];
#     return x_interpolate, ic_interpolate
# end

"""
    mse_error(target,prediction)

"""
function mse_error(target,prediction)
    return (1/size(target[:],1))*sum((prediction[:].-target[:]).^2,);
end

"""
    norm_rel_error(target,prediction)

"""
function norm_rel_error(target,prediction)
    if size(target) != size(prediction)
        println("Target size and prediction size do not match!")
        return nothing
    end
    error = zeros(size(target,1))
    for i in 1:size(target,1)
        error[i] = norm(prediction[i,:]-target[i,:])^2/norm(target[i,:])^2
    end
    return error
end

"""
    trapz1(x_range,integrand)

Numerical integration using the single-application trapezoidal rule.

"""
function trapz1(h,f1,f2)
    return h*((f1+f2)/2)
end

"""
    trapz(x_range,integrand)

Numerical integration using the multi-application trapezoidal rule.

"""
function trapz(x_range,integrand)
    int = 0.0;
    for i = 2:length(x_range)
        int += ((integrand[i-1] + integrand[i])/2)*(x_range[i] - x_range[i-1]);
    end
    return int
end

"""
    simpson13(h,f1,f2)

Numerical integration using the single-application Simpson's 1/3 rule.

"""
function simpson13(h,f1,f2,f3)
    return 2*h*((f1+4*f2+f3)/6);
end

"""
    simpson38(x_range,integrand)

Numerical integration using the single-application Simpson's 3/8 rule.

"""
function simpson38(h,f1,f2,f3,f4)
    return 3*h*((f1+3*(f2+f3)+f4)/8);
end

"""
    simpson(x_range,integrand)

Numerical integration using the multi-application Simpson's and trapezoid rules.

"""
function simpson(x_range,integrand)
    h = x_range[2]-x_range[1];
    k = 1;
    int = 0.0;
    for i = 2:length(x_range)-1
        hf = x_range[i+1]-x_range[i];
        if abs(h-hf) < 0.000001
            if k == 3
                int += simpson13(h,integrand[i-3],integrand[i-2],integrand[i-1]);
                k -= 1;
            else
                k += 1;
            end
        else
            if k == 1
                int += trapz1(h,integrand[i-1],integrand[i]);
            elseif k == 2
                int += simpson13(h,integrand[i-2],integrand[i-1],integrand[i]);
            else
                int += simpson38(h,integrand[i-3],integrand[i-2],integrand[i-1],integrand[i]);
            end
            k = 1;
        end
        h = hf;
    end
    if k == 1
        int += trapz1(h,integrand[end-1],integrand[end]);
    elseif k == 2
        int += simpson13(h,integrand[end-2],integrand[end-1],integrand[end]);
    else
        int += simpson38(h,integrand[end-3],integrand[end-2],integrand[end-1],integrand[end]);
    end
    return int
end

"""
    solution_spatial_sampling(x_prediction,x_target,solution)

"""
function solution_spatial_sampling(x_prediction,x_target,solution)
    ind = [findall(x_prediction[i].==x_target)[1] for i in 1:length(x_prediction)];
    solution_reduced = zeros(size(solution,1),size(ind,1));
    for i in 1:size(ind,1)
        solution_reduced[:,i] = solution[:,ind[i]]
    end
    return solution_reduced
end

"""
    fft_norm(solution)

"""
function fft_norm(solution)
    N = size(solution,1);
    return (1/N)*fft(solution)
end

"""
    ifft_norm(solution)

"""
function ifft_norm(solution)
    N = size(solution,1);
    return N*ifft(solution)
end
