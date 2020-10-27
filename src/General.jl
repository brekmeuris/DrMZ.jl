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
    error_test_sse(target,prediction)

"""
function error_test_rel(target,prediction)
    error_test_rel = zeros(size(target,1),size(target,2));
    for j in 1:size(target,1)
        for i in 1:size(target,2)
            error_test_rel[j,i] = (prediction[j,i] - target[j,i])/target[j,i];;
        end
    end
    return error_test_rel
end
