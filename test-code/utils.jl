# utils.jl
using LinearAlgebra
using LinearMapsAA
using Random: seed!
using Plots; default(markerstrokecolor=:auto)
using MIRT
using LaTeXStrings
using Distributions:Poisson
using Statistics
using Random: seed!
using JLD2
# use JLD2!
using MAT
using SparseArrays
using FFTW
using Optim
using BenchmarkTools

lf = x -> log10(max(x,1e-16))
grab = (o,i) -> hcat(o...)[i,:]
lg = (o,i) -> lf.(grab(o,i))
grab_cost(x, r_time) = mean([grab(x[r], 1) for r = 1:r_time])
grab_time(x, r_time) = mean([grab(x[r], 2) .- x[r][1][2] for r = 1:r_time])
grab_nrmse(x, r_time) = mean([grab(x[r], 3) for r = 1:r_time])

# Running power iterations to find the leading eigenvector
function power_iter(sys, x, niter)
        for iter = 1:niter
                x = sys * x
                x = x / norm(x)
        end
        return x
end
# The adjoint of 1D TV (finite difference)
function TV_adj(y::AbstractVector{<:Number})
    N = length(y)
    x = similar(y, N+1)
    x[1] = -y[1]
    @views x[2:end-1] .= y[1:(end-1)] - y[2:end]
    x[N+1] = y[N]
    return x
end

function cal_nrmse_lbfgs(results, xtrue)
    phase_shift = x -> iszero(x) ? 1 : sign(xtrue' * x)
    nrmse = x -> (norm(x - xtrue .* phase_shift(x)) / norm(xtrue .* phase_shift(x)))
    niter = length(Optim.trace(results))
    nrmse_vec = zeros(niter)
    for i = 1:niter
        nrmse_vec[i] = nrmse(Optim.trace(results)[i].metadata["x"])
    end
    return nrmse_vec
end


function cal_nrmse_lbfgs_mean(results, xtrue)
    niter = minimum([length(Optim.trace(results[n])) for n = 1:length(results)])
    n_tests = length(results)
    nrmse_mat = zeros(niter, n_tests)
    for n = 1:n_tests
        nrmse_mat[:, n] = cal_nrmse_lbfgs(results[n], xtrue)[1:niter]
    end
    nrmse_vec = vec(mean(nrmse_mat, dims = 2))
    return nrmse_vec
end


function cal_time_lbfgs_mean(results)
    niter = minimum([length(Optim.trace(results[n])) for n = 1:length(results)])
    n_tests = length(results)
    time_mat = zeros(niter, n_tests)
    for n = 1:n_tests
        for i = 1:niter
            time_mat[i, n] = Optim.trace(results[n])[i].metadata["time"]
        end
    end
    time_vec = vec(mean(time_mat, dims = 2))
    return time_vec
end


function cal_cost_lbfgs_mean(results)
    niter = maximum([length(Optim.trace(results[n])) for n = 1:length(results)])
    n_tests = length(results)
    cost_mat = zeros(niter, n_tests)
    for n = 1:n_tests
        for i = 1:niter
            cost_mat[i, n] = Optim.trace(results[n])[min(i, length(Optim.trace(results[n])))].value
        end
    end
    cost_vec = vec(mean(cost_mat, dims = 2))
    return cost_vec
end


function remove_gap(x)
    y = copy(x)
    n = length(y)
    mean_d = mean(diff(y))
    for i = 1:n-1
        gap = y[i+1] - y[i]
        if  gap > 5 * mean_d
            y[i+1:end] .-= gap - mean_d
        end
    end
    return y
end

function stop_at_converge(x; rtol = 1e-4)
    y = copy(x)
    for i = 1:length(y) - 1
        if abs((y[i+1] - y[i]) / y[1]) ≤ rtol
            return y[1:i+1]
        end
    end
end


include("../funcs/Wirtinger_flow.jl")
include("../funcs/Wirtinger_flow_huber.jl")
include("../funcs/Wirtinger_flow_inplace1.jl")
include("../funcs/Gerchberg_saxton.jl")
include("../funcs/Gerchberg_saxton_dft.jl")
include("../funcs/LSMM.jl")
include("../funcs/LSMM_TV.jl")
include("../funcs/LSMM_TV_huber.jl")
include("../funcs/LSMM_ODWT.jl")
include("../funcs/ADMM.jl")
include("../funcs/ADMM_TV.jl")
include("../funcs/ADMM_TV_huber.jl")
include("../funcs/ADMM_dft.jl")
include("../funcs/ADMM_ODWT.jl")
include("../funcs/ADMM_ODWT_dft.jl")
