using LinearAlgebra
using Random: seed!
using Plots; default(markerstrokecolor=:auto)
using LaTeXStrings
using Distributions:Poisson
using Statistics: mean
using Random: seed!
using JLD

N = 100

r_time = 10
xhow = :real
avg_count = 2
niter = 500

if xhow === :real
        xtrue = cumsum(mod.(1:N, 30) .== 0) .- 1.5
elseif xhow === :complex
        xtrue = (cumsum(mod.(1:N, 30) .== 0) .- 1.5) + im * (cumsum(mod.(1:N, 10) .== 0) .- 4.5)
else
        throw("unknown xhow")
end

phase_shift = x -> iszero(x) ? 1 : sign(xtrue' * x)
nrmse = x -> (norm(x - xtrue .* phase_shift(x)) / norm(xtrue .* phase_shift(x)))
phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)), time(), nrmse(x)]

lf = x -> log10(max(x,1e-16))
grab = (o,i) -> hcat(o...)[i,:]
lg = (o,i) -> lf.(grab(o,i))

include("Wirtinger_flow_fisher.jl")
include("Gerchberg_saxton.jl")
include("poisson_admm.jl")
include("poisson_admm_L1_diff.jl")


M_list = [2000, 3000, 4000, 5000, 6000]

for M in M_list
        global b = zeros(M)
        xout_wf_gau = Array{Array{Any, 1}, 1}(undef, r_time)
        cout_wf_gau = Array{Array{Any, 1}, 1}(undef, r_time)
        xout_wf_pois = Array{Array{Any, 1}, 1}(undef, r_time)
        cout_wf_pois = Array{Array{Any, 1}, 1}(undef, r_time)
        xout_gs = Array{Array{Any, 1}, 1}(undef, r_time)
        cout_gs = Array{Array{Any, 1}, 1}(undef, r_time)
        xout_admm = Array{Array{Any, 1}, 1}(undef, r_time)
        cout_admm = Array{Array{Any, 1}, 1}(undef, r_time)
        xout_admm_l1 = Array{Array{Any, 1}, 1}(undef, r_time)
        cout_admm_l1 = Array{Array{Any, 1}, 1}(undef, r_time)

        global var = 1
        global A = sqrt(var/2) * (randn(M, N) + im * randn(M, N)) # Each iteration, A is different.
        global cons = avg_count / mean(abs2.(A * xtrue))
        global A = sqrt(cons) * A # scale matrix A
        global y_true = abs2.(A * xtrue) .+ b
        global y_pos = rand.(Poisson.(y_true))
        if xhow === :real
                global x0_rand = randn(N)
        elseif xhow === :complex
                global x0_rand = randn(N) + im * randn(N)
        else
                throw("unknown xhow")
        end

        for r = 1:r_time

                local var = 1
                local A = sqrt(var/2) * (randn(M, N) + im * randn(M, N)) # Each iteration, A is different.
                local cons = avg_count / mean(abs2.(A * xtrue))
                local A = sqrt(cons) * A # scale matrix A
                local y_true = abs2.(A * xtrue) .+ b
                local y_pos = rand.(Poisson.(y_true))
                if xhow === :real
                        local x0_rand = randn(N)
                elseif xhow === :complex
                        local x0_rand = randn(N) + im * randn(N)
                else
                        throw("unknown xhow")
                end

                xout_wf_gau[r], cout_wf_gau[r] = Wirtinger_flow_fisher(A,y_pos,b;
                        gradhow = :gaussian, xhow = xhow, x0 = x0_rand, niter = niter, fun = cost_func)
                xout_wf_pois[r], cout_wf_pois[r] = Wirtinger_flow_fisher(A,y_pos,b;
                        gradhow = :poisson, xhow = xhow, x0 = x0_rand, niter = niter, fun = cost_func)
                xout_gs[r], cout_gs[r] = Gerchberg_saxton(A,y_pos,b;
                        xhow = xhow, updatehow =:cg, x0 = x0_rand, niter = niter, fun = cost_func)
                xout_admm[r], cout_admm[r] = poisson_admm(A,y_pos,b;
                        phow =:adaptive, xhow = xhow, updatehow =:cg, x0 = x0_rand, ρ = 16, niter = niter, fun = cost_func);
                xout_admm_l1[r], cout_admm_l1[r] = poisson_admm_L1_diff(A,y_pos,b;
                        phow =:adaptive, ninner = 3, reg1 = 32, reg2 = 32,
                        xhow = xhow, updatehow =:cg, x0 = x0_rand, ρ = 16, niter = niter, fun = cost_func)

        end

# mean_cout_wf_gau = mean(cout_wf_gau)
# mean_cout_wf_pois = mean(cout_wf_pois)
# mean_cout_gs = mean(cout_gs)
# mean_cout_lsmm = mean(cout_lsmm)
# mean_cout_admm = mean(cout_admm)
# mean_cout_lsmm_l1 = mean(cout_lsmm_l1)
# mean_cout_admm_l1 = mean(cout_admm_l1)

        save("./A_gau_b0/A_Gaussian_"*string(xhow)*"_M="*string(M)*".jld",
                "xout_wf_gau", xout_wf_gau,
                "cout_wf_gau", cout_wf_gau,
                "xout_wf_pois", xout_wf_pois,
                "cout_wf_pois", cout_wf_pois,
                "xout_gs", xout_gs,
                "cout_gs", cout_gs,
                "xout_admm", xout_admm,
                "cout_admm", cout_admm,
                "xout_admm_l1", xout_admm_l1,
                "cout_admm_l1", cout_admm_l1)
end
