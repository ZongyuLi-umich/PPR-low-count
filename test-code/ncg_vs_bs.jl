using LinearAlgebra
using Random: seed!
using Plots; default(markerstrokecolor=:auto)
using LaTeXStrings
using Distributions:Poisson
using Statistics: mean
using Random: seed!
include("poisson_admm.jl")
include("LSMM.jl")
include("Gerchberg_saxton.jl")
avg_count = 2
var = 1
# seed!(2333)
N = 100
M = 3000
# r_time = 3
niter = 250

"""
Real case, should do a warm up.
"""
# xtrue = (cumsum(mod.(1:N, 30) .== 0) .- 1.5) + im * (cumsum(mod.(1:N, 10) .== 0) .- 4.5)
xtrue = cumsum(mod.(1:N, 30) .== 0) .- 1.5
# x0_rand = sqrt(var/2) * (randn(N) + im * randn(N))
x0_rand = sqrt(var/2) * randn(N)
A = sqrt(var/2) * (randn(M, N) + im * randn(M, N)) # Each iteration, A is different.
b = 0.5 * ones(M)
cons = avg_count / mean(abs2.(A * xtrue))
A = sqrt(cons) * A # scale matrix A
#     print("Average counts after scaling: ", mean(abs2.(A * xtrue)))
y_true = abs2.(A * xtrue) + b
y_pos = rand.(Poisson.(y_true)); # poisson noise

phase_shift = x -> sign(xtrue' * x)
nrmse = x -> (norm(x - xtrue .* phase_shift(x)) / norm(xtrue .* phase_shift(x)))

phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)), time(), nrmse(x)]

# phi = (v, yi) -> (abs2(v)) - yi * log(abs2(v)) # (v^2 + b) - yi * log(v^2 + b)
# cost_func = (x,iter) -> [sum(phi.(A * x, y_pos)), time(), nrmse(x)]
lf = x -> log10(max(x,1e-16))
grab = (o,i) -> hcat(o...)[i,:]
lg = (o,i) -> lf.(grab(o,i))

r_time = 10
time_gs_cg = zeros(r_time, niter + 1)
nrmse_gs_cg = zeros(r_time, niter + 1)
time_gs_bs = zeros(r_time, niter + 1)
nrmse_gs_bs = zeros(r_time, niter + 1)

time_lsmm_cg = zeros(r_time, niter + 1)
nrmse_lsmm_cg = zeros(r_time, niter + 1)
time_lsmm_bs = zeros(r_time, niter + 1)
nrmse_lsmm_bs = zeros(r_time, niter + 1)

time_admm_cg = zeros(r_time, niter + 1)
nrmse_admm_cg = zeros(r_time, niter + 1)
time_admm_bs = zeros(r_time, niter + 1)
nrmse_admm_bs = zeros(r_time, niter + 1)

for r = 1:r_time
        xout_gs_cg, cout_gs_cg = Gerchberg_saxton(A,y_pos,b; xhow = :real,
                        updatehow =:cg, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_gs_bs, cout_gs_bs = Gerchberg_saxton(A,y_pos,b; xhow = :real,
                        updatehow =:bs, x0 = x0_rand, niter = niter, fun = cost_func)
# xout_lsmm_max_cg, cout_lsmm_max_cg = LSMM(A,y_pos,b; xhow = :real, curvhow =:max, updatehow =:cg, x0 = x0_rand, niter = niter, fun = cost_func)
# xout_lsmm_max_bs, cout_lsmm_max_bs = LSMM(A,y_pos,b; xhow = :real, curvhow =:max, updatehow =:bs, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_lsmm_imp_cg, cout_lsmm_imp_cg = LSMM(A,y_pos,b; xhow = :real, curvhow =:imp,
                        updatehow =:cg, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_lsmm_imp_bs, cout_lsmm_imp_bs = LSMM(A,y_pos,b; xhow = :real, curvhow =:imp,
                        updatehow =:bs, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_admm_adap_cg, cout_admm_adap_cg = poisson_admm(A,y_pos,b; xhow = :real,
                        phow =:constant, updatehow =:cg, ρ = 16, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_admm_adap_bs, cout_admm_adap_bs = poisson_admm(A,y_pos,b; xhow = :real,
                        phow =:constant, updatehow =:bs, ρ = 16, x0 = x0_rand, niter = niter, fun = cost_func)

        time_gs_cg[r, :] = grab(cout_gs_cg, 2) .- cout_gs_cg[1][2]
        nrmse_gs_cg[r, :] = grab(cout_gs_cg, 3)
        time_gs_bs[r, :] = grab(cout_gs_bs, 2) .- cout_gs_bs[1][2]
        nrmse_gs_bs[r, :] = grab(cout_gs_bs, 3)
        time_lsmm_cg[r, :] = grab(cout_lsmm_imp_cg, 2) .- cout_lsmm_imp_cg[1][2]
        nrmse_lsmm_cg[r, :] = grab(cout_lsmm_imp_cg, 3)
        time_lsmm_bs[r, :] = grab(cout_lsmm_imp_bs, 2) .- cout_lsmm_imp_bs[1][2]
        nrmse_lsmm_bs[r, :] = grab(cout_lsmm_imp_bs, 3)
        time_admm_cg[r, :] = grab(cout_admm_adap_cg, 2) .- cout_admm_adap_cg[1][2]
        nrmse_admm_cg[r, :] = grab(cout_admm_adap_cg, 3)
        time_admm_bs[r, :] = grab(cout_admm_adap_bs, 2) .- cout_admm_adap_bs[1][2]
        nrmse_admm_bs[r, :] = grab(cout_admm_adap_bs, 3)
end


# cost_gs_cg = grab(cout_gs_cg, 1)
# time_gs_cg = grab(cout_gs_cg, 2) .- cout_gs_cg[1][2]
# nrmse_gs_cg = grab(cout_gs_cg, 3);
#
# cost_gs_bs = grab(cout_gs_bs, 1)
# time_gs_bs = grab(cout_gs_bs, 2) .- cout_gs_bs[1][2]
# nrmse_gs_bs = grab(cout_gs_bs, 3);

# cost_lsmm_max_cg = grab(cout_lsmm_max_cg, 1)
# time_lsmm_max_cg = grab(cout_lsmm_max_cg, 2) .- cout_lsmm_max_cg[1][2]
# nrmse_lsmm_max_cg = grab(cout_lsmm_max_cg, 3);

# cost_lsmm_max_bs = grab(cout_lsmm_max_bs, 1)
# time_lsmm_max_bs = grab(cout_lsmm_max_bs, 2) .- cout_lsmm_max_bs[1][2]
# nrmse_lsmm_max_bs = grab(cout_lsmm_max_bs, 3);

# cost_lsmm_imp_cg = grab(cout_lsmm_imp_cg, 1)
# time_lsmm_imp_cg = grab(cout_lsmm_imp_cg, 2) .- cout_lsmm_imp_cg[1][2]
# nrmse_lsmm_imp_cg = grab(cout_lsmm_imp_cg, 3);

# cost_lsmm_imp_bs = grab(cout_lsmm_imp_bs, 1)
# time_lsmm_imp_bs = grab(cout_lsmm_imp_bs, 2) .- cout_lsmm_imp_bs[1][2]
# nrmse_lsmm_imp_bs = grab(cout_lsmm_imp_bs, 3);

# cost_admm_adap_cg = grab(cout_admm_adap_cg, 1)
# time_admm_adap_cg = grab(cout_admm_adap_cg, 2) .- cout_admm_adap_cg[1][2]
# nrmse_admm_adap_cg = grab(cout_admm_adap_cg, 3);
#
# cost_admm_adap_bs = grab(cout_admm_adap_bs, 1)
# time_admm_adap_bs = grab(cout_admm_adap_bs, 2) .- cout_admm_adap_bs[1][2]
# nrmse_admm_adap_bs = grab(cout_admm_adap_bs, 3);


scatter(1e3 * mean(time_gs_cg, dims=1)[1:40], mean(nrmse_gs_cg, dims=1)[1:40], color=:orange, markersize = 4, markershape = :circle,
        legend=:topright, legendfontsize = 12, ylims = (0,1.25), yticks = [0,0.25,0.50,0.75,1.0,1.25],
        label = "Gerchberg Saxton CG", xlabel = "time(ms)", xguidefontsize=15, ylabel = "NRMSE", yguidefontsize=15)
scatter!(1e3 * mean(time_gs_bs, dims=1)[1:40], mean(nrmse_gs_bs, dims=1)[1:40], color=:orange, markersize = 3, markershape = :rect, label = "Gerchberg Saxton BS")
# scatter!(1e3 * time_lsmm_max_cg[1:20], nrmse_lsmm_max_cg[1:20], color=:blue, markersize = 4, label = "LSMM max CG")
# scatter!(1e3 * time_lsmm_max_bs[1:20], nrmse_lsmm_max_bs[1:20], color=:cyan, markersize = 4, label = "LSMM max BS")
scatter!(1e3 * mean(time_lsmm_cg, dims=1)[1:60], mean(nrmse_lsmm_cg, dims=1)[1:60], color=:green, markershape = :circle, markersize = 4, label = "LSMM CG")
scatter!(1e3 * mean(time_lsmm_bs, dims=1)[1:40], mean(nrmse_lsmm_bs, dims=1)[1:40], color=:green, markershape = :rect, markersize = 3, label = "LSMM BS")
scatter!(1e3 * mean(time_admm_cg, dims=1)[1:100], mean(nrmse_admm_cg, dims=1)[1:100], color=:purple, markershape = :circle, markersize = 4, label = "ADMM CG")
scatter!(1e3 * mean(time_admm_bs, dims=1)[1:100], mean(nrmse_admm_bs, dims=1)[1:100], color=:purple, markershape = :rect, markersize = 3, label = "ADMM BS")

# title!("NRMSE vs time (M = 3000, real x0)")
savefig("nrmse_vs_time_cgvsbs"*string(avg_count)*"_real.pdf")

"""
Complex case, should do a warm up.
"""
xtrue = (cumsum(mod.(1:N, 30) .== 0) .- 1.5) + im * (cumsum(mod.(1:N, 10) .== 0) .- 4.5)
# xtrue = cumsum(mod.(1:N, 30) .== 0) .- 1.5
x0_rand = sqrt(var/2) * (randn(N) + im * randn(N))
# x0_rand = sqrt(var/2) * randn(N)
A = sqrt(var/2) * (randn(M, N) + im * randn(M, N)) # Each iteration, A is different.
b = ones(M)
cons = avg_count / mean(abs2.(A * xtrue))
A = sqrt(cons) * A # scale matrix A
#     print("Average counts after scaling: ", mean(abs2.(A * xtrue)))
y_true = abs2.(A * xtrue) + b
y_pos = rand.(Poisson.(y_true)); # poisson noise

phase_shift = x -> sign(xtrue' * x)
nrmse = x -> (norm(x - xtrue .* phase_shift(x)) / norm(xtrue .* phase_shift(x)))

phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)), time(), nrmse(x)]

lf = x -> log10(max(x,1e-16))
grab = (o,i) -> hcat(o...)[i,:]
lg = (o,i) -> lf.(grab(o,i))
# phi = (v, yi) -> (abs2(v)) - yi * log(abs2(v)) # (v^2 + b) - yi * log(v^2 + b)
# cost_func = (x,iter) -> [sum(phi.(A * x, y_pos)), time(), nrmse(x)]

time_gs_cg = zeros(r_time, niter + 1)
nrmse_gs_cg = zeros(r_time, niter + 1)
time_gs_bs = zeros(r_time, niter + 1)
nrmse_gs_bs = zeros(r_time, niter + 1)

time_lsmm_cg = zeros(r_time, niter + 1)
nrmse_lsmm_cg = zeros(r_time, niter + 1)
time_lsmm_bs = zeros(r_time, niter + 1)
nrmse_lsmm_bs = zeros(r_time, niter + 1)

time_admm_cg = zeros(r_time, niter + 1)
nrmse_admm_cg = zeros(r_time, niter + 1)
time_admm_bs = zeros(r_time, niter + 1)
nrmse_admm_bs = zeros(r_time, niter + 1)

for r = 1:r_time
        xout_gs_cg, cout_gs_cg = Gerchberg_saxton(A,y_pos,b; xhow = :complex,
                        updatehow =:cg, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_gs_bs, cout_gs_bs = Gerchberg_saxton(A,y_pos,b; xhow = :complex,
                        updatehow =:bs, x0 = x0_rand, niter = niter, fun = cost_func)
# xout_lsmm_max_cg, cout_lsmm_max_cg = LSMM(A,y_pos,b; xhow = :real, curvhow =:max, updatehow =:cg, x0 = x0_rand, niter = niter, fun = cost_func)
# xout_lsmm_max_bs, cout_lsmm_max_bs = LSMM(A,y_pos,b; xhow = :real, curvhow =:max, updatehow =:bs, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_lsmm_imp_cg, cout_lsmm_imp_cg = LSMM(A,y_pos,b; xhow = :complex, curvhow =:imp,
                        updatehow =:cg, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_lsmm_imp_bs, cout_lsmm_imp_bs = LSMM(A,y_pos,b; xhow = :complex, curvhow =:imp,
                        updatehow =:bs, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_admm_adap_cg, cout_admm_adap_cg = poisson_admm(A,y_pos,b; xhow = :complex,
                        phow =:constant, updatehow =:cg, ρ = 16, x0 = x0_rand, niter = niter, fun = cost_func)
        xout_admm_adap_bs, cout_admm_adap_bs = poisson_admm(A,y_pos,b; xhow = :complex,
                        phow =:constant, updatehow =:bs, ρ = 16, x0 = x0_rand, niter = niter, fun = cost_func)

        time_gs_cg[r, :] = grab(cout_gs_cg, 2) .- cout_gs_cg[1][2]
        nrmse_gs_cg[r, :] = grab(cout_gs_cg, 3)
        time_gs_bs[r, :] = grab(cout_gs_bs, 2) .- cout_gs_bs[1][2]
        nrmse_gs_bs[r, :] = grab(cout_gs_bs, 3)
        time_lsmm_cg[r, :] = grab(cout_lsmm_imp_cg, 2) .- cout_lsmm_imp_cg[1][2]
        nrmse_lsmm_cg[r, :] = grab(cout_lsmm_imp_cg, 3)
        time_lsmm_bs[r, :] = grab(cout_lsmm_imp_bs, 2) .- cout_lsmm_imp_bs[1][2]
        nrmse_lsmm_bs[r, :] = grab(cout_lsmm_imp_bs, 3)
        time_admm_cg[r, :] = grab(cout_admm_adap_cg, 2) .- cout_admm_adap_cg[1][2]
        nrmse_admm_cg[r, :] = grab(cout_admm_adap_cg, 3)
        time_admm_bs[r, :] = grab(cout_admm_adap_bs, 2) .- cout_admm_adap_bs[1][2]
        nrmse_admm_bs[r, :] = grab(cout_admm_adap_bs, 3)
end

scatter(1e3 * mean(time_gs_cg, dims=1)[1:60], mean(nrmse_gs_cg, dims=1)[1:60], color=:orange, markersize = 4, markershape = :circle,
        legend=:topright, legendfontsize = 12, ylims = (0,1.25), yticks = [0,0.25,0.50,0.75,1.0,1.25],
        label = "Gerchberg Saxton CG", xlabel = "time(ms)", xguidefontsize=15, ylabel = "NRMSE", yguidefontsize=15)
scatter!(1e3 * mean(time_gs_bs, dims=1)[1:60], mean(nrmse_gs_bs, dims=1)[1:60], color=:orange, markersize = 3, markershape = :rect, label = "Gerchberg Saxton BS")
# scatter!(1e3 * time_lsmm_max_cg[1:20], nrmse_lsmm_max_cg[1:20], color=:blue, markersize = 4, label = "LSMM max CG")
# scatter!(1e3 * time_lsmm_max_bs[1:20], nrmse_lsmm_max_bs[1:20], color=:cyan, markersize = 4, label = "LSMM max BS")
scatter!(1e3 * mean(time_lsmm_cg, dims=1)[1:80], mean(nrmse_lsmm_cg, dims=1)[1:80], color=:green, markershape = :circle, markersize = 4, label = "LSMM CG")
scatter!(1e3 * mean(time_lsmm_bs, dims=1)[1:60], mean(nrmse_lsmm_bs, dims=1)[1:60], color=:green, markershape = :rect, markersize = 3, label = "LSMM BS")
scatter!(1e3 * mean(time_admm_cg, dims=1)[1:150], mean(nrmse_admm_cg, dims=1)[1:150], color=:purple, markershape = :circle, markersize = 4, label = "ADMM CG")
scatter!(1e3 * mean(time_admm_bs, dims=1)[1:120], mean(nrmse_admm_bs, dims=1)[1:120], color=:purple, markershape = :rect, markersize = 3, label = "ADMM BS")

# title!("NRMSE vs time (M = 3000, real x0)")
savefig("nrmse_vs_time_cgvsbs"*string(avg_count)*"_complex.pdf")
