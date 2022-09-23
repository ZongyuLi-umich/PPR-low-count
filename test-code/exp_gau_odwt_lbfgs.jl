include("utils.jl")

r_time = 5
xhow = :real

# Define xtrue
xtrue = load("./test-data-2D/UM-1817-shepplogan.jld2")["x1"]
N = size(xtrue, 1)
avg_count = 0.25
niter = 500

phase_shift = x -> sign.(vec(xtrue)' * vec(x))
nrmse = x -> iszero(x) ? 1.0 : (norm(vec(x) - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x)))
phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
grad_phi = (v, yi, bi) -> 2 * v * (1 - yi / (abs2(v) + bi))
T, scales, mfun = Aodwt((N, N); T = ComplexF64) # T is real
β = 2
cost_func = (x,iter) -> [sum(Float64, phi.(A * vec(x), y_pos, b)) + β * norm(Float64.(vec(T * reshape(x,N,N))), 1), time(), nrmse(x)]
f(x) = sum(Float64, phi.(A * vec(x), y_pos, b)) + β * norm(Float64.(vec(T * reshape(x,N,N))), 1)
g!(y, x) = copyto!(y, real(A' * grad_phi.(A * vec(x), y_pos, b)) + β * vec(T' * sign.(T * reshape(x, N, N))))
M = 80000

A = sqrt(1/2) * (randn(M, N^2) + im * randn(M, N^2)) # Each iteration, A is different.
cons = avg_count / mean(abs2.(A * vec(xtrue)))
A = sqrt(cons) * A # scale matrix A
b = 0.1 * ones(M)
y_true = abs2.(A * xtrue[:]) .+ b
y_pos = rand.(Poisson.(y_true))
x0_rand = vec(rand(N, N))
# global B = LinearMapAA(x -> A'*(Diagonal(y_pos-b)*(A*x)), (N*N, N*N); T=ComplexF32)
# global x0_spectral = power_iter(B, x0_rand, 50)
x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)
α = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
x0_spectral = abs.(α * x0_spectral)
LAA = norm(A'*A, 2)

results_LBFGS = optimize(f, g!, x0_spectral, LBFGS(),
                                    Optim.Options(store_trace = true,
                                                  show_trace = false,
                                                  extended_trace = true))

xout_lsmm_ODWT, cout_lsmm_ODWT = LSMM_ODWT(A,y_pos,b,LAA;
            curvhow = :imp, xhow = xhow, reg = β,
            x0 = Optim.minimizer(results_LBFGS), niter = 50, ninner = 5,
            fun = cost_func)
plot(grab(cout_lsmm_ODWT, 1))
xout_admm_ODWT, cout_admm_ODWT = ADMM_ODWT(A,y_pos,b,LAA;
            phow =:adaptive, reg = β, xhow = xhow,
            x0 = Optim.minimizer(results_LBFGS), ρ = 32, niter = 50, fun = cost_func)


# plot data
# M = 80000
# data = load("./result/A_gau_ODWT_speed/optimal_init_new_exp_count=0.25_M="*string(M)*".jld2")
# scatter(1e3 * remove_gap(grab_time(data["cout_admm_ODWT"], r_time))[1:400],
#                         log10.(grab_cost(data["cout_admm_ODWT"], r_time))[1:400],
#                         color = "blue",
#                         label = "ADMM ODWT", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
#                         xlabel = "Time (ms)", ylabel = L"\log(f(x))")
# scatter!(1e3 * remove_gap(grab_time(data["cout_lsmm_ODWT"], r_time)[1:200]),
#                         log10.(grab_cost(data["cout_lsmm_ODWT"], r_time))[1:200],
#                         color = "red", legendfontsize = 12, label = "LSMM ODWT")
# scatter!(1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"]))[1:end],
#                         log10.(cal_cost_lbfgs_mean(data["results_LBFGS"]))[1:end],
#                         color = "purple", legendfontsize = 12, label = "LBFGS")
# savefig("./result/A_gau_ODWT_speed/optimal_init_new_exp_count=0.25_M="*string(M)*".pdf")
# NRMSE
# scatter(1e3 * remove_gap(grab_time(data["cout_admm_ODWT"], r_time)[1:500]),
#         1e2 * grab_nrmse(data["cout_admm_ODWT"], r_time)[1:500],
#                     color = "blue",
#                     label = "ADMM ODWT",
#                     legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
#                     xlabel = "Time (ms)", ylabel = "NRMSE (%)")
# scatter!(1e3 * remove_gap(grab_time(data["cout_lsmm_ODWT"], r_time)[1:280]),
#                     1e2 * grab_nrmse(data["cout_lsmm_ODWT"], r_time)[1:280],
#                     legendfontsize = 12,
#                     color = "red", label = "LSMM ODWT")
# savefig("./result/A_gau_ODWT_speed/optimal_init_nrmse_exp_count=0.25_M="*string(M)*".pdf")
