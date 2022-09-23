include("utils.jl")

r_time = 5
xhow = :real

# Define xtrue
xtrue = load("./test-data-2D/UM-1817-shepplogan.jld2")["x1"]
N = size(xtrue, 1)
avg_count = 0.25
niter = 100

phase_shift = x -> sign.(vec(xtrue)' * x)
nrmse = x -> iszero(x) ? 1.0 : (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x)))
phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
grad_phi = (v, yi, bi) -> 2 * v * (1 - yi / (abs2(v) + bi))
T, scales, mfun = Aodwt((N, N); T = ComplexF64) # T is real
β = 2
cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)) + β * norm(vec(T * reshape(x, N, N)), 1), time(), nrmse(x)]
f(x) = sum(phi.(A * vec(x), y_pos, b)) + β * norm(vec(T * reshape(x, N, N)), 1)
g!(y, x) = copyto!(y, A' * grad_phi.(A * vec(x), y_pos, b) + β * vec(T' * sign.(T * reshape(x, N, N))))
grad!(y, Ax, Tx) = copyto!(y, real(A' * grad_phi.(Ax, y_pos, b) + β * vec(T' * sign.(reshape(Tx, N, N)))))
num_mask = 100

K = 2 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]

(xout_lsmm_ODWT, cout_lsmm_ODWT,
        xout_admm_ODWT, cout_admm_ODWT) = [Array{Array{Any, 1}, 1}(undef, r_time) for _ = 1:4]
results_LBFGS = Vector(undef, r_time)
results_WFinplace = Vector(undef, r_time)

for r = 1:r_time
    global A1 = LinearMapAA(
            x -> vec(fft(pad(reshape(x, N, N)))),
            y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
            (K*K, N*N), (name="fft2D",), T=ComplexF64)
    global A = A1
    global mask_list = Array{AbstractArray{<:Number}, 1}(undef, num_mask)
    for i = 1:num_mask
        mask = rand(N, N) .< 0.5
        A_temp = LinearMapAA(
                    x -> vec(fft(pad(reshape(x, N, N) .* mask))),
                    y -> (K*K) * vec(unpad(ifft(reshape(y, K, K))) .* mask),
                    (K*K, N*N), (name="fft2D",), T=ComplexF64)
        global A = vcat(A, A_temp)
        global mask_list[i] = vec(mask)
    end
    global cons = avg_count / mean(abs2.(A * vec(xtrue)))
    global A = sqrt(cons) * A # scale matrix A
    # absA = sqrt(cons) * absA # scale absolute matrix
    global M = size(A, 1)
    global b = 0.1 * ones(M)
    global y_true = abs2.(A * vec(xtrue)) .+ b
    global y_pos = rand.(Poisson.(y_true))
    global ATA = K*K*cons*Diagonal(sum(mask_list) .+ 1)
    global LAA = maximum(ATA)
    global ATA_inv = inv(ATA)

    global x0_rand = vec(rand(N, N))
    global B = LinearMapAA(x -> A'*(Diagonal(y_pos ./(y_pos .+ 1))*(A*x)), (N*N, N*N); T=ComplexF32)
    global x0_spectral = power_iter(B, x0_rand, 50)
    global scale_factor = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
    global x0_spectral = abs.(scale_factor * x0_spectral)

    state = WFState(x0_spectral, grad!, A; β = β, α = scale_factor, R = T, niter = niter, xhow = xhow)
    Wirtinger_flow_inplace_reg!(A, b, niter, state; R = T)
    state.cost_vec[1] = f(x0_spectral)
    for i = 2:niter+1
        state.cost_vec[i] = f(state.x_hist[i-1, :])
    end
    results_WFinplace[r] = [state.cost_vec state.timer]

    results_LBFGS[r] = optimize(f, g!, ComplexF64.(x0_spectral), LBFGS(), Optim.Options(store_trace = true,
                                                                            show_trace = false,
                                                                            extended_trace = true))

    xout_lsmm_ODWT[r], cout_lsmm_ODWT[r] = LSMM_ODWT(A,y_pos,b,LAA;
                curvhow = :imp, xhow = xhow, reg = β,
                x0 = x0_spectral, niter = niter, ninner = 5, fun = cost_func)
    xout_admm_ODWT[r], cout_admm_ODWT[r] = ADMM_ODWT_dft(A,ATA_inv,y_pos,b;
                phow =:adaptive, reg = β, xhow = xhow,
                x0 = x0_spectral, ρ = 32, niter = niter, fun = cost_func)
end

save("./result/A_dft_ODWT_speed/optimal_init_new_exp_count="*string(avg_count)*"_num_mask="*string(num_mask)*".jld2",
                "xout_lsmm_ODWT", xout_lsmm_ODWT,
                "cout_lsmm_ODWT", cout_lsmm_ODWT,
                "xout_admm_ODWT", xout_admm_ODWT,
                "cout_admm_ODWT", cout_admm_ODWT,
                "results_LBFGS", results_LBFGS,
                "results_WFinplace", results_WFinplace)

# plot data
data = load("./result/A_dft_ODWT_speed/optimal_init_new_exp_count="*string(avg_count)*"_num_mask="*string(num_mask)*".jld2")
cost_wf_inplace = [Float64.(data["results_WFinplace"][i][:, 1]) for i = 1:length(data["results_WFinplace"])]
time_wf_inplace = [Float64.(data["results_WFinplace"][i][:, 2]) for i = 1:length(data["results_WFinplace"])]
scatter(1e3 * remove_gap(grab_time(data["cout_admm_ODWT"], r_time)),
                        log10.(grab_cost(data["cout_admm_ODWT"], r_time)),
                        color = "blue",
                        label = "ADMM ODWT", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
                        xlabel = "Time (ms)", ylabel = L"\log(f(x))")
scatter!(1e3 * remove_gap(grab_time(data["cout_lsmm_ODWT"], r_time)),
                        log10.(grab_cost(data["cout_lsmm_ODWT"], r_time)),
                        color = "red", legendfontsize = 12, label = "LSMM ODWT")
scatter!(1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
                        log10.(4e2 .+ cal_cost_lbfgs_mean(data["results_LBFGS"])),
                        color = "purple", legendfontsize = 12, label = "LBFGS")
scatter!(1e3 * remove_gap(cal_time_wf_inplace(time_wf_inplace)),
                        log10.(cal_cost_wf_inplace(cost_wf_inplace)),xlims = (0, 1e4),
                        color = "magenta", legendfontsize = 12, label = "WF Poisson ODWT")
savefig("./result/A_dft_ODWT_speed/cost_fun_optimal_init_new_num_mask="*string(num_mask)*"count="*string(avg_count)*".pdf")

# NRMSE
# scatter(1e3 * remove_gap(grab_time(data["cout_admm_ODWT"], r_time))[1:150],
#                         1e2 * (grab_nrmse(data["cout_admm_ODWT"], r_time))[1:150],
#                         color = "blue",
#                         label = "ADMM ODWT", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
#                         xlabel = "Time (ms)", ylabel = "NRMSE (%)")
# scatter!(1e3 * remove_gap(grab_time(data["cout_lsmm_ODWT"], r_time))[1:70],
#                         1e2 * (grab_nrmse(data["cout_lsmm_ODWT"], r_time))[1:70],
#                         color = "red", legendfontsize = 12, label = "LSMM ODWT")
#
# savefig("./result/A_dft_ODWT_speed/nrmse_num_mask="*string(num_mask)*"count="*string(avg_count)*".pdf")

# visualize the data
# data = load("./result/A_dft_ODWT_speed/optimal_init_exp_count="*string(avg_count)*"_num_mask="*string(num_mask)*".jld2")
# jim(Float64.(reshape(data["xout_admm_ODWT"][1], 32, 32)), xlims = (1, 32),
#     title = "NRMSE = "*string(round(100 * nrmse(data["xout_admm_ODWT"][1])))*"%")
# savefig("./result/A_dft_ODWT_speed/admm_nrmse.pdf")
# jim(Float64.(reshape(data["xout_lsmm_ODWT"][1], 32, 32)), xlims = (1, 32),
#     title = "NRMSE = "*string(round(100 * nrmse(data["xout_lsmm_ODWT"][1])))*"%")
# savefig("./result/A_dft_ODWT_speed/lsmm_nrmse.pdf")
