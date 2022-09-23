include("utils.jl")

seed!(5)
r_time = 1
xhow = :real
avg_count = 0.25 # 0.25, 1, 2
niter = 50

xtrue = load("../test-data-2D/4-test-images.jld2")["x2"]
nrmse2psnr(x) = 10 * log10(1 / (x * norm(vec(xtrue))^2/N2))
N = size(xtrue, 1)
N2 = N^2

phase_shift = x -> iszero(x) ? 1 : sign(vec(xtrue)' * x)
nrmse = x -> (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x)))

num_mask = 20
K = 2 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]

β = 32
α = 0.1
cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)) + β * sum(huber.(T * x, α)), time(), nrmse(x)]


# unregularized algorithms
(xout_wf_fisher_notrun, cout_wf_fisher_notrun,
        xout_wf_ls_notrun, cout_wf_ls_notrun,
        xout_wf_emp_notrun, cout_wf_emp_notrun,
        xout_wf_optim_gau_notrun, cout_wf_optim_gau_notrun) = [Array{Array{Any, 1}, 1}(undef, r_time) for _ = 1:8]

results_LBFGS = Vector(undef, r_time)

# regularized algorithms
(xout_wf_pois_tv_fisher, cout_wf_pois_tv_fisher,
        xout_wf_pois_tv_lineser, cout_wf_pois_tv_lineser,
        xout_lsmm_TV_max, cout_lsmm_TV_max,
        xout_lsmm_TV_imp, cout_lsmm_TV_imp,
        xout_admm_TV, cout_admm_TV) = [Array{Array{Any, 1}, 1}(undef, r_time) for _ = 1:10]
results_LBFGS_TV = Vector(undef, r_time)
results_WFinplace = Vector(undef, r_time)


for r = 1:r_time

        global A1 = LinearMapAA(
                    x -> vec(fft(pad(reshape(x, N, N)))),
                    y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
                    (K*K, N*N), (name="fft2D",), T=ComplexF32)
        global A = A1
        mask_list = Array{AbstractArray{<:Number}, 1}(undef, num_mask)

        for i = 1:num_mask
            mask = rand(N) .< 0.5
            A_temp = LinearMapAA(
                    x -> vec(fft(pad(reshape(x, N, N) .* mask))),
                    y -> (K*K) * vec(unpad(ifft(reshape(y, K, K))) .* mask),
                    (K*K, N*N), (name="fft2D",), T=ComplexF32)
            global A = vcat(A, A_temp)
            global mask_list[i] = vec(mask)
        end

        global cons = avg_count / mean(abs2.(A * vec(xtrue)))
        global A = sqrt(cons) * A # scale matrix A

        global b = 0.1 * ones(size(A, 1))
        global y_true = abs2.(A * vec(xtrue)) .+ b
        global y_pos = rand.(Poisson.(y_true))
        # Define x0
        global x0_rand = randn(N2) + im * randn(N2)
        global x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)

        global scale_factor = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
        global x0_spectral = abs.(scale_factor * x0_spectral)

        phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
        grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
        cost_func_unreg = (x,iter) -> [sum(phi.(A * x, y_pos, b)), time(), nrmse(x)]
        f_unreg(x) = sum(phi.(A * x, y_pos, b))
        g_unreg!(y, x) = copyto!(y, real(A' * grad_phi.(A * x, y_pos, b)))

        T = LinearMapAA(x -> diff2d_forw(x, N, N), y -> diff2d_adj(y, N, N), (2*N*(N-1), N2); T=Float64)
        huber(t, α) = abs(t) < α ? 0.5 * abs2(t) : α * abs(t) - 0.5 * α^2
        grad_huber(t, α) = abs(t) < α ? t : α * sign(t)
        cost_func_reg = (x,iter) -> [sum(phi.(A * x, y_pos, b)) + β * sum(huber.(T * x, α)), time(), nrmse(x)]

        f_reg(x) = sum(phi.(A * x, y_pos, b)) + β * sum(huber.(T * x, α))
        g_reg!(y, x) = copyto!(y, real(A' * grad_phi.(A * x, y_pos, b) + β * T' * grad_huber.(T * x, α)))
        grad_reg!(y, Ax, Tx) = copyto!(y, real(A' * grad_phi.(Ax, y_pos, b) + β * T' * grad_huber.(Tx, α)))

        # unregularized algorithms
        xout_wf_optim_gau_notrun[r], cout_wf_optim_gau_notrun[r] = Wirtinger_flow(A,y_pos,b;
                            gradhow = :poisson, sthow = :optim_gau, istrun = false,
                            xhow = xhow, x0 = x0_spectral, niter = niter, trunreg = 1000, fun = cost_func_unreg)

        xout_wf_ls_notrun[r], cout_wf_ls_notrun[r] = Wirtinger_flow(A,y_pos,b;
                            gradhow = :poisson, sthow = :lineser, istrun = false, mustep = 0.01,
                            mushrink = 2, xhow = xhow, x0 = x0_spectral, trunreg = 1000, niter = niter, fun = cost_func_unreg)
        xout_wf_emp_notrun[r], cout_wf_emp_notrun[r] = Wirtinger_flow(A,y_pos,b;
                            gradhow = :poisson, sthow = :empir, istrun = false,
                            xhow = xhow, x0 = x0_spectral, niter = niter, trunreg = 1000, fun = cost_func_unreg)
        xout_wf_fisher_notrun[r], cout_wf_fisher_notrun[r] = Wirtinger_flow(A,y_pos,b;
                            gradhow = :poisson, sthow = :fisher, istrun = false,
                            xhow = xhow, x0 = x0_spectral, niter = niter, trunreg = 1000, fun = cost_func_unreg)

        results_LBFGS[r] = optimize(f_unreg, g_unreg!, x0_spectral, LBFGS(), Optim.Options(store_trace = true,
                                                                                show_trace = false,
                                                                                extended_trace = true))

        # regularized algorithms
        state = WFState_TV(x0_spectral, grad_reg!, A; β = β, α = α, R = T, niter = niter, xhow = xhow)
        Wirtinger_flow_inplace_reg!(A, b, niter, state; R = T)
        state.cost_vec[1] = f_reg(x0_spectral)
        for i = 2:niter+1
            state.cost_vec[i] = f_reg(state.x_hist[i-1, :])
        end
        results_WFinplace[r] = [state.cost_vec state.timer]


        xout_wf_pois_tv_fisher[r], cout_wf_pois_tv_fisher[r] = Wirtinger_flow_huber(A,y_pos,b;
                            x0 = x0_spectral, niter = niter, xhow = xhow, sthow = :fisher,
                            reg1 = β, reg2 = α, fun = cost_func_reg)
        xout_wf_pois_tv_lineser[r], cout_wf_pois_tv_lineser[r] = Wirtinger_flow_huber(A,y_pos,b;
                            x0 = x0_spectral, niter = niter, xhow = xhow, sthow = :lineser,
                            reg1 = β, reg2 = α, fun = cost_func_reg)
        xout_lsmm_TV_max[r], cout_lsmm_TV_max[r] = LSMM_TV_huber(A,y_pos,b;
                            curvhow = :max, xhow = xhow, reg1 = β, reg2 = α,
                            x0 = x0_spectral, niter = niter, fun = cost_func_reg)
        xout_lsmm_TV_imp[r], cout_lsmm_TV_imp[r] = LSMM_TV_huber(A,y_pos,b;
                            curvhow = :imp, xhow = xhow, reg1 = β, reg2 = α,
                            x0 = x0_spectral, niter = niter, fun = cost_func_reg)
        xout_admm_TV[r], cout_admm_TV[r] = ADMM_TV_huber(A,y_pos,b;
                            phow =:adaptive, reg1 = β, reg2 = α,
                            xhow = xhow, updatehow =:cg, x0 = x0_spectral,
                            ρ = 32, niter = niter, fun = cost_func_reg)
        results_LBFGS_TV[r] = optimize(f_reg, g_reg!, x0_spectral, LBFGS(), Optim.Options(store_trace = true,
                                                                                show_trace = false,
                                                                                extended_trace = true))
end

save("../result/A_mask_dft_TV/optimal_init_exp_seed5_all_2D_count="*string(avg_count)*".jld2",
                "xout_wf_ls_notrun", xout_wf_ls_notrun,
                "cout_wf_ls_notrun", cout_wf_ls_notrun,
                "xout_wf_emp_notrun", xout_wf_emp_notrun,
                "cout_wf_emp_notrun", cout_wf_emp_notrun,
                "xout_wf_fisher_notrun", xout_wf_fisher_notrun,
                "cout_wf_fisher_notrun", cout_wf_fisher_notrun,
                "xout_wf_optim_gau_notrun", xout_wf_optim_gau_notrun,
                "cout_wf_optim_gau_notrun", cout_wf_optim_gau_notrun,
                "results_LBFGS", results_LBFGS,
                "xout_wf_pois_tv_fisher", xout_wf_pois_tv_fisher,
                "cout_wf_pois_tv_fisher", cout_wf_pois_tv_fisher,
                "xout_wf_pois_tv_lineser", xout_wf_pois_tv_lineser,
                "cout_wf_pois_tv_lineser", cout_wf_pois_tv_lineser,
                "xout_lsmm_TV_max", xout_lsmm_TV_max,
                "cout_lsmm_TV_max", cout_lsmm_TV_max,
                "xout_lsmm_TV_imp", xout_lsmm_TV_imp,
                "cout_lsmm_TV_imp", cout_lsmm_TV_imp,
                "xout_admm_TV", xout_admm_TV,
                "cout_admm_TV", cout_admm_TV,
                "results_LBFGS_TV", results_LBFGS_TV,
                "results_WFinplace", results_WFinplace)

# plot data
# data = load("../result/A_mask_dft_TV/optimal_init_num_mask="*string(num_mask)*".jld2")
# cost_wf_inplace = [Float64.(data["results_WFinplace"][i][:, 1]) for i = 1:length(data["results_WFinplace"])]
# time_wf_inplace = [Float64.(data["results_WFinplace"][i][:, 2]) for i = 1:length(data["results_WFinplace"])]
# scatter(1e3 * remove_gap(grab_time(data["cout_admm_TV"], r_time)),
#                         log10.(grab_cost(data["cout_admm_TV"], r_time) .+ 4e6),
#                         color = "blue", legend = :right,
#                         label = "ADMM TV", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
#                         xlabel = "Time (ms)", xlims = (0, 1e5), ylabel = L"\log_{10}(f(x)+4e6)",
#                         right_margin = 15Plots.mm)
# scatter!(1e3 * remove_gap(grab_time(data["cout_lsmm_TV_max"], r_time)),
#                         log10.(grab_cost(data["cout_lsmm_TV_max"], r_time) .+ 4e6),
#                         color = "red", legendfontsize = 12, legend = :right,
#                         xlims = (0, 1e5), label = "MM-max TV")
# scatter!(1e3 * remove_gap(grab_time(data["cout_lsmm_TV_imp"], r_time)),
#                         log10.(grab_cost(data["cout_lsmm_TV_imp"], r_time) .+ 4e6),
#                         color = "orange", legendfontsize = 12, legend = :right,
#                         xlims = (0, 1e5), label = "MM-imp TV")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_pois_tv_lineser"], r_time)),
#                         log10.(grab_cost(data["cout_wf_pois_tv_lineser"], r_time) .+ 4e6),
#                         color = "magenta", legendfontsize = 12, xlims = (0, 1e5),
#                         legend = :right, label = "WF Poisson TV backtracking")
# scatter!(1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
#                         log10.(cal_cost_lbfgs_mean(data["results_LBFGS"]) .+ 4e6), legend = :right,
#                         color = "purple", legendfontsize = 12, xlims = (0, 1e5), label = "LBFGS-TV")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_pois_tv_fisher"], r_time)),
#                         log10.(grab_cost(data["cout_wf_pois_tv_fisher"], r_time) .+ 2e6),
#                          color = "green", legendfontsize = 12, xlims = (0, 1e4), label = "WF Poisson TV Fisher")

# scatter!(1e3 * remove_gap(cal_time_wf_inplace(time_wf_inplace)),
#                         log10.(cal_cost_wf_inplace(cost_wf_inplace) .+ 4e6), xlims = (0, 1e5),
#                         color = "green", legendfontsize = 12, label = "WF Poisson TV Fisher")
# subplot = twinx()
# scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_admm_TV"], r_time)),
#          nrmse2psnr.(grab_nrmse(data["cout_admm_TV"], r_time)),
#          legendfontsize = 12, xlims = (0, 1e5), markershape = :rect,
#          color = "blue", label = "", ylabel = "PSNR (dB)")
# scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_lsmm_TV_max"], r_time)),
#          nrmse2psnr.(grab_nrmse(data["cout_lsmm_TV_max"], r_time)),
#          legendfontsize = 12, xlims = (0, 1e5), markershape = :rect,
#          color = "red", label = "")
# scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_lsmm_TV_imp"], r_time)),
#          nrmse2psnr.(grab_nrmse(data["cout_lsmm_TV_imp"], r_time)),
#          legendfontsize = 12, xlims = (0, 1e5), markershape = :rect,
#          color = "orange", label = "")
# scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_wf_pois_tv_lineser"], r_time)),
#          nrmse2psnr.(grab_nrmse(data["cout_wf_pois_tv_lineser"], r_time)),
#          legendfontsize = 12, xlims = (0, 1e5), markershape = :rect,
#          color = "magenta", label = "")
# scatter!(subplot, 1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
#          nrmse2psnr.(cal_nrmse_lbfgs_mean(data["results_LBFGS"], vec(xtrue))),
#          legendfontsize = 12, markershape = :rect, color = "purple", label = "",
#          xlims = (0, 1e5))
# scatter!(subplot, 1e3 * remove_gap(cal_time_wf_inplace(time_wf_inplace)),
#          nrmse2psnr.(grab_nrmse(data["cout_wf_pois_tv_fisher"], r_time)),
#          legendfontsize = 12, xlims = (0, 1e5), markershape = :rect,
#          color = "green", label = "")
# savefig("../result/A_mask_dft_TV/optimal_init_cost_fun_2D_count="*string(avg_count)*".pdf")

# data = load("./result/A_fft_huber_TV/A_fft_huber_2D_optimal_init_num_mask="*string(num_mask)*".jld2")
# cost_wf_inplace = [Float64.(data["results_WFinplace"][i][:, 1]) for i = 1:length(data["results_WFinplace"])]
# time_wf_inplace = [Float64.(data["results_WFinplace"][i][:, 2]) for i = 1:length(data["results_WFinplace"])]
# scatter(1e3 * remove_gap(grab_time(data["cout_admm_TV_huber"], r_time)),
#                         log10.(grab_cost(data["cout_admm_TV_huber"], r_time)),
#                         color = "blue",
#                         label = "ADMM TV", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
#                         xlabel = "Time (ms)", ylabel = L"\log(f(x))", xlims = (0, 2e4))
# scatter!(1e3 * remove_gap(grab_time(data["cout_lsmm_TV_huber"], r_time)),
#                         log10.(grab_cost(data["cout_lsmm_TV_huber"], r_time)),
#                         color = "red", legendfontsize = 12, label = "MM TV",xlims = (0, 2e4))
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_pois_huber_lineser"], r_time)),
#                         log10.(grab_cost(data["cout_wf_pois_huber_lineser"], r_time)),xlims = (0, 2e4),
#                         color = "magenta", legendfontsize = 12, label = "WF Poisson TV backtracking")
# # scatter!(1e3 * remove_gap(grab_time(data["cout_wf_pois_huber_fisher"], r_time))[2:400],
# #                         log10.(grab_cost(data["cout_wf_pois_huber_fisher"], r_time))[2:400],
# #                         color = "magenta", legendfontsize = 12, label = "WF Poisson TV Fisher")
# scatter!(1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
#                         log10.(cal_cost_lbfgs_mean(data["results_LBFGS"])),xlims = (0, 2e4),
#                         color = "purple", legendfontsize = 12, label = "LBFGS")
# scatter!(1e3 * remove_gap(cal_time_wf_inplace(time_wf_inplace)),
#                         log10.(cal_cost_wf_inplace(cost_wf_inplace)),xlims = (0, 2e4),
#                         color = "green", legendfontsize = 12, label = "WF Poisson TV Fisher")
# savefig("./result/A_fft_huber_TV/cost_fun_new_2D_num_mask="*string(num_mask)*"count="*string(avg_count)*".pdf")

# NRMSE
# data = load("./result/A_fft_huber_TV/A_fft_huber_optimal_init_num_mask="*string(num_mask)*".jld2")
# scatter(1e3 * remove_gap(grab_time(data["cout_admm_TV_huber"], r_time))[2:150],
#                         1e2 * (grab_nrmse(data["cout_admm_TV_huber"], r_time))[2:150],
#                         color = "blue",
#                         label = "ADMM TV", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
#                         xlabel = "Time (ms)", ylabel = "NRMSE (%)")
# scatter!(1e3 * remove_gap(grab_time(data["cout_lsmm_TV_huber"], r_time))[2:50],
#                         1e2 * (grab_nrmse(data["cout_lsmm_TV_huber"], r_time))[2:50],
#                         color = "red", legendfontsize = 12, label = "LSMM TV")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_pois_huber"], r_time))[2:400],
#                         1e2 * (grab_nrmse(data["cout_wf_pois_huber"], r_time))[2:400],
#                         color = "green", legendfontsize = 12, label = "WF Poisson TV")
#
# savefig(cpt*"result/A_fft_huber_TV/nrmse_M="*string(M)*"count="*string(avg_count)*".pdf")
