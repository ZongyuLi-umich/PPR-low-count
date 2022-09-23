include("utils.jl")

seed!(0)
r_time = 10
xhow = :real
avg_count = 0.25 # 0.25, 1, 2
niter = 100

xtrue = load("../test-data-2D/4-test-images.jld2")["x3"]
ref = load("../test-data-2D/4-test-images.jld2")["ref_x3"]
nrmse2psnr(x) = 10 * log10(1 / (x * norm(vec(xtrue))^2/N2))
N = size(xtrue, 1)
N2 = N^2

phase_shift = x -> iszero(x) ? 1 : sign(vec(xtrue)' * x)
nrmse = x -> (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x)))

K = 4 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]

β = 32
α = 0.1

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

        global A = LinearMapAA(
                    x -> vec(fft(pad(reshape(x, N, N)))),
                    y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
                    (K*K, N*N), (name="fft2D",), T=ComplexF32)

        global cons = avg_count / mean(abs2.(A * vec(xtrue)))
        global A = sqrt(cons) * A # scale matrix A

        global b = 0.1 * ones(size(A, 1))
        global y_true = abs2.(A * vec(xtrue + ref)) .+ b
        global y_pos = rand.(Poisson.(y_true))
        # Define x0
        global x0_rand = randn(N2) + im * randn(N2)
        global x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)

        global scale_factor = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
        global x0_spectral = abs.(scale_factor * x0_spectral)

        phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
        grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
        cost_func_unreg = (x,iter) -> [sum(phi.(A * (x+vec(ref)), y_pos, b)), time(), nrmse(x)]
        f_unreg(x) = sum(phi.(A * (x+vec(ref)), y_pos, b))
        g_unreg!(y, x) = copyto!(y, real(A' * grad_phi.(A * (x+vec(ref)), y_pos, b)))

        T = LinearMapAA(x -> diff2d_forw(x, N, N), y -> diff2d_adj(y, N, N), (2*N*(N-1), N2); T=Float64)
        huber(t, α) = abs(t) < α ? 0.5 * abs2(t) : α * abs(t) - 0.5 * α^2
        grad_huber(t, α) = abs(t) < α ? t : α * sign(t)
        cost_func_reg = (x,iter) -> [sum(phi.(A * (x+vec(ref)), y_pos, b)) + β * sum(huber.(T * x, α)), time(), nrmse(x)]

        f_reg(x) = sum(phi.(A * (x+vec(ref)), y_pos, b)) + β * sum(huber.(T * x, α))
        g_reg!(y, x) = copyto!(y, real(A' * grad_phi.(A * (x+vec(ref)), y_pos, b) + β * T' * grad_huber.(T * x, α)))
        grad_reg!(y, Ax, Tx) = copyto!(y, real(A' * grad_phi.(Ax, y_pos, b) + β * T' * grad_huber.(Tx, α)))

        # unregularized algorithms
        xout_wf_ls_notrun[r], cout_wf_ls_notrun[r] = Wirtinger_flow(A,y_pos,b;
                            gradhow = :poisson, sthow = :lineser, istrun = false, mustep = 0.01,
                            mushrink = 2, xhow = xhow, x0 = x0_spectral, niter = niter,
                            ref = ref, fun = cost_func_unreg)
        xout_wf_emp_notrun[r], cout_wf_emp_notrun[r] = Wirtinger_flow(A,y_pos,b;
                            gradhow = :poisson, sthow = :empir, istrun = false,
                            xhow = xhow, x0 = x0_spectral, niter = niter,
                            ref = ref, fun = cost_func_unreg)
        xout_wf_fisher_notrun[r], cout_wf_fisher_notrun[r] = Wirtinger_flow(A,y_pos,b;
                            gradhow = :poisson, sthow = :fisher, istrun = false,
                            xhow = xhow, x0 = x0_spectral, niter = niter,
                            ref = ref, fun = cost_func_unreg)

        xout_wf_optim_gau_notrun[r], cout_wf_optim_gau_notrun[r] = Wirtinger_flow(A,y_pos,b;
                            gradhow = :poisson, sthow = :optim_gau, istrun = false,
                            xhow = xhow, x0 = x0_spectral, niter = niter,
                            ref = ref, fun = cost_func_unreg)

        results_LBFGS[r] = optimize(f_unreg, g_unreg!, x0_spectral, LBFGS(), Optim.Options(store_trace = true,
                                                                                show_trace = false,
                                                                                extended_trace = true))

        # regularized algorithms
        state = WFState_TV(x0_spectral, grad_reg!, A; β = β, α = α, R = T, ref = ref, niter = niter, xhow = xhow)
        Wirtinger_flow_inplace_reg!(A, b, niter, state; R = T)
        state.cost_vec[1] = f_reg(x0_spectral)
        for i = 2:niter+1
            state.cost_vec[i] = f_reg(state.x_hist[i-1, :])
        end
        results_WFinplace[r] = [state.cost_vec state.timer]

        xout_wf_pois_tv_fisher[r], cout_wf_pois_tv_fisher[r] = Wirtinger_flow_huber(A,y_pos,b;
                x0 = x0_spectral, niter = niter, xhow = xhow, sthow = :fisher,
                reg1 = β, reg2 = α, ref = ref, fun = cost_func_reg)
        xout_wf_pois_tv_lineser[r], cout_wf_pois_tv_lineser[r] = Wirtinger_flow_huber(A,y_pos,b;
                x0 = x0_spectral, niter = niter, xhow = xhow, sthow = :lineser,
                reg1 = β, reg2 = α, ref = ref, fun = cost_func_reg)
        xout_lsmm_TV_max[r], cout_lsmm_TV_max[r] = LSMM_TV_huber(A,y_pos,b;
                curvhow = :max, xhow = xhow, reg1 = β, reg2 = α,
                x0 = x0_spectral, ref = ref, niter = niter, fun = cost_func_reg)
        xout_lsmm_TV_imp[r], cout_lsmm_TV_imp[r] = LSMM_TV_huber(A,y_pos,b;
                curvhow = :imp, xhow = xhow, reg1 = β, reg2 = α,
                x0 = x0_spectral, ref = ref, niter = niter, fun = cost_func_reg)
        xout_admm_TV[r], cout_admm_TV[r] = ADMM_TV_huber(A,y_pos,b;
                phow =:adaptive, reg1 = β, reg2 = α, ref = ref,
                xhow = xhow, updatehow =:cg, x0 = x0_spectral,
                ρ = 32, niter = niter, fun = cost_func_reg)
        results_LBFGS_TV[r] = optimize(f_reg, g_reg!, x0_spectral, LBFGS(), Optim.Options(store_trace = true,
                                                                                show_trace = false,
                                                                                extended_trace = true))
end

save("../result/A_canon_dft_TV/optimal_init_exp_10times_all_2D_count="*string(avg_count)*".jld2",
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
data = load("../result/A_canon_dft_TV/optimal_init_exp_10times_all_2D_count="*string(avg_count)*".jld2")
# unregularized algorithms
plot(remove_gap(grab_time(data["cout_wf_ls_notrun"], r_time)),
        log10.(grab_cost(data["cout_wf_ls_notrun"], r_time) .+ 6e6),
        color = "red", legendfontsize = 12, label = "Backtracking",
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        xlabel = "Time (s)", xlims = (0, 10), ylabel = L"\log_{10}\left(f(x)+6 \cdot 10^6 \right)",
        right_margin = 15Plots.mm, legend = :right,
        xlabelfontsize = 16, ylabelfontsize = 16)
plot!(remove_gap(grab_time(data["cout_wf_optim_gau_notrun"], r_time)),
        log10.(grab_cost(data["cout_wf_optim_gau_notrun"], r_time) .+ 6e6),
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "magenta", legendfontsize = 12, xlims = (0, 10),
        label = "Optim Gau", legend = :right)
plot!(remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
        log10.(cal_cost_lbfgs_mean(data["results_LBFGS"]) .+ 6e6),
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "purple", xlims = (0, 10), legendfontsize = 12,
        label = "LBFGS", legend = :right)
plot!(remove_gap(grab_time(data["cout_wf_fisher_notrun"], r_time)),
        log10.(grab_cost(data["cout_wf_fisher_notrun"], r_time) .+ 6e6),
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "green", xlims = (0, 10), legendfontsize = 12,
        label = "Fisher", legend = :right)
subplot = twinx()
plot!(subplot, remove_gap(grab_time(data["cout_wf_ls_notrun"], r_time)),
        nrmse2psnr.(grab_nrmse(data["cout_wf_ls_notrun"], r_time)),
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        legendfontsize = 12, markershape = :rect, color = "red", label = "",
        ylabel = "PSNR (dB)", xlims = (0, 10), right_margin = 15Plots.mm,
        xlabelfontsize = 16, ylabelfontsize = 16)
plot!(subplot, remove_gap(grab_time(data["cout_wf_optim_gau_notrun"], r_time)),
        nrmse2psnr.(grab_nrmse(data["cout_wf_optim_gau_notrun"], r_time)),
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        legendfontsize = 12, xlims = (0, 10), markershape = :rect,
        color = "magenta", label = "")
plot!(subplot, remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
        nrmse2psnr.(cal_nrmse_lbfgs_mean(data["results_LBFGS"], vec(xtrue))),
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        legendfontsize = 12, markershape = :rect, color = "purple", label = "",
        xlims = (0, 10))
plot!(subplot, remove_gap(grab_time(data["cout_wf_fisher_notrun"], r_time)),
        nrmse2psnr.(grab_nrmse(data["cout_wf_fisher_notrun"], r_time)),
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        legendfontsize = 12, markershape = :rect, color = "green", label = "",
        xlims = (0, 10))
savefig("../result/A_canon_dft_wf_ls_vs_fisher/wf_stepsize_optimal_init_10times_cost_fun_psnr_2D_count="*string(avg_count)*".pdf")


cost_wf_inplace = [Float64.(data["results_WFinplace"][i][:, 1]) for i = 1:length(data["results_WFinplace"])]
time_wf_inplace = [Float64.(data["results_WFinplace"][i][:, 2]) for i = 1:length(data["results_WFinplace"])]
plot(remove_gap(grab_time(data["cout_admm_TV"], r_time)),
        log10.(grab_cost(data["cout_admm_TV"], r_time) .+ 6e6),
        color = "blue", legend = :right,
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        label = "ADMM TV", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
        xlabel = "Time (s)", xlims = (0, 10), ylabel = L"\log_{10}\left(f(x)+6 \cdot 10^6 \right)",
        right_margin = 15Plots.mm, xlabelfontsize = 16, ylabelfontsize = 16)
plot!(remove_gap(grab_time(data["cout_lsmm_TV_max"], r_time)),
        log10.(grab_cost(data["cout_lsmm_TV_max"], r_time) .+ 6e6),
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "red", legendfontsize = 12, legend = :right,
        xlims = (0, 10), label = "MM-max TV")
plot!(remove_gap(grab_time(data["cout_lsmm_TV_imp"], r_time)),
        log10.(grab_cost(data["cout_lsmm_TV_imp"], r_time) .+ 6e6),
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "orange", legendfontsize = 12, legend = :right,
        xlims = (0, 10), label = "MM-imp TV")
plot!(remove_gap(grab_time(data["cout_wf_pois_tv_lineser"], r_time)),
        log10.(grab_cost(data["cout_wf_pois_tv_lineser"], r_time) .+ 6e6),
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "magenta", legendfontsize = 12, xlims = (0, 10),
        legend = :right, label = "WF Poisson TV backtracking")
plot!(remove_gap(cal_time_lbfgs_mean(data["results_LBFGS_TV"])),
        log10.(cal_cost_lbfgs_mean(data["results_LBFGS_TV"]) .+ 6e6), legend = :right,
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "purple", legendfontsize = 12, xlims = (0, 10), label = "LBFGS-TV")
# plot!(1e3 * remove_gap(grab_time(data["cout_wf_pois_tv_fisher"], r_time)),
#                         log10.(grab_cost(data["cout_wf_pois_tv_fisher"], r_time)),
#                         markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
#                         color = "green", legendfontsize = 12, xlims = (0, 8e3), label = "WF Poisson TV Fisher")

plot!(remove_gap(cal_time_wf_inplace(time_wf_inplace)),
        log10.(cal_cost_wf_inplace(cost_wf_inplace) .+ 6e6), xlims = (0, 10),
        markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "green", legendfontsize = 12, label = "WF Poisson TV Fisher")
subplot = twinx()
plot!(subplot, remove_gap(grab_time(data["cout_admm_TV"], r_time)),
        nrmse2psnr.(grab_nrmse(data["cout_admm_TV"], r_time)),
        legendfontsize = 12, xlims = (0, 10), markershape = :rect,
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "blue", label = "", ylabel = "PSNR (dB)",
        xlabelfontsize = 16, ylabelfontsize = 16)
plot!(subplot, remove_gap(grab_time(data["cout_lsmm_TV_max"], r_time)),
        nrmse2psnr.(grab_nrmse(data["cout_lsmm_TV_max"], r_time)),
        legendfontsize = 12, xlims = (0, 10), markershape = :rect,
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "red", label = "")
plot!(subplot, remove_gap(grab_time(data["cout_lsmm_TV_imp"], r_time)),
        nrmse2psnr.(grab_nrmse(data["cout_lsmm_TV_imp"], r_time)),
        legendfontsize = 12, xlims = (0, 10), markershape = :rect,
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "orange", label = "")
plot!(subplot, remove_gap(grab_time(data["cout_wf_pois_tv_lineser"], r_time)),
        nrmse2psnr.(grab_nrmse(data["cout_wf_pois_tv_lineser"], r_time)),
        legendfontsize = 12, xlims = (0, 10), markershape = :rect,
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "magenta", label = "")
plot!(subplot, remove_gap(cal_time_lbfgs_mean(data["results_LBFGS_TV"])),
        nrmse2psnr.(cal_nrmse_lbfgs_mean(data["results_LBFGS_TV"], vec(xtrue))),
        legendfontsize = 12, markershape = :rect, color = "purple", label = "",
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        xlims = (0, 10))
plot!(subplot, remove_gap(cal_time_wf_inplace(time_wf_inplace)),
        nrmse2psnr.(grab_nrmse(data["cout_wf_pois_tv_fisher"], r_time)),
        legendfontsize = 12, xlims = (0, 10), markershape = :rect,
        markersize = 4, linealpha = 0.5, linewidth=0.5,
        color = "green", label = "")
savefig("../result/A_canon_dft_TV/optimal_init_cost_fun_2D_10times_count="*string(avg_count)*".pdf")
