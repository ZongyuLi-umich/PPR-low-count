include("utils.jl")
seed!(0)
r_time = 1
xhow = :complex
xtrue = load("../test-data-2D/4-test-images.jld2")["x4"]
N = size(xtrue, 1)
N2 = N^2
avg_count = 0.25
niter = 100

file = matopen("../AmpSLM_16x16/A_prVAMP.mat")
A = read(file, "A")
close(file)
# idx = 1:4:size(AA, 2)
# A = AA[:, idx]
M = size(A, 1)

cpt = "/n/calumet/x/zonyul/Poisson_phase_retri_2021/"

b = 0.1 * ones(M)

(xout_wf_fisher_notrun, cout_wf_fisher_notrun,
        xout_wf_ls_notrun, cout_wf_ls_notrun,
        xout_wf_emp_notrun, cout_wf_emp_notrun,
        xout_wf_optim_gau_notrun, cout_wf_optim_gau_notrun) = [Array{Array{Any, 1}, 1}(undef, r_time) for _ = 1:8]

results_LBFGS = Vector(undef, r_time)


for r = 1:r_time

    global cons = avg_count / mean(abs2.(A * vec(xtrue)))
    global A = sqrt(cons) * A # scale matrix A
    global y_true = abs2.(A * vec(xtrue)) .+ b
    global y_pos = rand.(Poisson.(y_true))
    if xhow === :real
        global x0_rand = randn(N2)
    elseif xhow === :complex
        global x0_rand = randn(N2) + im * randn(N2)
    else
        throw("unknown xhow")
    end
    # global x0_spectral = power_iter(A'*Diagonal(y_pos-b)*A, x0_rand, 50)
    global x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)
    global α = sqrt(dot((y_pos-b), abs2.(A * x0_spectral))) / (norm(A * x0_spectral, 4)^2)
    global x0_spectral = α * x0_spectral

    global phase_shift = x -> iszero(x) ? 1 : sign(vec(xtrue)' * x)
    global nrmse = x -> (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x)))
    global phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
    global grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
    global cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)), time(), nrmse(x)]
    f(x) = sum(phi.(A * x, y_pos, b))
    g!(y, x) = copyto!(y, A' * grad_phi.(A * x, y_pos, b))

    xout_wf_ls_notrun[r], cout_wf_ls_notrun[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, sthow = :lineser, istrun = false, mustep = 0.01,
                        mushrink = 2, xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
    xout_wf_emp_notrun[r], cout_wf_emp_notrun[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, sthow = :empir, istrun = false,
                        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
    xout_wf_fisher_notrun[r], cout_wf_fisher_notrun[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, sthow = :fisher, istrun = false,
                        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
    xout_wf_optim_gau_notrun[r], cout_wf_optim_gau_notrun[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, sthow = :optim_gau, istrun = false,
                        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)

    results_LBFGS[r] = optimize(f, g!, x0_spectral, LBFGS(), Optim.Options(store_trace = true,
                                                                            show_trace = false,
                                                                            extended_trace = true))

end

save(cpt*"result/A_ETM_wf_ls_vs_fisher/wf_stepsize_2D_optimal_init_M="*string(M)*"count="*string(avg_count)*".jld2",
                "xout_wf_ls_notrun", xout_wf_ls_notrun,
                "cout_wf_ls_notrun", cout_wf_ls_notrun,
                "xout_wf_emp_notrun", xout_wf_emp_notrun,
                "cout_wf_emp_notrun", cout_wf_emp_notrun,
                "xout_wf_fisher_notrun", xout_wf_fisher_notrun,
                "cout_wf_fisher_notrun", cout_wf_fisher_notrun,
                "xout_wf_optim_gau_notrun", xout_wf_optim_gau_notrun,
                "cout_wf_optim_gau_notrun", cout_wf_optim_gau_notrun,
                "results_LBFGS", results_LBFGS)

# plot the cost function
data = load(cpt*"result/A_ETM_wf_ls_vs_fisher/wf_stepsize_2D_optimal_init_M="*string(M)*"count="*string(avg_count)*".jld2")
scatter(1e3 * remove_gap(grab_time(data["cout_wf_emp_notrun"], r_time))[1:100],
                        log10.(grab_cost(data["cout_wf_emp_notrun"], r_time))[1:100],
                        color = "blue",
                        label = "Empirical", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
                        xlabel = "Time (ms)", ylabel = L"\log(f(x))")
savefig(cpt*"result/A_ETM_wf_ls_vs_fisher/optimal_init_wf_empirical_stepsize_cost_fun_2D_num_mask="*string(num_mask)*"count="*string(avg_count)*".pdf")
scatter(1e3 * remove_gap(grab_time(data["cout_wf_ls_notrun"], r_time)),
                        log10.(grab_cost(data["cout_wf_ls_notrun"], r_time)),
                        color = "red", legendfontsize = 12, label = "Backtracking",
                        xlabel = "Time (ms)", xlims = (0, 1e3), ylabel = L"\log_{10}(f(x))",
                        right_margin = 15Plots.mm, legend = :right)
scatter!(1e3 * remove_gap(grab_time(data["cout_wf_optim_gau_notrun"], r_time)),
                         log10.(grab_cost(data["cout_wf_optim_gau_notrun"], r_time)),
                         color = "magenta", legendfontsize = 12, xlims = (0, 1e3),
                         label = "Optim Gau", legend = :right)
scatter!(1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
                         log10.(cal_cost_lbfgs_mean(data["results_LBFGS"])),
                         color = "purple", xlims = (0, 1e3), legendfontsize = 12,
                         label = "LBFGS", legend = :right)
scatter!(1e3 * remove_gap(grab_time(data["cout_wf_fisher_notrun"], r_time)),
                         log10.(grab_cost(data["cout_wf_fisher_notrun"], r_time)),
                         color = "green", xlims = (0, 1e3), legendfontsize = 12,
                         label = "Fisher", legend = :right)
subplot = twinx()
scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_wf_ls_notrun"], r_time)),
                         nrmse2psnr.(grab_nrmse(data["cout_wf_ls_notrun"], r_time)),
                         legendfontsize = 12, markershape = :rect, color = "red", label = "",
                         ylabel = "PSNR (dB)", xlims = (0, 1e3), right_margin = 15Plots.mm)
scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_wf_optim_gau_notrun"], r_time)),
                         nrmse2psnr.(grab_nrmse(data["cout_wf_optim_gau_notrun"], r_time)),
                         legendfontsize = 12, xlims = (0, 1e3), markershape = :rect,
                         color = "magenta", label = "")
scatter!(subplot, 1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
                         nrmse2psnr.(cal_nrmse_lbfgs_mean(data["results_LBFGS"], vec(xtrue))),
                         legendfontsize = 12, markershape = :rect, color = "purple", label = "",
                         xlims = (0, 1e3))
scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_wf_fisher_notrun"], r_time)),
                         nrmse2psnr.(grab_nrmse(data["cout_wf_fisher_notrun"], r_time)),
                         legendfontsize = 12, markershape = :rect, color = "green", label = "",
                         xlims = (0, 1e3))

savefig(cpt*"result/A_ETM_wf_ls_vs_fisher/optimal_init_wf_stepsize_cost_fun_psnr_2D_count="*string(avg_count)*".pdf")

# plot the NRMSE results
# idx = M
# avg_count = 0.25
# data = load(cpt*"result/A_gau_wf_ls_vs_fisher/wf_stepsize_2D_optimal_init_M="*string(idx)*"count="*string(avg_count)*".jld2")
#
# scatter(1e3 * remove_gap(grab_time(data["cout_wf_emp_notrun"], r_time)[1:80]),
#         1e2 * grab_nrmse(data["cout_wf_emp_notrun"], r_time)[1:80],
#                     color = "blue",
#                     label = "Empirical",legendfontsize = 12, xguidefontsize=15, yguidefontsize=15, xlabel = "Time (ms)", ylabel = "NRMSE (%)")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_ls_notrun"], r_time)[1:50]),
#                     1e2 * grab_nrmse(data["cout_wf_ls_notrun"], r_time)[1:50],
#                     legendfontsize = 12,
#                     color = "red", label = "Backtracking")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_optim_gau_notrun"], r_time)[1:40]),
#                     1e2 * grab_nrmse(data["cout_wf_optim_gau_notrun"], r_time)[1:40],
#                     legendfontsize = 12,
#                     color = "magenta", label = "Optim Gau")
# scatter!(1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"]))[1:17],
#                     1e2 * cal_nrmse_lbfgs_mean(data["results_LBFGS"], vec(xtrue))[1:17],
#                     legendfontsize = 12,
#                     color = "purple", label = "LBFGS")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_fisher_notrun"], r_time)[1:40]),
#                     1e2 * grab_nrmse(data["cout_wf_fisher_notrun"], r_time)[1:40],
#                     legendfontsize = 12,
#                     color = "green", label = "Fisher")
# savefig(cpt*"result/A_gau_wf_ls_vs_fisher/wf_stepsize_2D_optimal_init_M="*string(idx)*"count="*string(avg_count)*".pdf")

# plot the cost function
# idx = M
# avg_count = 0.25
# data = load(cpt*"result/A_ETM_wf_ls_vs_fisher/wf_stepsize_2D_optimal_init_M="*string(idx)*"count="*string(avg_count)*".jld2")
#
# scatter(1e3 * remove_gap(grab_time(data["cout_wf_emp_notrun"], r_time))[1:80],
#                         log10.(grab_cost(data["cout_wf_emp_notrun"], r_time))[1:80],
#                         color = "blue",
#                         label = "Empirical", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
#                         xlabel = "Time (ms)", ylabel = L"\log(f(x))")
# savefig(cpt*"result/A_ETM_wf_ls_vs_fisher/wf_empirical_stepsize_optimal_init_cost_fun_2D_M="*string(idx)*"count="*string(avg_count)*".pdf")
# scatter(1e3 * remove_gap(grab_time(data["cout_wf_ls_notrun"], r_time)),
#                         log10.(grab_cost(data["cout_wf_ls_notrun"], r_time)),
#                         color = "red", legendfontsize = 12, label = "Backtracking", xlims = (0, 3000),
#                         xlabel = "Time (ms)", ylabel = L"\log(f(x))")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_optim_gau_notrun"], r_time)),
#                         log10.(grab_cost(data["cout_wf_optim_gau_notrun"], r_time)),
#                         color = "magenta", legendfontsize = 12, label = "Optim Gau")
# scatter!(1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
#                         log10.(cal_cost_lbfgs_mean(data["results_LBFGS"])),
#                         color = "purple", legendfontsize = 12, label = "LBFGS")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_fisher_notrun"], r_time)),
#                         log10.(grab_cost(data["cout_wf_fisher_notrun"], r_time)),
#                         color = "green", legendfontsize = 12, label = "Fisher")
# #
# savefig(cpt*"result/A_ETM_wf_ls_vs_fisher/wf_stepsize_optimal_init_cost_fun_2D_M="*string(idx)*"count="*string(avg_count)*".pdf")
