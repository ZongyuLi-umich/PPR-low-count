include("utils.jl")
r_time = 10
data = Vector{Any}(undef, r_time)
avg_count = 0.25
xtrue = load("../test-data-2D/4-test-images.jld2")["x2"]
N = size(xtrue, 1)
N2 = N^2
nrmse2psnr(x) = 10 * log10(1 / (x * norm(vec(xtrue))^2/N2))

data[1] = load("../result/A_mask_dft_TV/optimal_init_exp_seed0_all_2D_count="*string(avg_count)*".jld2")
data[2] = load("../result/A_mask_dft_TV/optimal_init_exp_seed1_all_2D_count="*string(avg_count)*".jld2")
data[3] = load("../result/A_mask_dft_TV/optimal_init_exp_seed2_all_2D_count="*string(avg_count)*".jld2")
data[4] = load("../result/A_mask_dft_TV/optimal_init_exp_seed3_all_2D_count="*string(avg_count)*".jld2")
data[5] = load("../result/A_mask_dft_TV/optimal_init_exp_seed4_all_2D_count="*string(avg_count)*".jld2")
data[6] = load("../result/A_mask_dft_TV/optimal_init_exp_seed5_all_2D_count="*string(avg_count)*".jld2")
data[7] = load("../result/A_mask_dft_TV/optimal_init_exp_seed6_all_2D_count="*string(avg_count)*".jld2")
data[8] = load("../result/A_mask_dft_TV/optimal_init_exp_seed7_all_2D_count="*string(avg_count)*".jld2")
data[9] = load("../result/A_mask_dft_TV/optimal_init_exp_seed8_all_2D_count="*string(avg_count)*".jld2")
data[10] = load("../result/A_mask_dft_TV/optimal_init_exp_seed9_all_2D_count="*string(avg_count)*".jld2")


time_wf_ls_notrun = mean([grab_time(data[i]["cout_wf_ls_notrun"], 1) for i = 1:r_time])
cost_wf_ls_notrun = mean([grab_cost(data[i]["cout_wf_ls_notrun"], 1) for i = 1:r_time])
nrmse_wf_ls_notrun = mean([grab_nrmse(data[i]["cout_wf_ls_notrun"], 1) for i = 1:r_time])

time_wf_optim_gau_notrun = mean([grab_time(data[i]["cout_wf_optim_gau_notrun"], 1) for i = 1:r_time])
cost_wf_optim_gau_notrun = mean([grab_cost(data[i]["cout_wf_optim_gau_notrun"], 1) for i = 1:r_time])
nrmse_wf_optim_gau_notrun = mean([grab_nrmse(data[i]["cout_wf_optim_gau_notrun"], 1) for i = 1:r_time])

time_LBFGS = mean([cal_time_lbfgs_mean(data[i]["results_LBFGS"])[1:974] for i = 1:r_time])
cost_LBFGS = mean([cal_cost_lbfgs_mean(data[i]["results_LBFGS"])[1:974] for i = 1:r_time])
nrmse_LBFGS = mean([cal_nrmse_lbfgs_mean(data[i]["results_LBFGS"], vec(xtrue))[1:974] for i = 1:r_time])

time_wf_fisher_notrun = mean([grab_time(data[i]["cout_wf_fisher_notrun"], 1) for i = 1:r_time])
cost_wf_fisher_notrun = mean([grab_cost(data[i]["cout_wf_fisher_notrun"], 1) for i = 1:r_time])
nrmse_wf_fisher_notrun = mean([grab_nrmse(data[i]["cout_wf_fisher_notrun"], 1) for i = 1:r_time])

plot(remove_gap(time_wf_ls_notrun),
            log10.(cost_wf_ls_notrun .+ 4e6),
            color = "red", legendfontsize = 12, label = "Backtracking",
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            xlabel = "Time (s)", xlims = (0, 1e2), ylabel = L"\log_{10}\left(f(x) + 4 \cdot 10^6 \right)",
            right_margin = 15Plots.mm, legend = :right,
            xlabelfontsize = 16, ylabelfontsize = 16)
plot!(remove_gap(time_wf_optim_gau_notrun),
            log10.(cost_wf_optim_gau_notrun .+ 4e6),
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "magenta", legendfontsize = 12, xlims = (0, 1e2),
            label = "Optim Gau", legend = :right)
plot!(remove_gap(time_LBFGS),
            log10.(cost_LBFGS .+ 4e6),
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "purple", xlims = (0, 1e2), legendfontsize = 12,
            label = "LBFGS", legend = :right)
plot!(remove_gap(time_wf_fisher_notrun),
            log10.(cost_wf_fisher_notrun .+ 4e6),
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "green", xlims = (0, 1e2), legendfontsize = 12,
            label = "Fisher", legend = :right)
subplot = twinx()
plot!(subplot, remove_gap(time_wf_ls_notrun),
                    nrmse2psnr.(nrmse_wf_ls_notrun),
                    markersize = 4, linealpha = 0.5, linewidth=0.5,
                    legendfontsize = 12, markershape = :rect, color = "red", label = "",
                    ylabel = "PSNR (dB)", xlims = (0, 1e2), right_margin = 15Plots.mm,
                    xlabelfontsize = 16, ylabelfontsize = 16)
plot!(subplot, remove_gap(time_wf_optim_gau_notrun),
                    nrmse2psnr.(nrmse_wf_optim_gau_notrun),
                    markersize = 4, linealpha = 0.5, linewidth=0.5,
                    legendfontsize = 12, xlims = (0, 1e2), markershape = :rect,
                    color = "magenta", label = "")
plot!(subplot, remove_gap(time_LBFGS),
                    nrmse2psnr.(nrmse_LBFGS),
                    markersize = 4, linealpha = 0.5, linewidth=0.5,
                    legendfontsize = 12, markershape = :rect, color = "purple", label = "",
                    xlims = (0, 1e2))
plot!(subplot, remove_gap(time_wf_fisher_notrun),
                    nrmse2psnr.(nrmse_wf_fisher_notrun),
                    markersize = 4, linealpha = 0.5, linewidth=0.5,
                    legendfontsize = 12, markershape = :rect, color = "green", label = "",
                    xlims = (0, 1e2))
#
savefig("../result/A_mask_dft_wf_ls_vs_fisher/wf_stepsize_optimal_init_10seeds_cost_fun_psnr_2D_count="*string(avg_count)*".pdf")


cost_wf_inplace = [Float64.(data[i]["results_WFinplace"][1][:, 1]) for i = 1:r_time]
time_wf_inplace = [Float64.(data[i]["results_WFinplace"][1][:, 2]) for i = 1:r_time]

time_admm_TV = mean([grab_time(data[i]["cout_admm_TV"], 1) for i = 1:r_time])
cost_admm_TV = mean([grab_cost(data[i]["cout_admm_TV"], 1) for i = 1:r_time])
nrmse_admm_TV = mean([grab_nrmse(data[i]["cout_admm_TV"], 1) for i = 1:r_time])

time_lsmm_TV_max = mean([grab_time(data[i]["cout_lsmm_TV_max"], 1) for i = 1:r_time])
cost_lsmm_TV_max = mean([grab_cost(data[i]["cout_lsmm_TV_max"], 1) for i = 1:r_time])
nrmse_lsmm_TV_max = mean([grab_nrmse(data[i]["cout_lsmm_TV_max"], 1) for i = 1:r_time])

time_lsmm_TV_imp = mean([grab_time(data[i]["cout_lsmm_TV_imp"], 1) for i = 1:r_time])
cost_lsmm_TV_imp = mean([grab_cost(data[i]["cout_lsmm_TV_imp"], 1) for i = 1:r_time])
nrmse_lsmm_TV_imp = mean([grab_nrmse(data[i]["cout_lsmm_TV_imp"], 1) for i = 1:r_time])

time_wf_pois_tv_lineser = mean([grab_time(data[i]["cout_wf_pois_tv_lineser"], 1) for i = 1:r_time])
cost_wf_pois_tv_lineser = mean([grab_cost(data[i]["cout_wf_pois_tv_lineser"], 1) for i = 1:r_time])
nrmse_wf_pois_tv_lineser = mean([grab_nrmse(data[i]["cout_wf_pois_tv_lineser"], 1) for i = 1:r_time])

time_LBFGS_TV = mean([cal_time_lbfgs_mean(data[i]["results_LBFGS_TV"])[1:400] for i = 1:r_time])
cost_LBFGS_TV = mean([cal_cost_lbfgs_mean(data[i]["results_LBFGS_TV"])[1:400] for i = 1:r_time])
nrmse_LBFGS_TV = mean([cal_nrmse_lbfgs_mean(data[i]["results_LBFGS_TV"], vec(xtrue))[1:400] for i = 1:r_time])

time_wf_fisher_tv = mean([grab_time(data[i]["cout_wf_pois_tv_fisher"], 1) for i = 1:r_time])
cost_wf_fisher_tv = mean([grab_cost(data[i]["cout_wf_pois_tv_fisher"], 1) for i = 1:r_time])
nrmse_wf_fisher_tv = mean([grab_nrmse(data[i]["cout_wf_pois_tv_fisher"], 1) for i = 1:r_time])


plot(remove_gap(time_admm_TV),
            log10.(cost_admm_TV .+ 4e6),
            color = "blue", legend = :right,
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            label = "ADMM TV", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
            xlabel = "Time (s)", xlims = (0, 1e2), ylabel = L"\log_{10}\left(f(x) + 4 \cdot 10^6\right)",
            right_margin = 15Plots.mm, xlabelfontsize = 16, ylabelfontsize = 16)
plot!(remove_gap(time_lsmm_TV_max),
            log10.(cost_lsmm_TV_max .+ 4e6),
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "red", legendfontsize = 12, legend = :right,
            xlims = (0, 1e2), label = "MM-max TV")
plot!(remove_gap(time_lsmm_TV_imp),
            log10.(cost_lsmm_TV_imp .+ 4e6),
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "orange", legendfontsize = 12, legend = :right,
            xlims = (0, 1e2), label = "MM-imp TV")
plot!(remove_gap(time_wf_pois_tv_lineser),
            log10.(cost_wf_pois_tv_lineser .+ 4e6),
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "magenta", legendfontsize = 12, xlims = (0, 1e2),
            legend = :right, label = "WF Poisson TV backtracking")
plot!(remove_gap(time_LBFGS_TV),
            log10.(cost_LBFGS_TV .+ 4e6), legend = :right,
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "purple", legendfontsize = 12, xlims = (0, 1e2), label = "LBFGS-TV")
# plot!(1e3 * remove_gap(grab_time(data["cout_wf_pois_tv_fisher"], r_time)),
#                         log10.(grab_cost(data["cout_wf_pois_tv_fisher"], r_time)),
#                         markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
#                         color = "green", legendfontsize = 12, xlims = (0, 8e3), label = "WF Poisson TV Fisher")

plot!(remove_gap(cal_time_wf_inplace(time_wf_inplace)),
            log10.(cal_cost_wf_inplace(cost_wf_inplace) .+ 4e6), xlims = (0, 1e2),
            markershape = :circle, markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "green", legendfontsize = 12, label = "WF Poisson TV Fisher")
subplot = twinx()
plot!(subplot, remove_gap(time_admm_TV),
            nrmse2psnr.(nrmse_admm_TV),
            legendfontsize = 12, xlims = (0, 1e2), markershape = :rect,
            markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "blue", label = "", ylabel = "PSNR (dB)",
            xlabelfontsize = 16, ylabelfontsize = 16)
plot!(subplot, remove_gap(time_lsmm_TV_max),
            nrmse2psnr.(nrmse_lsmm_TV_max),
            legendfontsize = 12, xlims = (0, 1e2), markershape = :rect,
            markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "red", label = "")
plot!(subplot, remove_gap(time_lsmm_TV_imp),
            nrmse2psnr.(nrmse_lsmm_TV_imp),
            legendfontsize = 12, xlims = (0, 1e2), markershape = :rect,
            markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "orange", label = "")
plot!(subplot, remove_gap(time_wf_pois_tv_lineser),
            nrmse2psnr.(nrmse_wf_pois_tv_lineser),
            legendfontsize = 12, xlims = (0, 1e2), markershape = :rect,
            markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "magenta", label = "")
plot!(subplot, remove_gap(time_LBFGS_TV),
            nrmse2psnr.(nrmse_LBFGS_TV),
            legendfontsize = 12, markershape = :rect, color = "purple", label = "",
            markersize = 4, linealpha = 0.5, linewidth=0.5,
            xlims = (0, 1e2))
plot!(subplot, remove_gap(cal_time_wf_inplace(time_wf_inplace)),
            nrmse2psnr.(nrmse_wf_fisher_tv),
            legendfontsize = 12, xlims = (0, 1e2), markershape = :rect,
            markersize = 4, linealpha = 0.5, linewidth=0.5,
            color = "green", label = "")
savefig("../result/A_mask_dft_TV/optimal_init_cost_fun_psnr_2D_10seeds_count="*string(avg_count)*".pdf")
