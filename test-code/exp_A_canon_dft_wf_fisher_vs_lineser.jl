include("utils.jl")
seed!(0)
# N = 500
r_time = 1
xhow = :real
avg_count = 0
niter = 100
# M = 80000
cpt = "/n/calumet/x/zonyul/Poisson_phase_retri_2021/"
# xtrue = load("./test-data-2D/UM-1817-shepplogan.jld2")["x1"]
xtrue = load("../test-data-2D/4-test-images.jld2")["x3"]
nrmse2psnr(x) = 10 * log10(1 / (x * norm(vec(xtrue))^2/N2))
ref = load("../test-data-2D/4-test-images.jld2")["ref_x3"]
N = size(xtrue, 1)
N2 = N^2

K = 4 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]

(xout_wf_fisher_notrun, cout_wf_fisher_notrun,
        xout_wf_ls_notrun, cout_wf_ls_notrun,
        xout_wf_emp_notrun, cout_wf_emp_notrun,
        xout_wf_optim_gau_notrun, cout_wf_optim_gau_notrun) = [Array{Array{Any, 1}, 1}(undef, r_time) for _ = 1:8]

results_LBFGS = Vector(undef, r_time)


for r = 1:r_time
    global A = LinearMapAA(
            x -> vec(fft(pad(reshape(x, N, N)))),
            y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
            (K*K, N*N), (name="fft2D",), T=ComplexF32)
    if avg_count > 0
        global cons = avg_count / mean(abs2.(A * vec(xtrue)))
        global A = sqrt(cons) * A # scale matrix A
    end
    global b = 0.1 * ones(size(A, 1))
    global y_true = abs2.(A * vec(xtrue + ref)) + b # xtrue + ref
    global y_pos = rand.(Poisson.(y_true))
    global x0_rand = vec(rand(N, N))
    global x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)
    global α = sqrt(dot((y_pos-b), abs2.(A * x0_spectral))) / (norm(A * x0_spectral, 4)^2)
    global x0_spectral = abs.(α * x0_spectral)

    global phase_shift = x -> iszero(x) ? 1 : sign(vec(xtrue)' * x)
    global nrmse = x -> (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x)))
    global phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
    global grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
    global cost_func = (x,iter) -> [sum(phi.(A * (x + vec(ref)), y_pos, b)), time(), nrmse(x)]
    f(x) = sum(phi.(A * (x + vec(ref)), y_pos, b))
    g!(y, x) = copyto!(y, real(A' * grad_phi.(A * (x + vec(ref)), y_pos, b)))

    xout_wf_ls_notrun[r], cout_wf_ls_notrun[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, sthow = :lineser, istrun = false, mustep = 0.01,
                        mushrink = 2, xhow = xhow, x0 = x0_spectral, niter = niter,
                        ref = ref, fun = cost_func)
    xout_wf_emp_notrun[r], cout_wf_emp_notrun[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, sthow = :empir, istrun = false,
                        xhow = xhow, x0 = x0_spectral, niter = niter,
                        ref = ref, fun = cost_func)
    xout_wf_fisher_notrun[r], cout_wf_fisher_notrun[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, sthow = :fisher, istrun = false,
                        xhow = xhow, x0 = x0_spectral, niter = niter,
                        ref = ref, fun = cost_func)

    xout_wf_optim_gau_notrun[r], cout_wf_optim_gau_notrun[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, sthow = :optim_gau, istrun = false,
                        xhow = xhow, x0 = x0_spectral, niter = niter,
                        ref = ref, fun = cost_func)

    results_LBFGS[r] = optimize(f, g!, x0_spectral, LBFGS(), Optim.Options(store_trace = true,
                                                                            show_trace = false,
                                                                            extended_trace = true))

end

save(cpt*"result/A_canon_dft_wf_ls_vs_fisher/optimal_init_wf_2D_wref_stepsize_count="*string(avg_count)*".jld2",
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
data = load(cpt*"result/A_canon_dft_wf_ls_vs_fisher/optimal_init_wf_2D_wref_stepsize_count="*string(avg_count)*".jld2")
x = Float32.(reshape(data["xout_wf_fisher_notrun"][1], N, N))
jim(x, clim = (0,1))
scatter(1e3 * remove_gap(grab_time(data["cout_wf_emp_notrun"], r_time))[1:100],
                        log10.(grab_cost(data["cout_wf_emp_notrun"], r_time))[1:100],
                        color = "blue",
                        label = "Empirical", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
                        xlabel = "Time (ms)", ylabel = L"\log(f(x))")
savefig(cpt*"result/A_canon_dft_wf_ls_vs_fisher/optimal_init_wf_empirical_stepsize_cost_fun_2D_count="*string(avg_count)*".pdf")
cons = 1e13
scatter(1e3 * remove_gap(grab_time(data["cout_wf_ls_notrun"], r_time)),
        log10.(grab_cost(data["cout_wf_ls_notrun"], r_time) .+ cons),
        color = "red", legendfontsize = 12, label = "Backtracking",
        xlabel = "Time (ms)", xlims = (0, 2e4), ylabel = L"\log_{10}(f(x) + 1e13)",
        right_margin = 15Plots.mm, legend = :right)
scatter!(1e3 * remove_gap(grab_time(data["cout_wf_optim_gau_notrun"], r_time)),
         log10.(grab_cost(data["cout_wf_optim_gau_notrun"], r_time) .+ 1e13),
         color = "magenta", legendfontsize = 12, xlims = (0, 2e4),
         label = "Optim Gau", legend = :right)
scatter!(1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
         log10.(cal_cost_lbfgs_mean(data["results_LBFGS"]) .+ 1e13),
         color = "purple", xlims = (0, 2e4), legendfontsize = 12,
         label = "LBFGS", legend = :right)
scatter!(1e3 * remove_gap(grab_time(data["cout_wf_fisher_notrun"], r_time)),
         log10.(grab_cost(data["cout_wf_fisher_notrun"], r_time) .+ 1e13),
         color = "green", xlims = (0, 2e4), legendfontsize = 12,
         label = "Fisher", legend = :right)
subplot = twinx()
scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_wf_ls_notrun"], r_time)),
         nrmse2psnr.(grab_nrmse(data["cout_wf_ls_notrun"], r_time)),
         legendfontsize = 12, markershape = :rect, color = "red", label = "",
         ylabel = "PSNR (dB)", xlims = (0, 2e4), right_margin = 15Plots.mm)
scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_wf_optim_gau_notrun"], r_time)),
         nrmse2psnr.(grab_nrmse(data["cout_wf_optim_gau_notrun"], r_time)),
         legendfontsize = 12, xlims = (0, 2e4), markershape = :rect,
         color = "magenta", label = "")
scatter!(subplot, 1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"])),
         nrmse2psnr.(cal_nrmse_lbfgs_mean(data["results_LBFGS"], vec(xtrue))),
         legendfontsize = 12, markershape = :rect, color = "purple", label = "",
         xlims = (0, 2e4))
scatter!(subplot, 1e3 * remove_gap(grab_time(data["cout_wf_fisher_notrun"], r_time)),
         nrmse2psnr.(grab_nrmse(data["cout_wf_fisher_notrun"], r_time)),
         legendfontsize = 12, markershape = :rect, color = "green", label = "",
         xlims = (0, 2e4))

savefig(cpt*"result/A_canon_dft_wf_ls_vs_fisher/optimal_init_wf_stepsize_cost_fun_psnr_wref_2D_count="*string(avg_count)*".pdf")

# Todo: fisher vs lineser, no trun, plots conv time vs M
# Todo: trun vs notrun, fisher and liner, fix M, NRMSE vs trunreg; Pick a reasonable trunreg, then compare speed.
# trunreg = 25, nrmse_ls_trun = 0.107, nrmse_fisher_trun = 0.107
# trunreg = 30, nrmse_ls_trun = 0.113, nrmse_fisher_trun = 0.115
# trunreg = 35, nrmse_ls_trun = 0.118, nrmse_fisher_trun = 0.118

# plot the NRMSE results
# num_mask = 100
# avg_count = 0.25
# data = load(cpt*"result/A_dft_wf_ls_vs_fisher/optimal_init_wf_2D_stepsize_num_mask="*string(num_mask)*"count="*string(avg_count)*".jld2")
# scatter(1e3 * remove_gap(grab_time(data["cout_wf_emp_notrun"], r_time)[1:100]), 1e2 * grab_nrmse(data["cout_wf_emp_notrun"], r_time)[1:100],
#                     color = "blue",
#                     label = "Empirical",legendfontsize = 12, xguidefontsize=15, yguidefontsize=15, xlabel = "Time (ms)", ylabel = "NRMSE (%)")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_ls_notrun"], r_time)[1:75]),
#                     1e2 * grab_nrmse(data["cout_wf_ls_notrun"], r_time)[1:75],
#                     legendfontsize = 12,
#                     color = "red", label = "Backtracking")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_optim_gau_notrun"], r_time)[1:40]),
#                     1e2 * grab_nrmse(data["cout_wf_optim_gau_notrun"], r_time)[1:40],
#                     legendfontsize = 12,
#                     color = "magenta", label = "Optim Gau")
# scatter!(1e3 * remove_gap(cal_time_lbfgs_mean(data["results_LBFGS"]))[1:27],
#                     1e2 * cal_nrmse_lbfgs_mean(data["results_LBFGS"], xtrue)[1:27],
#                     legendfontsize = 12,
#                     color = "purple", label = "LBFGS")
# scatter!(1e3 * remove_gap(grab_time(data["cout_wf_fisher_notrun"], r_time)[1:40]),
#                     1e2 * grab_nrmse(data["cout_wf_fisher_notrun"], r_time)[1:40],
#                     legendfontsize = 12,
#                     color = "green", label = "Fisher")

# savefig(cpt*"result/A_dft_wf_ls_vs_fisher/optimal_init_wf_stepsize_nrmse_num_mask="*string(num_mask)*"count="*string(avg_count)*".pdf")

# M_list = [9000]
# if xhow === :real
#         xtrue = cumsum(mod.(1:N, 30) .== 0) .- 1.5
# elseif xhow === :complex
#         xtrue = (cumsum(mod.(1:N, 30) .== 0) .- 1.5) + im * (cumsum(mod.(1:N, 10) .== 0) .- 4.5)
# else
#         throw("unknown xhow")
# end

# plot_idx = 20
# scatter(1e3 * time_ls_notrun[1:plot_idx-1], nrmse_ls_notrun[1:plot_idx], label = "Line search w/o trun",
#         markershape = :rect, color = :blue, legendfontsize = 12, xguidefontsize=15, yguidefontsize=15)
# scatter!(1e3 * time_ls_trun[1:plot_idx-10], nrmse_ls_trun[1:plot_idx], label = "Line search w/ trun",
#         markershape = :rect, color = :cyan)
# scatter!(1e3 * time_fisher_notrun[1:3*plot_idx], nrmse_fisher_notrun[1:3*plot_idx], label = "Fisher w/o trun",
#         markershape = :circle, color = :red)
# scatter!(1e3 * time_fisher_trun[1:3*plot_idx-15], nrmse_fisher_trun[1:3*plot_idx], label = "Fisher w/ trun",
#         markershape = :circle, color = :orange)
# xlims!(0,30)
# xticks!(0:5:30)
# xlabel!("Time (ms)")
# ylabel!("NRMSE")
# savefig("plot_wf_fisher_vs_trun.pdf")


# A1 = LinearMapAA(
#     x -> fft(pad(x))[:],
#     y -> K * unpad(ifft(y))[:],
#     (K, N), (name="fft1D",), T=ComplexF32)
# A = A1
# mask_list = Array{AbstractArray{<:Number}, 1}(undef, num_mask)
#
# for i = 1:num_mask
#     mask = rand(N) .< 0.5
#     A_temp = LinearMapAA(
#                 x -> fft(pad(x .* mask))[:],
#                 y -> K * vec(unpad(ifft(y)) .* mask),
#                 (K, N), (name="fft1D",), T=ComplexF32)
#     global A = vcat(A, A_temp)
#     global mask_list[i] = vec(mask)
# end
# cons = avg_count / mean(abs2.(A * xtrue))
# A = sqrt(cons) * A
# b = 0.1 * ones(size(A, 1))
# y_true = abs2.(A * xtrue) .+ b
# y_pos = rand.(Poisson.(y_true))
# x0_rand = randn(N) + im * randn(N)
# x0_spectral = power_iter(A'*Diagonal(y_pos-b)*A, x0_rand, 50)
# α = sqrt(dot((y_pos-b), abs2.(A * x0_spectral))) / (norm(A * x0_spectral, 4)^2)
# x0_spectral = α * x0_spectral

# plan = WFState(x0_spectral, A, y_pos, b;
#                  niter = 200,
#                  gradhow = :poisson,
#                  xhow = :complex)
#
# Wirtinger_flow_inplace1!(A, y_pos, b, niter, plan; sthow = :fisher)
#
# xout, cout = Wirtinger_flow(A,y_pos,b; gradhow = :poisson, sthow = :fisher, istrun = false,
#                                         xhow = xhow, x0 = x0_spectral, niter = niter)
#
# isapprox(plan.x, xout)
#
#
# @btime Wirtinger_flow_inplace1!(A, y_pos, b, niter, plan; sthow = :fisher)
#
# @btime Wirtinger_flow(A,y_pos,b;
#                         gradhow = :poisson, sthow = :fisher, istrun = false,
#                         xhow = xhow, x0 = x0_spectral, niter = niter)
