include("utils.jl")

xhow = :real
niter = 200
# Define xtrue
xtrue = load("./test-data-2D/UM-1817-shepplogan.jld2")["x1"]
N = size(xtrue, 1)
N2 = N^2
avg_count = 0.25

phase_shift = x -> iszero(x) ? 1 : sign(vec(xtrue)' * x)
nrmse = x -> (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x)))
phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)), time(), nrmse(x)]


# M = 80000
num_mask = 100
K = 2 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]

A1 = LinearMapAA(
    x -> vec(fft(pad(reshape(x, N, N)))),
    y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
    (K*K, N*N), (name="fft2D",), T=ComplexF64)

A = A1
mask_list = Array{AbstractArray{<:Number}, 1}(undef, num_mask)

for i = 1:num_mask
    mask = rand(N, N) .< 0.5
    A_temp = LinearMapAA(
            x -> vec(fft(pad(reshape(x, N, N) .* mask))),
            y -> (K*K) * vec(unpad(ifft(reshape(y, K, K))) .* mask),
            (K*K, N*N), (name="fft2D",), T=ComplexF64)
    global A = vcat(A, A_temp)
    global mask_list[i] = vec(mask)
end

cons = avg_count / mean(abs2.(A * vec(xtrue)))
A = sqrt(cons) * A # scale matrix A
b = 0.1 * ones(size(A, 1))
y_true = abs2.(A * vec(xtrue)) .+ b
y_pos = rand.(Poisson.(y_true))
# Define x0
x0_rand = randn(N2) + im * randn(N2)
x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)
scale_factor = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
x0_spectral = abs.(scale_factor * x0_spectral)

tot = 20
cout_fisher_notrun_list = Vector(undef, tot)
cout_fisher_trun_list = Vector(undef, tot)

xout_wf_fisher_notrun, cout_wf_fisher_notrun = Wirtinger_flow(A,y_pos,b;
        gradhow = :poisson, sthow = :fisher, istrun = false,
        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)

for i = 1:tot
        cout_fisher_notrun_list[i] = cout_wf_fisher_notrun
end

for i = 1:tot
        trunreg = 50 * i
        xout_wf_fisher_trun, cout_wf_fisher_trun = Wirtinger_flow(A,y_pos,b;
                gradhow = :poisson, sthow = :fisher, istrun = true, trunreg = trunreg,
                xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
        cout_fisher_trun_list[i] = cout_wf_fisher_trun
end

save("./result/A_dft_wf_trun_vs_notrun/cout_trun_vs_notrun.jld2",
    "cout_fisher_notrun_list", cout_fisher_notrun_list,
    "cout_fisher_trun_list", cout_fisher_trun_list)

nrmse_fisher_notrun_list = load("./result/A_dft_wf_trun_vs_notrun/nrmse_trun_vs_notrun.jld2")["nrmse_fisher_notrun_list"]
nrmse_fisher_trun_list = load("./result/A_dft_wf_trun_vs_notrun/nrmse_trun_vs_notrun.jld2")["nrmse_fisher_trun_list"]
# idxrange = 50:50:1000
# plot(idxrange, nrmse_fisher_trun_list, label = "Truncated WF", xlabel = L"a^h", ylabel = "NRMSE", color = :blue, legendfontsize = 16, xguidefontsize=16, yguidefontsize=15)
# # # plot!(idxrange, nrmse_fisher_notrun_list, label = "Non-truncated WF", color = :red, linestyle =:dash)
# # # xlabel!(L"a^h")
# # # ylabel!("NRMSE")
# savefig("./result/A_dft_wf_trun_vs_notrun/trun_vs_notrun_20220310.pdf")
# plot(idxrange, keep_rate_list, label = "Truncated WF", color = :blue, legendfontsize = 12, xguidefontsize=15, yguidefontsize=15)
# xlabel!(L"a^h")
# ylabel!("Fraction of kept indicies")
# yticks!(0.3:0.1:1.0)
# savefig("./A_gau_wf_ls_vs_fisher/trun_vs_notrun_frac_kept_data_20210427.pdf")
#
# trunreg = 100
# seed!(0)
# b = 0.1 * ones(M)
# A = randn(M, N) + im * randn(M, N) # Each iteration, A is different.
# cons = avg_count / mean(abs2.(A * xtrue))
# A = sqrt(cons) * A # scale matrix A
# y_true = abs2.(A * xtrue) .+ b
# y_pos = rand.(Poisson.(y_true))
# if xhow === :real
#         x0_rand = randn(N)
# elseif xhow === :complex
#         x0_rand = randn(N) + im * randn(N)
# else
#         throw("unknown xhow")
# end
#
# xout_wf_fisher_notrun, cout_wf_fisher_notrun = Wirtinger_flow(A,y_pos,b;
#         gradhow = :poisson, sthow = :fisher, istrun = false,
#         xhow = xhow, x0 = x0_rand, niter = niter, fun = cost_func)
# xout_wf_fisher_trun, cout_wf_fisher_trun = Wirtinger_flow(A,y_pos,b;
#         gradhow = :poisson, sthow = :fisher, istrun = true, trunreg = trunreg,
#         xhow = xhow, x0 = x0_rand, niter = niter, fun = cost_func)
#
# time_fisher_notrun = grab(cout_wf_fisher_notrun, 2) .- cout_wf_fisher_notrun[1][2]
# nrmse_fisher_notrun = grab(cout_wf_fisher_notrun, 3)
# time_fisher_trun = grab(cout_wf_fisher_trun, 2) .- cout_wf_fisher_trun[1][2]
# nrmse_fisher_trun = grab(cout_wf_fisher_trun, 3)
#
# plot_idx = 25
# scatter(1e3 * time_fisher_trun[1:plot_idx], nrmse_fisher_trun[1:plot_idx], label = "Truncated WF",
#         markershape = :circle, color = :blue, legendfontsize = 12, xguidefontsize=15, yguidefontsize = 15)
# scatter!(1e3 * time_fisher_notrun[1:3*plot_idx], nrmse_fisher_notrun[1:3*plot_idx], label = "Non-truncated WF",
#         markershape = :circle, color = :red)
#
# xlims!(0,80)
# xticks!(0:10:80)
# yticks!(0:0.2:1.4)
# xlabel!("Time (ms)")
# ylabel!("NRMSE")
# savefig("./A_gau_wf_ls_vs_fisher/trun_vs_notrun_speed_20210427.pdf")

# conv_fisher_notrun = 1
# conv_fisher_trun = 1

# for i = 1:niter-1
#         if abs(nrmse_fisher_notrun[i] - nrmse_fisher_notrun[i+1]) > 1e-3
#                 conv_fisher_notrun += 1
#         end
#         if abs(nrmse_fisher_trun[i] - nrmse_fisher_trun[i+1]) > 1e-3
#                 conv_fisher_trun += 1
#         end
# end

# @show 1000 * time_ls_notrun[conv_ls_notrun]
# @show 1000 * time_fisher_notrun[conv_fisher_notrun]
# @show 1000 * time_ls_trun[conv_ls_trun]
# @show 1000 * time_fisher_trun[conv_fisher_trun]


# Todo: fisher vs lineser, no trun, plots conv time vs M
# Todo: trun vs notrun, fisher and liner, fix M, NRMSE vs trunreg; Pick a reasonable trunreg, then compare speed.
# trunreg = 25, nrmse_ls_trun = 0.107, nrmse_fisher_trun = 0.107
# trunreg = 30, nrmse_ls_trun = 0.113, nrmse_fisher_trun = 0.115
# trunreg = 35, nrmse_ls_trun = 0.118, nrmse_fisher_trun = 0.118



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
