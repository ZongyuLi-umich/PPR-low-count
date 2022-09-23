include("utils.jl")

# N = 500
seed!(0)
r_time = 1
xhow = :real
niter = 100
# Define xtrue
xtrue = load("../test-data-2D/4-test-images.jld2")["x3"]
nrmse2psnr(x) = 10 * log10(1 / (x * norm(vec(xtrue))^2/N2))
ref = load("../test-data-2D/4-test-images.jld2")["ref_x3"]
N = size(xtrue, 1)
N2 = N^2

avg_count = 0.25
K = 4 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]


(xout_wf_gau, cout_wf_gau,
        xout_wf_pois, cout_wf_pois,
        xout_wf_pois_huber, cout_wf_pois_huber) = [Array{Array{Any, 1}, 1}(undef, r_time) for _ = 1:6]

for r = 1:r_time
        global A = LinearMapAA(x -> vec(fft(pad(reshape(x, N, N)))),
                    y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
                    (K*K, N*N), (name="fft2D",), T=ComplexF64)

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

        xout_wf_pois_huber[r], cout_wf_pois_huber[r] = Wirtinger_flow_huber(A,y_pos,b;
                        x0 = x0_spectral, niter = niter, xhow = xhow,
                        reg1 = 32, reg2 = 0.1, ref = ref, fun = cost_func)

        xout_wf_gau[r], cout_wf_gau[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :gaussian, istrun = false, sthow = :fisher,
                        xhow = xhow, x0 = x0_spectral, ref = ref, niter = niter, fun = cost_func)
        xout_wf_pois[r], cout_wf_pois[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, istrun = false, sthow = :fisher,
                        xhow = xhow, x0 = x0_spectral, ref = ref, niter = niter, fun = cost_func)

end

save("./result/A_canon_dft_wf_vs_poi/optimal_init_2D_gau_vs_poi_avgcount="*string(avg_count)*".jld2",
                "xout_wf_gau", xout_wf_gau,
                "cout_wf_gau", cout_wf_gau,
                "xout_wf_pois", xout_wf_pois,
                "cout_wf_pois", cout_wf_pois,
                "xout_wf_pois_huber", xout_wf_pois_huber,
                "cout_wf_pois_huber", cout_wf_pois_huber)

# plot data
data = load("./result/A_canon_dft_wf_vs_poi/optimal_init_2D_gau_vs_poi_avgcount="*string(avg_count)*".jld2")
xout_wf_gau = Float32.(reshape(data["xout_wf_gau"][1], N, N))
xout_wf_pois = Float32.(reshape(data["xout_wf_pois"][1], N, N))
xout_wf_pois_huber = Float32.(reshape(data["xout_wf_pois_huber"][1], N, N))
phase_shift = x -> iszero(x) ? 1 : sign(vec(xtrue)' * x)
nrmse = x -> round(100 * (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x))); digits = 1)
nrmse_gau = nrmse(vec(xout_wf_gau))
jim(xout_wf_gau, "WF Gaussian, NRMSE = $nrmse_gau%")
savefig("./result/A_canon_dft_wf_vs_poi/optimal_init_2D_wf_gau.pdf")
nrmse_pois = nrmse(vec(xout_wf_pois))
jim(xout_wf_pois, "WF Poisson, NRMSE = $nrmse_pois%")
savefig("./result/A_canon_dft_wf_vs_poi/optimal_init_2D_wf_pois.pdf")
nrmse_pois_huber = nrmse(vec(xout_wf_pois_huber))
jim(xout_wf_pois_huber, "WF Poisson TV, NRMSE = $nrmse_pois_huber%")
savefig("./result/A_canon_dft_wf_vs_poi/optimal_init_2D_wf_pois_TV.pdf")
xx
# avg_count = 0.25:0.25:2
# data_list = Vector{Dict{String, Any}}(undef, length(avg_count))
# for (i, c) in enumerate(avg_count)
#     data_list[i] = load("./result/A_dft_wf_vs_poi/optimal_init_2D_gau_vs_poi_avgcount="*string(c)*".jld2")
# end
# nrmse_wf_gau = Vector(undef, length(avg_count))
# nrmse_wf_poi = Vector(undef, length(avg_count))
# nrmse_gs = Vector(undef, length(avg_count))
# nrmse_wf_poi_huber = Vector(undef, length(avg_count))
#
# for i = 1:length(avg_count)
#     nrmse_wf_gau[i] = grab_nrmse(data_list[i]["cout_wf_gau"], r_time)[end]
#     nrmse_wf_poi[i] = grab_nrmse(data_list[i]["cout_wf_pois"], r_time)[end]
#     nrmse_gs[i] = grab_nrmse(data_list[i]["cout_gs"], r_time)[end]
#     nrmse_wf_poi_huber[i] = grab_nrmse(data_list[i]["cout_wf_pois_huber"], r_time)[end]
# end
#
# plot(avg_count, 1e2 * nrmse_wf_gau, color = "red",markershape = :circle,
#         label = "WF Gaussian", legendfontsize = 12)
# plot!(avg_count, 1e2 * nrmse_gs, color = "blue", markershape = :circle,
#         label = "GS", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
#         xlabel = "Count level", xticks = 0.25:0.25:2.0, ylabel = "NRMSE (%)")
# plot!(avg_count, 1e2 * nrmse_wf_poi, color = "magenta",markershape = :circle,
#         label = "WF Poisson", legendfontsize = 12)
# plot!(avg_count, 1e2 * nrmse_wf_poi_huber, color = "green",markershape = :circle,
#         label = "WF Poisson TV", legendfontsize = 12)
#
# savefig("./result/A_dft_wf_vs_poi/optimal_init_2D_gau_vs_poi.pdf")
