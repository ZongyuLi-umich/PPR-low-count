include("utils.jl")

# N = 500
seed!(0)
r_time = 1
xhow = :complex
niter = 200
# Define xtrue
xtrue = load("../test-data-2D/4-test-images.jld2")["x4"]
N = size(xtrue, 1)
N2 = N^2

M = size(A, 1)

avg_count = 0.25



(xout_wf_gau, cout_wf_gau,
                xout_gs, cout_gs,
                xout_wf_pois, cout_wf_pois,
                xout_wf_pois_huber, cout_wf_pois_huber) = [Array{Array{Any, 1}, 1}(undef, r_time) for _ = 1:8]

for r = 1:r_time
        file = matopen("../AmpSLM_16x16/A_prVAMP.mat")
        A = read(file, "A")
        close(file)
        # idx = 1:4:size(AA, 2)
        # A = AA[:, idx]
        cons = avg_count / mean(abs2.(A * vec(xtrue)))
        A = sqrt(cons) * A # scale matrix A
        b = 0.1 * ones(M)
        y_true = abs2.(A * vec(xtrue)) .+ b
        y_pos = rand.(Poisson.(y_true))
        # Define x0
        x0_rand = randn(N2) + im * randn(N2)
        # global x0_spectral = power_iter(A'*Diagonal(y_pos-b)*A, x0_rand, 50)
        x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)
        scale_factor = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
        x0_spectral = scale_factor * x0_spectral

        phase_shift = x -> iszero(x) ? 1 : sign(vec(xtrue)' * x)
        nrmse = x -> (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x)))
        phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
        cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)), time(), nrmse(x)]

        xout_wf_gau[r], cout_wf_gau[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :gaussian, istrun = false, sthow = :fisher,
                        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
        xout_wf_pois[r], cout_wf_pois[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, istrun = false, sthow = :fisher,
                        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
        xout_wf_pois_huber[r], cout_wf_pois_huber[r] = Wirtinger_flow_huber(A,y_pos,b;
                        x0 = x0_spectral, niter = niter, xhow = xhow,
                        reg1 = 32, reg2 = 0.1, fun = cost_func)
        xout_gs[r], cout_gs[r] = Gerchberg_saxton(A,y_pos,b;
                        xhow = xhow, updatehow =:cg, x0 = x0_spectral,
                        niter = niter, fun = cost_func)
end

save("./result/A_ETM_wf_vs_poi/optimal_init_2D_gau_vs_poi_avgcount="*string(avg_count)*".jld2",
                "xout_wf_gau", xout_wf_gau,
                "cout_wf_gau", cout_wf_gau,
                "xout_wf_pois", xout_wf_pois,
                "cout_wf_pois", cout_wf_pois,
                "xout_wf_pois_huber", xout_wf_pois_huber,
                "cout_wf_pois_huber", cout_wf_pois_huber,
                "xout_gs", xout_gs,
                "cout_gs", cout_gs)

# plot data
data = load("./result/A_ETM_wf_vs_poi/optimal_init_2D_gau_vs_poi_avgcount="*string(avg_count)*".jld2")
xout_wf_gau = ComplexF32.(reshape(data["xout_wf_gau"][1], N, N))
xout_wf_pois = ComplexF32.(reshape(data["xout_wf_pois"][1], N, N))
xout_wf_pois_huber = ComplexF32.(reshape(data["xout_wf_pois_huber"][1], N, N))
phase_shift = x -> iszero(x) ? 1 : sign(vec(xtrue)' * x)
nrmse = x -> round(100 * (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x))); digits = 1)
nrmse_gau = nrmse(vec(xout_wf_gau))
jim(xout_wf_gau, "WF Gaussian, NRMSE = $nrmse_gau%")
savefig("./result/A_ETM_wf_vs_poi/optimal_init_2D_wf_gau.pdf")
nrmse_pois = nrmse(vec(xout_wf_pois))
jim(xout_wf_pois, "WF Poisson, NRMSE = $nrmse_pois%")
savefig("./result/A_ETM_wf_vs_poi/optimal_init_2D_wf_pois.pdf")
nrmse_pois_huber = nrmse(vec(xout_wf_pois_huber))
jim(xout_wf_pois_huber, "WF Poisson TV, NRMSE = $nrmse_pois_huber%")
savefig("./result/A_ETM_wf_vs_poi/optimal_init_2D_wf_pois_TV.pdf")
xx
# plot data
# avg_count = 0.25:0.25:2.0
# data_list = Vector{Dict{String, Any}}(undef, 8)
# for (i, c) in enumerate(avg_count)
#     data_list[i] = load("./result/A_ETM_wf_vs_poi/optimal_init_2D_gau_vs_poi_avgcount="*string(c)*".jld2")
# end
# nrmse_wf_gau = Vector(undef, 8)
# nrmse_wf_poi = Vector(undef, 8)
# nrmse_gs = Vector(undef, 8)
# nrmse_wf_poi_huber = Vector(undef, 8)
#
# for i = 1:8
#     nrmse_wf_gau[i] = grab_nrmse(data_list[i]["cout_wf_gau"], r_time)[end]
#     nrmse_wf_poi[i] = grab_nrmse(data_list[i]["cout_wf_pois"], r_time)[end]
#     nrmse_gs[i] = grab_nrmse(data_list[i]["cout_gs"], r_time)[end]
#     nrmse_wf_poi_huber[i] = grab_nrmse(data_list[i]["cout_wf_pois_huber"], r_time)[end]
# end
#
# plot(avg_count, 1e2 * nrmse_gs, color = "blue", markershape = :circle,
#         label = "GS", legendfontsize = 12, xguidefontsize=15, yguidefontsize=15,
#         xlabel = "Count level", xticks = 0.25:0.25:2.0, ylabel = "NRMSE (%)")
# plot!(avg_count, 1e2 * nrmse_wf_gau, color = "red",markershape = :circle,
#         label = "WF Gaussian", legendfontsize = 12)
# plot!(avg_count, 1e2 * nrmse_wf_poi, color = "magenta",markershape = :circle,
#         label = "WF Poisson", legendfontsize = 12)
# plot!(avg_count, 1e2 * nrmse_wf_poi_huber, color = "green",markershape = :circle,
#         label = "WF Poisson TV", legendfontsize = 12)
#
# savefig("./result/A_ETM_wf_vs_poi/optimal_init_nrmse_gau_vs_poi.pdf")
