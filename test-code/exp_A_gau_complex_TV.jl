include("utils.jl")

N = 100
r_time = 50
xhow = :complex
avg_count = 2
niter = 200
# Define xtrue
xtrue = (cumsum(mod.(1:N, 30) .== 0) .- 1.5) + im * (cumsum(mod.(1:N, 10) .== 0) .- 4.5)

phase_shift = x -> iszero(x) ? 1 : sign(xtrue' * x)
nrmse = x -> (norm(x - xtrue .* phase_shift(x)) / norm(xtrue .* phase_shift(x)))
phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)), time(), nrmse(x)]


M_list = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000]


for M in M_list

        (xout_wf_gau, cout_wf_gau,
                xout_wf_pois, cout_wf_pois,
                xout_wf_pois_huber, cout_wf_pois_huber,
                xout_gs, cout_gs,
                xout_lsmm, cout_lsmm,
                xout_admm, cout_admm,
                xout_lsmm_TV, cout_lsmm_TV,
                xout_lsmm_TV_huber, cout_lsmm_TV_huber,
                xout_admm_TV, cout_admm_TV,
                xout_admm_TV_huber, cout_admm_TV_huber) = [Array{Array{Any, 1}, 1}(undef, r_time) for _ = 1:20]

        for r = 1:r_time

                global A = sqrt(1/2) * (randn(M, N) + im * randn(M, N)) # Each iteration, A is different.
                global cons = avg_count / mean(abs2.(A * xtrue))
                global A = sqrt(cons) * A # scale matrix A
                global b = 0.1 * ones(M)
                global y_true = abs2.(A * xtrue) .+ b
                global y_pos = rand.(Poisson.(y_true))
                # Define x0
                global x0_rand = randn(N) + im * randn(N)
                global x0_spectral = power_iter(A'*Diagonal(y_pos-b)*A, x0_rand, 50)
                global α = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
                global x0_spectral = α * x0_spectral

                xout_wf_gau[r], cout_wf_gau[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :gaussian, istrun = false, sthow = :fisher,
                        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
                xout_wf_pois[r], cout_wf_pois[r] = Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, istrun = false, sthow = :fisher,
                        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
                xout_wf_pois_huber[r], cout_wf_pois_huber[r] = Wirtinger_flow_huber(A,y_pos,b;
                        x0 = x0_spectral, niter = niter, xhow = xhow,
                        reg1 = 8, reg2 = 0.5, fun = cost_func)
                xout_gs[r], cout_gs[r] = Gerchberg_saxton(A,y_pos,b;
                        xhow = xhow, updatehow =:cg, x0 = x0_spectral,
                        niter = niter, fun = cost_func)
                xout_lsmm[r], cout_lsmm[r] = LSMM(A,y_pos,b;
                        curvhow = :imp, xhow = xhow, updatehow =:cg,
                        x0 = x0_spectral, niter = niter, fun = cost_func)
                xout_admm[r], cout_admm[r] = ADMM(A,y_pos,b;
                        phow =:adaptive, xhow = xhow, updatehow =:cg,
                        x0 = x0_spectral, ρ = 16, niter = niter, fun = cost_func)
                xout_lsmm_TV[r], cout_lsmm_TV[r] = LSMM_TV(A,y_pos,b;
                        curvhow = :imp, xhow = xhow, reg1 = 8, reg2 = 0.5,
                        x0 = x0_spectral, niter = niter, ninner = 3, fun = cost_func)
                xout_admm_TV[r], cout_admm_TV[r] = ADMM_TV(A,y_pos,b;
                        phow =:adaptive, ninner = 3, reg1 = 8, reg2 = 0.5,
                        xhow = xhow, updatehow =:cg, x0 = x0_spectral, ρ = 16,
                        niter = niter, fun = cost_func)
                xout_lsmm_TV_huber[r], cout_lsmm_TV_huber[r] = LSMM_TV_huber(A,y_pos,b;
                        curvhow = :imp, xhow = xhow, reg1 = 8, reg2 = 0.5,
                        x0 = x0_spectral, niter = niter, fun = cost_func)
                xout_admm_TV_huber[r], cout_admm_TV_huber[r] = ADMM_TV_huber(A,y_pos,b;
                        phow =:adaptive, reg1 = 8, reg2 = 0.5,
                        xhow = xhow, updatehow =:cg, x0 = x0_spectral,
                        ρ = 16, niter = niter, fun = cost_func)
        end

        save("./A_gau_bne0/A_Gaussian_spectral_"*string(xhow)*"_M="*string(M)*".jld",
                "xout_wf_gau", xout_wf_gau,
                "cout_wf_gau", cout_wf_gau,
                "xout_wf_pois", xout_wf_pois,
                "cout_wf_pois", cout_wf_pois,
                "xout_wf_pois_huber", xout_wf_pois_huber,
                "cout_wf_pois_huber", cout_wf_pois_huber,
                "xout_gs", xout_gs,
                "cout_gs", cout_gs,
                "xout_lsmm", xout_lsmm,
                "cout_lsmm", cout_lsmm,
                "xout_admm", xout_admm,
                "cout_admm", cout_admm,
                "xout_lsmm_TV", xout_lsmm_TV,
                "cout_lsmm_TV", cout_lsmm_TV,
                "xout_lsmm_TV_huber", xout_lsmm_TV_huber,
                "cout_lsmm_TV_huber", cout_lsmm_TV_huber,
                "xout_admm_TV", xout_admm_TV,
                "cout_admm_TV", cout_admm_TV,
                "xout_admm_TV_huber", xout_admm_TV_huber,
                "cout_admm_TV_huber", cout_admm_TV_huber)
end
