include("utils.jl")
xhow = :real
# Read data
file = matopen("./AmpSLM_16x16/XH_test.mat")
xtrue = read(file, "XH_test") / 255 * 0.25
xtrue = xtrue[5,:]
close(file)
file = matopen("./AmpSLM_16x16//A_prVAMP.mat")
idx = Int.(floor.(LinRange(1,65536,10000)))
A = read(file, "A")[idx,:]
close(file)

# Calculate Lipschitz constant
LAA = opnorm(A'*A, 2)
M = size(A, 1)
N = size(A, 2)
b = 0.1 * ones(M)
niter = 500
y_true = abs2.(A * vec(xtrue)) .+ b
y_pos = rand.(Poisson.(y_true))
x0_rand = rand(N)

phase_shift = x -> sign.(vec(xtrue)' * vec(x))
nrmse = x -> iszero(x) ? 1.0 : (norm(x[:] - xtrue[:] .* phase_shift(x)) / norm(xtrue[:] .* phase_shift(x)))
phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
cost_func = (x,iter) -> [sum(phi.(A * x[:], y_pos, b)), time(), nrmse(x)]

lf = x -> log10(max(x,1e-16))
grab = (o,i) -> hcat(o...)[i,:]
lg = (o,i) -> lf.(grab(o,i))



xout_wf_gau, cout_wf_gau = Wirtinger_flow_fisher(A,y_pos,b;
            gradhow = :gaussian, xhow = xhow, x0 = x0_rand, niter = niter, fun = cost_func)
xout_wf_pois, cout_wf_pois = Wirtinger_flow_fisher(A,y_pos,b;
            gradhow = :poisson, xhow = xhow, x0 = x0_rand, niter = niter, fun = cost_func)
xout_gs, cout_gs = Gerchberg_saxton(A,y_pos,b;
            xhow = xhow, updatehow =:cg, x0 = x0_rand, niter = niter, fun = cost_func)
xout_lsmm, cout_lsmm = LSMM(A,y_pos,b;
            curvhow = :imp, xhow = xhow, updatehow =:cg, x0 = x0_rand, niter = niter, fun = cost_func)
xout_admm, cout_admm = poisson_admm(A,y_pos,b;
            phow =:adaptive, xhow = xhow, updatehow =:cg, x0 = x0_rand, ρ = 16, niter = niter, fun = cost_func);
xout_lsmm_PGM, cout_lsmm_PGM = LSMM_L1_PGM(A,y_pos,b,LAA;
            curvhow = :imp, xhow = xhow, reg = 32,
            x0 = x0_rand, niter = niter, ninner = 3, fun = cost_func)
xout_admm_PGM, cout_admm_PGM = poisson_admm_PGM(A,y_pos,b,LAA;
            phow =:adaptive, ninner = 3, reg = 32,
            xhow = xhow, x0 = x0_rand, ρ = 16, niter = niter, fun = cost_func)

save("./A_ETM/A_etm_16.jld","xout_wf_gau", xout_wf_gau,
                "cout_wf_gau", cout_wf_gau,
                "xout_wf_pois", xout_wf_pois,
                "cout_wf_pois", cout_wf_pois,
                "xout_gs", xout_gs,
                "cout_gs", cout_gs,
                "xout_lsmm", xout_lsmm,
                "cout_lsmm", cout_lsmm,
                "xout_admm", xout_admm,
                "cout_admm", cout_admm,
                "xout_lsmm_PGM", xout_lsmm_PGM,
                "cout_lsmm_PGM", cout_lsmm_PGM,
                "xout_admm_PGM", xout_admm_PGM,
                "cout_admm_PGM", cout_admm_PGM)
