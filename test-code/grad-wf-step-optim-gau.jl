include("utils.jl")
N = 500
xhow = :complex
avg_count = 0.25 # lowest count
niter = 200
M = 80000
cpt = "/n/calumet/x/zonyul/Poisson_phase_retri_2021/result/grad_wf_step_optim_gau/"

if xhow === :real
        xtrue = cumsum(mod.(1:N, 30) .== 0) .- 1.5
elseif xhow === :complex
        xtrue = (cumsum(mod.(1:N, 30) .== 0) .- 1.5) + im * (cumsum(mod.(1:N, 10) .== 0) .- 4.5)
else
        throw("unknown xhow")
end


lf = x -> log10(max(x,1e-16))
grab = (o,i) -> hcat(o...)[i,:]
lg = (o,i) -> lf.(grab(o,i))

b = 0.1 * ones(M)
A = randn(M, N) + im * randn(M, N) # Each iteration, A is different.
cons = avg_count / mean(abs2.(A * xtrue))
A = sqrt(cons) * A # scale matrix A
y_true = abs2.(A * xtrue) .+ b
y_pos = rand.(Poisson.(y_true))
if xhow === :real
        global x0_rand = randn(N)
elseif xhow === :complex
        global x0_rand = randn(N) + im * randn(N)
else
        throw("unknown xhow")
end
x0_spectral = power_iter(A'*Diagonal(y_pos-b)*A, x0_rand, 50)
α = sqrt(dot(y_pos - b, abs2.(A * x0_spectral))) / (norm(A * x0_spectral, 4)^2)
x0_spectral = α * x0_spectral

phase_shift = x -> iszero(x) ? 1 : sign(xtrue' * x)
nrmse = x -> (norm(x - xtrue .* phase_shift(x)) / norm(xtrue .* phase_shift(x)))
phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
# cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)), time(), nrmse(x), norm(grad_phi.(A * x, y_pos, b))]
f(x) = sum(phi.(A * x, y_pos, b))
cost_func = (x,iter) -> [f(x), time(), nrmse(x), norm(A' * grad_phi.(A * x, y_pos, b))]

g!(y, x) = copyto!(y, A' * grad_phi.(A * x, y_pos, b))

# LBFGS
results_LBFGS = optimize(f, g!, x0_spectral, LBFGS(), Optim.Options(store_trace = true,
                                                                       show_trace = false,
                                                                       extended_trace = true))
norm_grad_LBFGS = [Optim.trace(results_LBFGS)[i].g_norm for i = 1:length(Optim.trace(results_LBFGS))]
cost_fun_LBFGS = [Optim.trace(results_LBFGS)[i].value for i = 1:length(Optim.trace(results_LBFGS))]
# the step size mu vanishes before gradient goes to zero
xout_wf_optim_gau_notrun, cout_wf_optim_gau_notrun = Wirtinger_flow(A,y_pos,b;
        gradhow = :poisson, sthow = :optim_gau, istrun = false,
        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
mu_list = [-24.94, 8.23, 0.334, 0.0196, 0.0012, 6.98e-5, 4.17e-6, 2.49e-7, 1.485e-8, 8.867e-10, 5.292e-11, 3.15e-12, 1.85e-13, 0, 0]
scatter(mu_list, xlabel = "niter", label = "step size")
savefig(cpt*"step-size-optim-gau.pdf")
xout_iter = grab(cout_wf_optim_gau_notrun, 1)
nrmse_x = grab(cout_wf_optim_gau_notrun, 3)
norm_grad = grab(cout_wf_optim_gau_notrun, 4)
scatter(norm_grad, label = "norm of gradient", xlabel = "niter")
savefig(cpt*"norm-grad-optim-gau.pdf")
norm(xout_iter[end] - xout_iter[end-4])

xout_wf_fisher_notrun, cout_wf_fisher_notrun = Wirtinger_flow(A,y_pos,b;
        gradhow = :poisson, sthow = :fisher, istrun = false,
        xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)

cost_fun0 = grab(cout_wf_fisher_notrun, 1)
norm_grad_fisher = grab(cout_wf_fisher_notrun, 4)
scatter(log10.(norm_grad_fisher[1:100]), label = "Fisher", xlabel = "niter", ylabel = "norm of grad")
scatter!(log10.(norm_grad_LBFGS), label = "LBFGS")
savefig(cpt*"norm-grad-fisher-LBFGS.pdf")

# initialize LBFGS with xout_wf_fisher_notrun
results1_LBFGS = optimize(f, g!, xout_wf_fisher_notrun, LBFGS(), Optim.Options(store_trace = true,
                                                                               show_trace = false,
                                                                               extended_trace = true))
# initialize WF with Optim.minimizer(results_LBFGS)
xout1_wf_fisher_notrun, cout1_wf_fisher_notrun = Wirtinger_flow(A,y_pos,b;
        gradhow = :poisson, sthow = :fisher, istrun = false,
        xhow = xhow, x0 = Optim.minimizer(results_LBFGS), niter = niter, fun = cost_func)

cost_fun1 = grab(cout1_wf_fisher_notrun, 1)
scatter(cost_fun1, title = "cost function value of WF when initializing with LBFGS")
