include("utils.jl")

N = 500
xhow = :complex
avg_count = 5
niter = 200

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

M = 80000
b = 0.1 * ones(M)
A = randn(M, N) + im * randn(M, N)
cons = avg_count / mean(abs2.(A * xtrue))
A = sqrt(cons) * A
y_true = abs2.(A * xtrue) .+ b
y_pos = rand.(Poisson.(y_true))
x0_rand = randn(N) + im * randn(N)
x0_spectral = power_iter(A'*Diagonal(y_pos-b)*A, x0_rand, 50)
α = sqrt(dot((y_pos-b), abs2.(A * x0_spectral))) / (norm(A * x0_spectral, 4)^2)
x0_spectral = α * x0_spectral

plan = WFState(x0_spectral, A, y_pos, b;
                 niter = 200,
                 gradhow = :poisson,
                 xhow = :complex)

Wirtinger_flow_inplace1!(A, y_pos, b, niter, plan; sthow = :fisher)

xout, cout = Wirtinger_flow(A,y_pos,b; gradhow = :poisson, sthow = :fisher, istrun = false,
                                        xhow = xhow, x0 = x0_spectral, niter = niter)

isapprox(plan.x, xout)

# 6.130 s (0 allocations: 0 bytes)
@btime Wirtinger_flow_inplace1!(A, y_pos, b, niter, plan; sthow = :fisher)

# 6.027 s (3008 allocations: 1.08 GiB)
@btime Wirtinger_flow(A,y_pos,b;
                        gradhow = :poisson, sthow = :fisher, istrun = false,
                        xhow = xhow, x0 = x0_spectral, niter = niter)
