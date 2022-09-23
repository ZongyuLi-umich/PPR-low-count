include("utils.jl")

r_time = 5
xhow = :real
avg_count = 0.25
niter = 100
# Define xtrue
xtrue = load("../test-data-2D/4-test-images.jld2")["x3"]
nrmse2psnr(x) = 10 * log10(1 / (x * norm(vec(xtrue))^2/N2))
ref = load("../test-data-2D/4-test-images.jld2")["ref_x3"]
zz = zeros(size(xtrue))
# zz = imresize(xall[5:1114, 1121:2230], 256,256)
# ref = imresize(xall[5:1114, 2241:3350],256,256)
# ref = imresize(xall[5:1114, 2241:3350],32,32)
# ref = zz
jim(ref)
# xtrue = JLD2.load("../test-data-2D/UM-1817-shepplogan.jld2")["x1"]
N = size(xtrue, 2)
N2 = N^2

phase_shift = x -> iszero(x) ? 1 : sign(vec(xtrue)' * x)
nrmse = x -> (norm(x - vec(xtrue) .* phase_shift(x)) / norm(vec(xtrue) .* phase_shift(x)))
phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
grad_phi = (v, yi, bi) -> 2 * v * (1 - yi / (abs2(v) + bi))
T = LinearMapAA(x -> diff2d_forw(x, N, N), y -> diff2d_adj(y, N, N), (2*N*(N-1), N2); T=Float64)
huber(t, α) = abs(t) < α ? 0.5 * abs2(t) : α * abs(t) - 0.5 * α^2
grad_huber(t, α) = abs(t) < α ? t : α * sign(t)
β = 2
α = 0.5
cost_func = (x,iter) -> [sum(phi.(A * x, y_pos, b)) + β * sum(huber.(T * x, α)), time(), nrmse(x)]

K = 4 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]

A = LinearMapAA(x -> vec(fft(pad(reshape(x, N, N)))),
                y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
                (K*K, N*N), (name="fft2D",), T=ComplexF32)
cons = avg_count / mean(abs2.(A * vec(xtrue)))
A = sqrt(cons) * A # scale matrix A
b = 0.1 * ones(size(A, 1))
y_true = abs2.(A * (vec(xtrue) + vec(ref))) + b
y_pos = rand.(Poisson.(y_true))
# Define x0
x0_rand = randn(N2)
# global x0_spectral = power_iter(A'*Diagonal(y_pos-b)*A, x0_rand, 50)
x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)
scale_factor = sqrt(y_pos' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
x0_spectral = abs.(scale_factor * x0_spectral)
jim(reshape(x0_spectral, N, N))
# f(x) = sum(phi.(A * (x+vec(ref)), y_pos, b)) + β * sum(huber.(T * x, α))
# g!(y, x) = copyto!(y, real(A' * grad_phi.(A * (x+vec(ref)), y_pos, b) + β * T' * grad_huber.(T * x, α)))
f(x) = sum(phi.(A * (x+vec(ref)), y_pos, b))
g!(y, x) = copyto!(y, real(A' * grad_phi.(A * (x+vec(ref)), y_pos, b)))
results_LBFGS = optimize(f, g!, x0_spectral, LBFGS(), Optim.Options(store_trace = true,
                            show_trace = false, extended_trace = true))
xhat = reshape(results_LBFGS.minimizer, N, N)
jim(xhat)
# f(x) = sum(phi.(A * x, y_pos, b)) + β * sum(huber.(T * x, α))
# g!(y, x) = copyto!(y, real(A' * grad_phi.(A * x, y_pos, b) + β * T' * grad_huber.(T * x, α)))
grad!(y, Ax, Tx) = copyto!(y, real(A' * grad_phi.(Ax, y_pos, b) + β * T' * grad_huber.(Tx, α)))
xout_wf_pois_huber_fisher, cout_wf_pois_huber_fisher = Wirtinger_flow_huber(A,y_pos,b;
                    x0 = x0_spectral, niter = niter, xhow = xhow, sthow = :fisher,
                    reg1 = β, reg2 = α, fun = cost_func)
results_LBFGS = optimize(f, g!, x0_spectral, LBFGS(), Optim.Options(store_trace = true,
                            show_trace = false, extended_trace = true))
x1 = reshape(xout_wf_pois_huber_fisher, N, N)
x2 = reshape(results_LBFGS.minimizer, N, N)
jim(jim(reshape(xtrue, N, N), "xtrue"),
    jim(reshape(x0_spectral, N, N), "init"),
    jim(x1, "x1"), jim(x2, "x2"))
xout_lsmm_TV_huber, cout_lsmm_TV_huber = LSMM_TV_huber(A,y_pos,b;
                    curvhow = :imp, xhow = xhow, reg1 = β, reg2 = α,
                    x0 = x0_spectral, niter = niter, fun = cost_func)
xout_admm_TV_huber, cout_admm_TV_huber = ADMM_TV_huber(A,y_pos,b;
                    phow =:adaptive, reg1 = β, reg2 = α,
                    xhow = xhow, updatehow =:cg, x0 = x0_spectral,
                    ρ = 32, niter = niter, fun = cost_func)
jim(jim(reshape(xout_lsmm_TV_huber, N, N)),
    jim(reshape(xout_admm_TV_huber, N, N)))
xx
# """
#     pad2d(x, up)
# """
# function pad2d(x, up)
#         y = zeros(eltype(x), up .* size(x))
#         y[1:up:up*size(x, 1), 1:up:up*size(x, 2)] .= x
#         return y
# end
# """
#     canonical_fft2d(size_in, up)
# create a canonical fft matrix with upsample rate `up`
# `size_in`: size of the input
# `up`: upsample rate
# """
# function canonical_fft2d(size_in::Tuple{<:Int64, <:Int64},
#                          up::Int;
#                          is_real = true)
#     m, n = size_in
#     if is_real
#         row = repeat(Vector(1:up:up*m),outer = n)
#         col = repeat(Vector(1:up:up*n),inner = m)
#         return LinearMapAA(x -> vec(fft(sparse(row, col, vec(x), up*m, up*n))),
#                         y -> up*m*up*n * vec(ifft(reshape(y, up*m, up*n))[1:up:up*m, 1:up:up*n]),
#                         (up*m*up*n, m*n);
#                         # idim = (m, n),
#                         # odim = (up*m, up*n);
#                         T = ComplexF32)
#     else
#         return LinearMapAA(x -> vec(fft(pad2d(reshape(x, m, n), up))),
#                         y -> up*m*up*n * vec(ifft(reshape(y, up*m, up*n))[1:up:up*m, 1:up:up*n]),
#                         (up*m*up*n, m*n);
#                         # idim = (m, n),
#                         # odim = (up*m, up*n);
#                         T = ComplexF32)
#     end
# end
#
# # adjoint test
# # f = x -> vec(fft(pad2d(x, up)))
# # x = randn(8, 6)
# # y = randn(16, 12)
# # up = 2
# # A = canonical_fft2d(size(x), up; is_real = true)
# # isapprox(vec(y)' * (A * vec(x)), conj(vec(x)' * (A' * vec(y))))
