include("utils.jl")

xhow = :real
xtrue = load("./result/A_fft_shepplogan/test_data_fft.jld")["x1"]
N = size(xtrue, 1)
avg_count = 1
num_mask = 20
niter = 200

K = 2 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]
A1 = LinearMapAA(
    x -> fft(pad(reshape(x, N, N)))[:],
    y -> (K*K) * unpad(ifft(reshape(y, K, K)))[:],
    (K*K, N*N), (name="fft2D",), T=ComplexF32)
A = A1
mask_list = Array{AbstractArray{<:Number}, 1}(undef, num_mask)
for i = 1:num_mask
    mask = rand(N, N) .< 0.5
    A_temp = LinearMapAA(
                x -> fft(pad(reshape(x, N, N) .* mask))[:],
                y -> (K*K) * vec(unpad(ifft(reshape(y, K, K))) .* mask),
                (K*K, N*N), (name="fft2D",), T=ComplexF32)
    global A = vcat(A, A_temp)
    global mask_list[i] = vec(mask)
end
cons = avg_count / mean(abs2.(A * xtrue[:]))
A = sqrt(cons) * A # scale matrix A
# absA = sqrt(cons) * absA # scale absolute matrix
M = size(A, 1)
b = 0.1 * ones(M)
y_true = abs2.(A * xtrue[:]) .+ b
y_pos = rand.(Poisson.(y_true))
ATA = K*K*cons*Diagonal(sum(mask_list) .+ 1)
LAA = maximum(ATA)
ATA_inv = inv(ATA)

x0_rand = vec(rand(N, N))
B = LinearMapAA(x -> A'*(Diagonal(y_pos-b)*(A*x)), (N*N, N*N); T=ComplexF32)
x0_spectral = power_iter(B, x0_rand, 50)
α = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
x0_spectral = abs.(α * x0_spectral)
# α = sqrt(mean(y_pos - b) / mean(abs2.(A * x0_rand)))
# x0_rand = α * x0_rand

phase_shift = x -> sign.(vec(xtrue)' * vec(x))
nrmse = x -> iszero(x) ? 1.0 : (norm(x[:] - xtrue[:] .* phase_shift(x)) / norm(xtrue[:] .* phase_shift(x)))
phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
cost_func = (x,iter) -> [sum(phi.(A * x[:], y_pos, b)), time(), nrmse(x)]
# reg = maximum(2 .+ y_pos ./ (4 * b)) * LAA* 0.004

lf = x -> log10(max(x,1e-16))
grab = (o,i) -> hcat(o...)[i,:]
lg = (o,i) -> lf.(grab(o,i))



xout_wf_gau, cout_wf_gau = Wirtinger_flow(A,y_pos,b;
                gradhow = :gaussian, istrun = false, sthow = :fisher,
                xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
xout_wf_pois, cout_wf_pois = Wirtinger_flow(A,y_pos,b;
                gradhow = :poisson, istrun = false, sthow = :fisher,
                xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
xout_wf_pois_ls, cout_wf_pois_ls = Wirtinger_flow(A,y_pos,b;
                gradhow = :poisson, istrun = false, sthow = :lineser,
                xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
xout_gs, cout_gs = Gerchberg_saxton_dft(A,ATA_inv,y_pos,b;
            xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
xout_lsmm, cout_lsmm = LSMM(A,y_pos,b;
            curvhow = :imp, xhow = xhow, updatehow =:cg,
            x0 = x0_spectral, niter = niter, fun = cost_func)
xout_admm, cout_admm = ADMM_dft(A,ATA_inv,y_pos,b;
            phow =:adaptive, xhow = xhow, x0 = x0_spectral, ρ = 16,
            niter = niter, fun = cost_func)
xout_lsmm_ODWT, cout_lsmm_ODWT = LSMM_ODWT(A,y_pos,b,LAA;
            curvhow = :imp, xhow = xhow, reg = 32,
            x0 = x0_spectral, niter = niter, ninner = 10, fun = cost_func)
xout_admm_ODWT, cout_admm_ODWT = ADMM_ODWT_dft(A,ATA_inv,y_pos,b;
            phow =:adaptive, reg = 32, xhow = xhow,
            x0 = x0_spectral, ρ = 16, niter = niter, fun = cost_func)

save("./A_fft_shepplogan/A_FFT_x1_spectral_"*string(xhow)*"_mask="*string(num_mask)*".jld",
                    "xout_wf_gau", xout_wf_gau,
                    "cout_wf_gau", cout_wf_gau,
                    "xout_wf_pois", xout_wf_pois,
                    "cout_wf_pois", cout_wf_pois,
                    "xout_gs", xout_gs,
                    "cout_gs", cout_gs,
                    "xout_lsmm", xout_lsmm,
                    "cout_lsmm", cout_lsmm,
                    "xout_admm", xout_admm,
                    "cout_admm", cout_admm,
                    "xout_lsmm_ODWT", xout_lsmm_ODWT,
                    "cout_lsmm_ODWT", cout_lsmm_ODWT,
                    "xout_admm_ODWT", xout_admm_ODWT,
                    "cout_admm_ODWT", cout_admm_ODWT)
