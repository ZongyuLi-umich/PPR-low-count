include("utils.jl")
nrmse(x, xtrue) = round(100 * norm(vec(x) - vec(xtrue)) / norm(xtrue); digits=1)
## Gaussian
xtrue = load("../test-data-2D/4-test-images.jld2")["x1"]
N = size(xtrue, 1)
N2 = N^2
avg_count = 0.25
M = 80000
A = sqrt(1/2) * (randn(M, N2) + im * randn(M, N2)) # Each iteration, A is different.
cons = avg_count / mean(abs2.(A * vec(xtrue)))
A = sqrt(cons) * A # scale matrix A
b = 0.1 * ones(size(A, 1))
y_true = abs2.(A * vec(xtrue)) .+ b
y_pos = rand.(Poisson.(y_true))
# Define x0
x0_rand = randn(N2)
# global x0_spectral = power_iter(A'*Diagonal(y_pos-b)*A, x0_rand, 50)
x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)
scale_factor = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
x0_spectral_gau = deepcopy(reshape(abs.(scale_factor * x0_spectral), N, N))
nrmse_x0_spectral_gau = nrmse(x0_spectral_gau, xtrue)
## mask DFT
xtrue = load("../test-data-2D/4-test-images.jld2")["x2"]
N = size(xtrue, 1)
N2 = N^2
avg_count = 0.25
num_mask = 20
K = 2 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]
A1 = LinearMapAA(x -> vec(fft(pad(reshape(x, N, N)))),
                y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
                (K*K, N*N), (name="fft2D",), T=ComplexF32)
A = A1
mask_list = Array{AbstractArray{<:Number}, 1}(undef, num_mask)

for i = 1:num_mask
    mask = rand(N) .< 0.5
    A_temp = LinearMapAA(x -> vec(fft(pad(reshape(x, N, N) .* mask))),
                        y -> (K*K) * vec(unpad(ifft(reshape(y, K, K))) .* mask),
                        (K*K, N*N), (name="fft2D",), T=ComplexF32)
    A = vcat(A, A_temp)
    mask_list[i] = vec(mask)
end

cons = avg_count / mean(abs2.(A * vec(xtrue)))
A = sqrt(cons) * A # scale matrix A

b = 0.1 * ones(size(A, 1))
y_true = abs2.(A * vec(xtrue)) .+ b
y_pos = rand.(Poisson.(y_true))
# Define x0
x0_rand = randn(N2)
x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)

scale_factor = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
x0_spectral_mask_dft = deepcopy(reshape(abs.(scale_factor * x0_spectral), N, N))
nrmse_x0_spectral_mask_dft = nrmse(x0_spectral_mask_dft, xtrue)
## canon DFT
xtrue = load("../test-data-2D/4-test-images.jld2")["x3"]
ref = load("../test-data-2D/4-test-images.jld2")["ref_x3"]
N = size(xtrue, 1)
N2 = N^2
avg_count = 0.25
K = 4 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]
A = LinearMapAA(x -> vec(fft(pad(reshape(x, N, N)))),
                y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
                (K*K, N*N), (name="fft2D",), T=ComplexF32)

cons = avg_count / mean(abs2.(A * vec(xtrue)))
A = sqrt(cons) * A # scale matrix A

b = 0.1 * ones(size(A, 1))
y_true = abs2.(A * vec(xtrue + ref)) .+ b
y_pos = rand.(Poisson.(y_true))
# Define x0
x0_rand = randn(N2)
x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)

scale_factor = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
x0_spectral_canon_dft = deepcopy(reshape(abs.(scale_factor * x0_spectral), N, N))
nrmse_x0_spectral_canon_dft = nrmse(x0_spectral_canon_dft, xtrue)
## ETM
xtrue = load("../test-data-2D/4-test-images.jld2")["x4"]
N = size(xtrue, 1)
N2 = N^2
file = matopen("../AmpSLM_16x16/A_prVAMP.mat")
A = read(file, "A")
close(file)
M = size(A, 1)
cons = avg_count / mean(abs2.(A * vec(xtrue)))
A = sqrt(cons) * A # scale matrix A
b = 0.1 * ones(size(A, 1))
y_true = abs2.(A * vec(xtrue)) .+ b
y_pos = rand.(Poisson.(y_true))
# Define x0
x0_rand = randn(N2) + im * randn(N2)
# global x0_spectral = power_iter(A'*Diagonal(y_pos-b)*A, x0_rand, 50)
x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)
scale_factor = sqrt((y_pos-b)' * abs2.(A * x0_spectral)) / (norm(A * x0_spectral, 4)^2)
x0_spectral_etm = deepcopy(reshape(scale_factor * x0_spectral, N, N))
nrmse_x0_spectral_etm = nrmse(x0_spectral_etm, xtrue)
##
jim(x0_spectral_gau, "Spectral init, NRMSE=$nrmse_x0_spectral_gau%", xlims=(1,size(x0_spectral_gau,1)))
savefig("../result/spectral_init_gau.pdf")
jim(x0_spectral_mask_dft, "Spectral init, NRMSE=$nrmse_x0_spectral_mask_dft%", xlims=(1,size(x0_spectral_mask_dft,1)))
savefig("../result/spectral_init_mask_dft.pdf")
jim(x0_spectral_canon_dft, "Spectral init, NRMSE=$nrmse_x0_spectral_canon_dft%", xlims=(1,size(x0_spectral_canon_dft,1)))
savefig("../result/spectral_init_canon_dft.pdf")
jim(x0_spectral_etm, "Spectral init, NRMSE=$nrmse_x0_spectral_etm%", xlims=(1,size(x0_spectral_etm,1)))
savefig("../result/spectral_init_etm.pdf")
