include("utils.jl")
xhow = :real
avg_count = 0.25
niter = 200
cpt = "/n/calumet/x/zonyul/Poisson_phase_retri_2021/"
xtrue = load("./test-data-2D/UM-1817-shepplogan.jld2")["x1"]
N = size(xtrue, 1)
N2 = N^2
num_mask = 100
K = 2 * N - 1
pad = x -> vcat(hcat(x, zeros(eltype(x), N, K - N)), zeros(eltype(x), K - N, K))
unpad = x -> x[1:N, 1:N]

M = 80000
A_GAU = randn(M, N2) + im * randn(M, N2)

A1_DFT = LinearMapAA(
        x -> vec(fft(pad(reshape(x, N, N)))),
        y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
        (K*K, N*N), (name="fft2D",), T=ComplexF32)
A_DFT = A1_DFT
mask_list = Array{AbstractArray{<:Number}, 1}(undef, num_mask)

for i = 1:num_mask
    mask = rand(N) .< 0.5
    A_temp = LinearMapAA(
            x -> vec(fft(pad(reshape(x, N, N)))),
            y -> (K*K) * vec(unpad(ifft(reshape(y, K, K)))),
            (K*K, N*N), (name="fft2D",), T=ComplexF32)
    A_DFT = vcat(A_DFT, A_temp)
    mask_list[i] = vec(mask)
end

file = matopen("./AmpSLM_64x64/A_prVAMP.mat")
AA_ETM = read(file, "A")
close(file)
idx = 1:4:size(AA_ETM, 2)
A_ETM = AA_ETM[:, idx]

## now run the code
A = A_GAU
cons = avg_count / mean(abs2.(A * vec(xtrue)))
A = sqrt(cons) * A # scale matrix A
b = 0.1 * ones(size(A, 1))
y_true = abs2.(A * vec(xtrue)) .+ b
y_pos = rand.(Poisson.(y_true))
x0_rand = vec(rand(N, N))
x0_spectral = power_iter(A'*Diagonal(y_pos ./ (y_pos .+ 1))*A, x0_rand, 50)
α = sqrt(dot((y_pos-b), abs2.(A * x0_spectral))) / (norm(A * x0_spectral, 4)^2)
x0_spectral = abs.(α * x0_spectral)
function compare_fisher(x)
    Ax = A * x
    Ax2 = abs2.(Ax)
    fisher = 4 * Ax2 ./ (Ax2 .+ b)
    hessian = 2 .+ 2 * y_pos .* (Ax2 .- b) ./ ((Ax2 .+ b).^2)
    return fisher, hessian
end

fisher, hessian = compare_fisher(x0_spectral)
histogram(hessian, xlims = (-10, 10), ylims = (0, 1e4))

plot(histogram(hessian), color = :blue, label = "Hessian")
plot!(histogram(fisher), color = :red, legendfontsize = 12, label = "Fisher")
savefig("./result/fisher-vs-hessian.pdf")
# cost_func = (x,iter) -> [mean_fisher(x)]

# xout_wf_fisher, cout_wf_fisher = Wirtinger_flow(A,y_pos,b;
#         gradhow = :poisson, sthow = :fisher, istrun = false,
#         xhow = xhow, x0 = x0_spectral, niter = niter, fun = cost_func)
# bias = grab(cout_wf_fisher, 1)
# scatter(bias)
