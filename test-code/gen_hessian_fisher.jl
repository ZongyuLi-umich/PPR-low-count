using LinearAlgebra
using Distributions: Poisson
using LaTeXStrings
m = 50
n = 10
x = randn(n)
A = randn(m, n) + im * randn(m, n)
b = 5
num_realization = 100
h = Vector(undef, num_realization)
f = Vector(undef, num_realization)
for i = 1:num_realization
    y = rand.(Poisson.(abs2.(A*x).+b))
    v = A*x
    h[i] = 2 .+ 2 * y .* (abs2.(v) .- b) ./ (abs2.(v) .+ b).^2
    f[i] = 4 * abs2.(v) ./ (abs2.(v) .+ b)
end
h_mean = zeros(m)
f_mean = zeros(m)
for i = 1:m
    h_mean[i] = mean([h[j][i] for j = 1:num_realization])
    f_mean[i] = mean([f[j][i] for j = 1:num_realization])
end

scatter(h_mean, label = "Hessian", color = :blue, xlabel = L"i", xlabelfontsize = 16, legendfontsize=16)
scatter!(f_mean, label = "Fisher", color = :red, legendfontsize=16)

for i = 1:num_realization
    scatter!(h[i], label = "", color = :blue, markeralpha = 0.1)
    scatter!(f[i], label = "", color = :red, markeralpha = 0.1)
end
current()
savefig("../result/fisher-vs-hessian.pdf")
