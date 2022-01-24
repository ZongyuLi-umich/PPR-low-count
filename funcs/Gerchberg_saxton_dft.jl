# Gerchberg_saxton_dft.jl
"""
This code implements the Gerchberg Saxton method for (Gaussian) phase retrieval
for the case when the system matrix A is DFT, so A'A is a simple digonal matrix.
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`ATA_inv`: N x N (precomputed) diagonal matrix, inverse of A'A
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
Optional Input:
`x0`: initial estimate of x
`xhow`: type of x, ":real" or ":complex"
`niter`: number of outer iterations, default is 100
`fun` User-defined function to be evaluated with two arguments `(x,iter)`
It is evaluated at `(x0,0)` and then after each iteration.
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function Gerchberg_saxton_dft(A::AbstractMatrix{<:Number},
                          ATA_inv::AbstractMatrix{<:Number},
                          y::AbstractVector{<:Number},
                          b::AbstractVector{<:Number};
                          x0 = nothing,
                          xhow::Symbol = :real,
                          niter = 100,
                          fun::Function = (x,iter) -> undef)

        M, N = size(A)
        out = Array{Any}(undef, niter+1)
        if isnothing(x0)
            if xhow === :real
                x0 = randn(N)
            elseif xhow === :complex
                x0 = sqrt(var/2) * (randn(N) + im * randn(N))
            else
                throw("unknown xhow")
            end
        end
        out[1] = fun(x0,0)
        x = copy(x0)

        for iter = 1:niter
            c = sign.(A*x)
            if xhow === :real
                    x = real(ATA_inv) * real(A' * (Diagonal(sqrt.(max.(y-b,0))) * c))
            elseif xhow ===:complex
                    x = ATA_inv * (A' * (Diagonal(sqrt.(max.(y-b,0))) * c))
            else
                    throw("unknown xhow")
            end
            out[iter + 1] = fun(x, iter)
        end
        return x, out
end
