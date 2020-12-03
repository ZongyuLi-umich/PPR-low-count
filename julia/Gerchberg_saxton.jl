using LinearAlgebra
"""
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
Optional Input:
`x0`: initial guess
`niter`: number of iterations
`fun` User-defined function to be evaluated with two arguments `(x,iter).
It is evaluated at `(x0,0)` and then after each iteration.
This minimizes the cost function: f = ||Ax - diag(sqrt(y))*c||_2^2
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function Gerchberg_saxton(A::AbstractMatrix{<:Number},
                          y::AbstractVector{<:Number},
                          b::AbstractVector{<:Number};
                          x0 = nothing,
                          xhow::Symbol = :real,
                          niter = 100,
                          fun::Function = (x,iter) -> undef)

        M, N = size(A)
        out = Array{Any}(undef, niter+1)
        if isnothing(x0)
            if xhow == :real
                x0 = randn(N)
            elseif xhow == :complex
                x0 = sqrt(var/2) * (randn(N) + im * randn(N))
            else
                throw("unknown xhow")
            end
        end
        c = sign.(A*x0)
        out[1] = fun(x0,0)
        for iter = 1:niter
            c = sign.(A*x0)
            if xhow == :real
                x0 = real(A'*A) \ real(A' * (Diagonal(sqrt.(max.(y-b,0))) * c))
            elseif xhow ==:complex
                x0 = A' * A \ (A' * (Diagonal(sqrt.(max.(y-b,0))) * c))
            else
                throw("unknown xhow")
            end
            out[iter + 1] = fun(x0, iter)
        end
        return x0, out
end
