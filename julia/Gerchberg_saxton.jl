include("ncg_phase.jl")
"""
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
Optional Input:
`x0`: initial guess
`niter`: number of iterations
`updatehow`: update using black slash or CG
`xhow`: x is real or complex
`fun` User-defined function to be evaluated with two arguments `(x,iter).
It is evaluated at `(x0,0)` and then after each iteration.
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function Gerchberg_saxton(A::AbstractMatrix{<:Number},
                          y::AbstractVector{<:Number},
                          b::AbstractVector{<:Number};
                          x0 = nothing,
                          xhow::Symbol = :real,
                          updatehow::Symbol =:bs,
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
                    if updatehow === :bs
                            x = real(A'*A) \ real(A' * (Diagonal(sqrt.(max.(y-b,0))) * c))
                    elseif updatehow ===:cg
                            ∇f = t -> t - Diagonal(sqrt.(max.(y-b,0))) * c
                            xk = copy(x)
                            x, _ = ncg_phase(A, ∇f, xk; niter = 2*(iter<10)+1, xhow =:real)
                    else
                            throw("unknown updatehow")
                    end

            elseif xhow ===:complex
                    if updatehow === :bs
                            x = A' * A \ (A' * (Diagonal(sqrt.(max.(y-b,0))) * c))
                    elseif updatehow === :cg
                            ∇f = t -> t - Diagonal(sqrt.(max.(y-b,0))) * c
                            xk = copy(x)
                            x, _ = ncg_phase(A, ∇f, xk; niter = 2*(iter<10)+1, xhow =:complex)
                    else
                            throw("unknown updatehow")
                    end
            else
                throw("unknown xhow")
            end
            out[iter + 1] = fun(x, iter)
        end
        return x, out
end
