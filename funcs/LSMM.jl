# LSMM.jl
include("ncg_phase.jl")
"""
This code implements MM algorithm with quadratic majorizer for Poisson phase retrieval
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
Optional Input:
`curvhow`: curvature for the quadratic majorizer
`xhow`: type of x, ":real" or ":complex"
`updatehow`: method of updating `x`, ":bs" or ":cg" (see our paper for details)
`x0`: initial estimate of x
`niter`: number of outer iterations, default is 100
`fun` User-defined function to be evaluated with two arguments `(x,iter)`
It is evaluated at `(x0,0)` and then after each iteration.
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function LSMM(A::AbstractMatrix{<:Number},
                y::AbstractVector{<:Number},
                b::AbstractVector{<:Number};
                curvhow::Symbol = :max,
                xhow::Symbol = :real,
                updatehow::Symbol = :bs,
                x0 = nothing,
                niter = 100,
                fun::Function = (x,iter) -> undef)

        M, N = size(A)
        cost_out = zeros(niter + 1)
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
        grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi)) # Here we know x is real
        curv_phi = (v, yi, bi) -> 2 + 2 * yi * (abs2(v) - bi) / (abs2(v) + bi)^2
        out[1] = fun(x0,0)
        x = copy(x0)
        if curvhow === :max
                W = 2 .+ y ./ (4 * b) # precompute W
                W = Diagonal(min.(W, mean(W) + 3 * std(W)))
                for iter = 1:niter
                        if updatehow === :bs
                                if xhow === :real
                                        x = x - real(A' * W * A) \ real(A' * grad_phi.(A * x, y, b))
                                elseif xhow === :complex
                                        x = x - (A' * W * A) \ (A' * grad_phi.(A * x, y, b))
                                else
                                        throw("unknown xhow")
                                end
                        elseif updatehow === :cg
                                xk = copy(x)
                                ∇f = t ->  W * t .+ grad_phi.(A * xk, y, b)
                                res, _ = ncg_phase(A, ∇f, zeros(N); niter = 2*(iter<10)+1, W = W, xhow = xhow)
                                x = xk + res
                        else
                                throw("unknown updatehow")
                        end
                        out[iter + 1] = fun(x, iter)
                end
        elseif curvhow === :imp
                for iter = 1:niter
                        s = A * x
                        gs = (b .+ sqrt.(b.^2 + b .* abs2.(s))) ./ abs.(s)
                        W = curv_phi.(gs, y, b)
                        W = Diagonal(min.(W, mean(W) + 3 * std(W)))
                        if updatehow === :bs
                                if xhow === :real
                                        x = x - real(A' * W * A) \ real(A' * grad_phi.(s, y, b))
                                elseif xhow === :complex
                                        x = x - (A' * W * A) \ (A' * grad_phi.(s, y, b))
                                else
                                        throw("unknown xhow")
                                end
                        elseif updatehow === :cg
                                xk = copy(x)
                                ∇f = t -> W * t .+ grad_phi.(A * xk, y, b)
                                res, _ = ncg_phase(A, ∇f, zeros(N); niter = 2*(iter<10)+1, W = W, xhow = xhow)
                                x = xk + res
                        else
                                throw("unknown updatehow")
                        end
                        out[iter + 1] = fun(x, iter)
                end
        else
                throw("unknown curvhow")

        end
        return x, out
end
