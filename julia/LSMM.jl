using LinearAlgebra
using Statistics
include("ncg_phase.jl")
"""
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: background counts
Optional Input:
`x0`: initial guess
`niter`: number of iterations, default = 100.
`curvhow`: curvature for the quadratic majorizer
`xhow`: x is real or complex
`updatehow`: update using black slash or CG
`fun` User-defined function to be evaluated with two arguments `(x,iter).
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
                        if xhow === :real
                                if updatehow === :bs
                                        x = x - real(A' * W * A) \ real(A' * grad_phi.(A * x, y, b))
                                elseif updatehow === :cg
                                        xk = copy(x)
                                        ∇f = t ->  W * t .+ grad_phi.(A * xk, y, b)
                                        res, _ = ncg_phase(A, ∇f, zeros(N); niter = 2*(iter<10)+1, W = W, xhow =:real)
                                        x = xk + res
                                else
                                        throw("unknown updatehow")
                                end
                        elseif xhow === :complex
                                if updatehow === :bs
                                        x = x - (A' * W * A) \ (A' * grad_phi.(A * x, y, b))
                                elseif updatehow === :cg
                                        xk = copy(x)
                                        ∇f = t ->  W * t .+ grad_phi.(A * xk, y, b)
                                        res, _ = ncg_phase(A, ∇f, zeros(N); niter = 2*(iter<10)+1, W = W, xhow =:complex)
                                        x = xk + res
                                else
                                        throw("unknown updatehow")
                                end
                        else
                                throw("unknown xhow")
                        end
                        out[iter + 1] = fun(x, iter)
                end
        elseif curvhow === :imp
                for iter = 1:niter
                        s = A * x
                        gs = (b .+ sqrt.(b.^2 + b .* abs2.(s))) ./ abs.(s)
                        W = curv_phi.(gs, y, b)
                        W = Diagonal(min.(W, mean(W) + 3 * std(W)))
                        if xhow === :real
                                if updatehow === :bs
                                        x = x - real(A' * W * A) \ real(A' * grad_phi.(s, y, b))
                                elseif updatehow === :cg
                                        xk = copy(x)
                                        ∇f = t -> W * t .+ grad_phi.(A * xk, y, b)
                                        res, _ = ncg_phase(A, ∇f, zeros(N); niter = 2*(iter<10)+1, W = W, xhow =:real)
                                        x = xk + res
                                else
                                        throw("unknown updatehow")
                                end
                        elseif xhow === :complex
                                if updatehow === :bs
                                        x = x - (A' * W * A) \ (A' * grad_phi.(s, y, b))
                                elseif updatehow === :cg
                                        xk = copy(x)
                                        ∇f = t ->  W * t .+ grad_phi.(A * xk, y, b)
                                        res, _ = ncg_phase(A, ∇f, zeros(N); niter = 2*(iter<10)+1, W = W, xhow =:complex)
                                        x = xk + res
                                else
                                        throw("unknown updatehow")
                                end
                        else
                                throw("unknown xhow")
                        end

                        out[iter + 1] = fun(x, iter)
                end
        else
                throw("unknown curvhow")

        end
        return x, out

end
