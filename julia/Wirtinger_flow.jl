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
This minimizes the cost function: f = ||y - |Ax|^2||_2^2
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function Wirtinger_flow(A::AbstractMatrix{<:Number},
                        y::AbstractVector{<:Number},
                        b::AbstractVector{<:Number};
                        x0 = nothing,
                        niter = 100,
                        gradhow::Symbol = :gaussian,
                        xhow::Symbol = :real,
                        fun::Function = (x, iter) -> undef)

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
    out[1] = fun(x0,0)
    if gradhow ==:gaussian
        cost_fun = x -> norm(max.(y-b,0) - abs2.(A*x))^2
    elseif gradhow ==:poisson
        phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
        grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
        cost_fun = x -> sum(phi.(A * x, y, b))
    else
        throw("unknown gradhow")
    end
    for iter = 1:niter
        if gradhow ==:gaussian
            if xhow ==:real
                ∇f = 4 * real(A' * Diagonal(abs2.(A * x0) - max.(y-b,0)) * A) * x0
            elseif xhow ==:complex
                ∇f = 4 * A' * Diagonal(abs2.(A * x0) - max.(y-b,0)) * A * x0
            else
                throw("unknown xhow")
            end
        elseif gradhow ==:poisson
            if xhow ==:real
                ∇f = real(A' * grad_phi.(A * x0, y, b))
            elseif xhow ==:complex
                ∇f = A' * grad_phi.(A * x0, y, b)
            else
                throw("unknown xhow")
            end
        else
            throw("unknown gradhow")
        end

        μ = 1
        old_x0 = x0
        new_x0 = x0 - μ * ∇f
        while cost_fun(new_x0) > cost_fun(old_x0) - 0.01 * μ * norm(∇f, 2)^2
            μ = μ / 2
            new_x0 = old_x0 - μ * ∇f # Find a suitable μ
        end
        x0 = new_x0
        out[iter + 1] = fun(x0, iter)
    end
    return x0, out
end
