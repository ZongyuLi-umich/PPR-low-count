"""
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
Optional Input:
`x0`: initial guess
`niter`: number of iterations
`gradhow`: gradient descent method, ":poisson" or ":gaussian"
`sthow`: method of choosing step size, ":fisher" or ":lineser"
`istrun`: if using truncated version
`trunreg`: truncated parameter
`xhow`: x is real or complex
`mustep`: parameter for inner loop
`fun` User-defined function to be evaluated with two arguments `(x,iter).
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function Wirtinger_flow(A::AbstractMatrix{<:Number},
                        y::AbstractVector{<:Number},
                        b::AbstractVector{<:Number};
                        x0 = nothing,
                        niter = 100,
                        gradhow::Symbol = :poisson,
                        sthow::Symbol = :fisher,
                        xhow::Symbol = :real,
                        istrun = false,
                        mustep = 0.01,
                        trunreg::Real = 25,
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
    x = copy(x0)
    Ax = A * x
    if gradhow ==:gaussian
        cost_fun = x -> norm(max.(y-b,0) - abs2.(A * x))^2
    elseif gradhow ==:poisson
        phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
        cost_fun = x -> sum(phi.(A * x, y, b))
    else
        throw("unknown gradhow")
    end

    for iter = 1:niter
        # First compute A * x
        if istrun
            idx = abs.(y - abs2.(Ax)) .<= trunreg * mean(abs.(y - abs2.(Ax))) * abs2.(Ax) / norm(x)
            B = LinearMapAA(x -> (A * x)[idx], y -> A' * embed(y, idx), (sum(idx), N); T = ComplexF32)
            ∇f, fisher = cal_grad(B, x, y[idx], b[idx], gradhow, xhow)
        else
            ∇f, fisher = cal_grad(A, x, y, b, gradhow, xhow)
        end

        if sthow === :fisher
            D = Diagonal(fisher.(Ax, b))
            Adk = A * ∇f
            μ = - norm(∇f)^2 / real(Adk' * D * Adk)
            x_old = copy(x)
            x += μ * ∇f
            Ax += μ * Adk
        elseif sthow === :lineser
            μ = 1
            x_old = copy(x)
            x_new = x - μ * ∇f
            # cost fun should take A * x0 as input.
            # Tuning param input args
            while cost_fun(x_new) > cost_fun(x_old) - mustep * μ * norm(∇f, 2)^2
                μ = μ / 2
                x_new = x_old - μ * ∇f # Find a suitable μ
            end
            x = x_new
            Ax = A * x
        else
            throw("unknown sthow!")
        end

        out[iter + 1] = fun(x, iter)
    end
    return x, out
end

function cal_grad(sys, x, y, b, gradhow, xhow)
    if gradhow ===:gaussian
        fisher = (vi, bi) -> 16 * abs2(vi) * (abs2(vi) + bi)
        grad_phi = (v, yi, bi) -> (abs2(v) - max(yi - bi, 0)) * v
        if xhow ==:real
            ∇f = 4 * real(sys' * grad_phi.(sys * x, y, b))
        elseif xhow ==:complex
            ∇f = 4 * sys' * grad_phi.(sys * x, y, b)
        else
            throw("unknown xhow")
        end
    elseif gradhow ===:poisson
        fisher = (vi, bi) -> 4 * abs2(vi) / (abs2(vi) + bi)
        grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
        if xhow ==:real
            ∇f = real(sys' * grad_phi.(sys * x, y, b))
        elseif xhow ==:complex
            ∇f = sys' * grad_phi.(sys * x, y, b)
        else
            throw("unknown xhow")
        end
    else
        throw("unknown gradhow")
    end
    return ∇f, fisher
end
