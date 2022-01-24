# Wirtinger_flow_huber.jl
"""
This code implements regularized WF method with TV regularizer
approximated by a Huber function.
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
Optional Input:
`x0`: initial estimate of x
`niter`: number of outer iterations, default is 100
`xhow`: type of x, ":real" or ":complex"
`reg1`: parameter for the Huber function
`reg2`: parameter for the Huber function
`fun` User-defined function to be evaluated with two arguments `(x,iter)`
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function Wirtinger_flow_huber(A::AbstractMatrix{<:Number},
                        y::AbstractVector{<:Number},
                        b::AbstractVector{<:Number};
                        x0 = nothing,
                        niter = 100,
                        xhow::Symbol = :real,
                        reg1::Real = 32,
                        reg2::Real = 0.5,
                        fun::Function = (x, iter) -> undef)

    M, N = size(A)
    out = Array{Any}(undef, niter+1)
    if isnothing(x0)
        if xhow == :real
            x0 = randn(N)
        elseif xhow == :complex
            x0 = sqrt(1/2) * (randn(N) + im * randn(N))
        else
            throw("unknown xhow")
        end
    end
    T = LinearMapAA(x -> diff(x), y -> TV_adj(y), (N-1, N); T=Float64)
    out[1] = fun(x0,0)
    x = copy(x0)
    Ax = A * x
    Tx = T * x
    phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi)
    grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
    cost_fun = (Ax, Tx) -> sum(phi.(Ax, y, b)) + reg1 * huber.(Tx, reg2)
    fisher = (vi, bi) -> 4 * abs2(vi) / (abs2(vi) + bi)
    curv_huber = (t, α) -> abs(t) > α ? α/abs(t) : 1
    for iter = 1:niter
        if xhow === :real
            ∇f = real(A' * grad_phi.(Ax, y, b) + reg1 * T' * (grad_huber.(Tx, reg2)))
        elseif xhow === :complex
            ∇f = A' * grad_phi.(Ax, y, b) + reg1 * T' * (grad_huber.(Tx, reg2))
        else
            throw("unknown xhow")
        end

        Adk = A * ∇f
        Tdk = T * ∇f

        D1 = Diagonal(fisher.(Ax, b))
        D2 = Diagonal(curv_huber.(Tx, reg2))
        μ = - norm(∇f)^2 / real(Adk' * D1 * Adk + reg1 * Tdk' * D2 * Tdk)
        if isnan(μ)
            @warn("break due to nan!")
            break
        end
        x += μ * ∇f
        Ax += μ * Adk
        Tx += μ * Tdk

        out[iter + 1] = fun(x, iter)
    end

    return x, out
end

function huber(t, α)
    if abs(t) < α
        return 1/2 * abs2(t)
    else
        return α * abs(t) - 1/2 * α^2
    end
end
function grad_huber(t, α)
    if abs(t) < α
        return t
    else
        return α * sign(t)
    end
end
