# Wirtinger_flow_ODWT.jl
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
`sthow`: method of choosing step size, ":fisher" or ":lineser"
`xhow`: type of x, ":real" or ":complex"
`reg1`: parameter for the Huber function
`reg2`: parameter for the Huber function
`fun` User-defined function to be evaluated with two arguments `(x,iter)`
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function Wirtinger_flow_ODWT(A::AbstractMatrix{<:Number},
                        y::AbstractVector{<:Number},
                        b::AbstractVector{<:Number};
                        x0 = nothing,
                        niter = 100,
                        xhow::Symbol = :real,
                        sthow::Symbol = :fisher,
                        reg1::Real = 32,
                        mustep = 0.01,
                        mushrink = 2,
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
    # T = LinearMapAA(x -> diff(x), y -> TV_adj(y), (N-1, N); T=Float64) # for 1D
    sn = Int(sqrt(N))
    # T = LinearMapAA(x -> diff2d_forw(x, sn, sn), y -> diff2d_adj(y, sn, sn), (2*sn*(sn-1), N); T=Float64) # for 2D
    T, scales, mfun = Aodwt((sn, sn); T = ComplexF64)
    out[1] = fun(x0,0)
    x = copy(x0)
    Ax = A * x
    Tx = vec(T * reshape(x, sn, sn))
    phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi)
    grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
    cost_fun = (Ax, Tx) -> sum(phi.(Ax, y, b)) + reg1 * norm(Tx, 1)
    fisher = (vi, bi) -> 4 * abs2(vi) / (abs2(vi) + bi)

    for iter = 1:niter
        if xhow === :real
            ∇f = real(A' * grad_phi.(A * vec(x), y_pos, b) + β * vec(T' * sign.(T * reshape(x, N, N))))
        elseif xhow === :complex
            ∇f = A' * grad_phi.(A * vec(x), y_pos, b) + β * vec(T' * sign.(T * reshape(x, N, N)))
        else
            throw("unknown xhow")
        end

        Adk = A * ∇f
        Tdk = vec(T * reshape(∇f, sn, sn))
        if sthow === :fisher
            # D1 = Diagonal(fisher.(Ax, b))
            D1 = sqrt.(fisher.(Ax, b))
            # D2 = Diagonal(curv_huber.(Tx, reg2))
            D2 = 1
            # μ = - norm(∇f)^2 / real(Adk' * D1 * Adk + reg1 * Tdk' * D2 * Tdk)
            μ = - norm(∇f)^2 / (norm(Adk .* D1)^2 + reg1 * norm(Tdk .* D2)^2)
            if isnan(μ)
                @warn("break due to nan!")
                break
            end
        elseif sthow === :lineser
            μ = -1
            x_old = copy(x)
            Ax_old = copy(Ax)
            Tx_old = copy(Tx)
            cost_old = cost_fun(Ax_old, Tx_old)

            x_new = x + μ * ∇f
            Ax_new = Ax + μ * Adk
            Tx_new = Tx + μ * Tdk
            mu_grad_f = mustep * norm(∇f, 2)^2
            # cost fun should take A * x0 as input.
            # Tuning param input args
            while cost_fun(Ax_new, Tx_new) > cost_old + μ * mu_grad_f
                μ = μ / mushrink
                x_new = x_old + μ * ∇f
                Ax_new = Ax_old + μ * Adk
                Tx_new = Tx_old + μ * Tdk # Find a suitable μ
            end
        else
            throw("unknown sthow")
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
