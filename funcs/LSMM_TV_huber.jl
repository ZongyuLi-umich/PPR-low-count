# LSMM_TV_huber.jl
include("ncg_phase.jl")
"""
This code implements regularized MM (with TV regularizer)
approximated by a Huber function.
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
Optional Input:
`curvhow`: curvature for the quadratic majorizer
`xhow`: type of x, ":real" or ":complex"
`x0`: initial estimate of x
`reg1`: regulairzer parameter for Huber function
`reg2`: regulairzer parameter for Huber function
`niter`: number of outer iterations, default is 100
`fun` User-defined function to be evaluated with two arguments `(x,iter)`
It is evaluated at `(x0,0)` and then after each iteration.
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function LSMM_TV_huber(A::AbstractMatrix{<:Number},
                y::AbstractVector{<:Number},
                b::AbstractVector{<:Number};
                curvhow::Symbol = :max,
                xhow::Symbol = :real,
                x0 = nothing,
                reg1 = 1,
                reg2 = 1,
                niter = 100,
                fun::Function = (x,iter) -> undef)

        M, N = size(A)
        cost_out = zeros(niter + 1)
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
        grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi)) # Here we know x is real
        curv_phi = (v, yi, bi) -> 2 + 2 * yi * (abs2(v) - bi) / (abs2(v) + bi)^2
        out[1] = fun(x0,0)

        soft(v, reg) = sign(v) * max(abs(v) - reg, 0)
        curv_huber = (t, α) -> abs(t) > α ? α/abs(t) : 1
        # T = spdiagm(0 => -ones(N-1), 1 => ones(N-1))[1:end-1,:]
        T = LinearMapAA(x -> diff(x), y -> TV_adj(y), (N-1, N); T=Float64)

        x = copy(x0)
        xk = copy(x0)
        z = T * x

        if curvhow === :max
                W = Diagonal(I(M) * (2 .+ y ./ (4 * b))) # precompute W
                AWA = A'*W*A
                for iter = 1:niter
                        xk = copy(x)
                        ∇q = x -> AWA * (x - xk) + A' * grad_phi.(A * xk, y, b) + reg1 * T' * (grad_huber.(T * x, reg2))
                        x, _ = ncg_phase(I(N), ∇q, xk;
                                        niter = 3,
                                        W = LinearMapAA(x -> AWA * x + reg1 * T' * (Diagonal(curv_huber.(T*x,reg2)) * (T * x)), (N,N); T=ComplexF32),
                                        xhow = xhow)
                        out[iter + 1] = fun(x, iter)
                end
        elseif curvhow === :imp
                for iter = 1:niter
                        s = A * x
                        gs = (b .+ sqrt.(b.^2 + b .* abs2.(s))) ./ abs.(s)
                        W = Diagonal(curv_phi.(gs, y, b))
                        AWA = A'*W*A
                        ∇q = x -> AWA * (x - xk) + A' * grad_phi.(A * xk, y, b) + reg1 * T' * (grad_huber.(T * x, reg2))
                        xk = copy(x)
                        x, _ = ncg_phase(I(N), ∇q, xk;
                                        niter = 3,
                                        W = LinearMapAA(x -> AWA * x + reg1 * T' * (Diagonal(curv_huber.(T*x, reg2)) * (T * x)), (N,N); T=ComplexF32),
                                        xhow = xhow)
                        out[iter + 1] = fun(x, iter)
                end
        else
                throw("unknown curvhow")
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
