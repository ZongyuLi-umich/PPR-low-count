include("ncg_phase.jl")
"""
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: background counts
Optional Input:
`x0`: initial guess
`niter`: number of iterations, default = 100.
`ninner`: number of inner iterations.
`curvhow`: curvature for the quadratic majorizer
`reg1`: regulairzer parameter β
`reg2`: regulairzer parameter α
`xhow`: x is real or complex
`fun` User-defined function to be evaluated with two arguments `(x,iter).
It is evaluated at `(x0,0)` and then after each iteration.
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function LSMM_TV(A::AbstractMatrix{<:Number},
                y::AbstractVector{<:Number},
                b::AbstractVector{<:Number};
                curvhow::Symbol = :max,
                xhow::Symbol = :real,
                x0 = nothing,
                reg1 = 1,
                reg2 = 1,
                niter = 100,
                ninner = 5,
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

        soft(v, reg) = sign(v) * max(abs(v) - reg, 0)
        T = spdiagm(0 => -ones(N-1), 1 => ones(N-1))[1:end-1,:]

        x = copy(x0)
        xk = copy(x0)
        z = T * x

        ∇q = x -> AWA * (x - xk) + A' * grad_phi.(A * xk, y, b) + reg1 * T' * (T * x - z)

        if curvhow === :max
                W = Diagonal(I(M) * (2 .+ y ./ (4 * b))) # precompute W
                AWA = A'*W*A
                for iter = 1:niter
                        xk = copy(x)
                        for inner = 1:ninner
                                xi = copy(x)
                                x, _ = ncg_phase(I(N), ∇q, xi; niter = 3, W = AWA + reg1 * T' * T, xhow = xhow)
                                z = soft.(T * x, reg2)
                        end
                        out[iter + 1] = fun(x, iter)
                end
        elseif curvhow === :imp
                for iter = 1:niter
                        s = A * x
                        gs = (b .+ sqrt.(b.^2 + b .* abs2.(s))) ./ abs.(s)
                        W = Diagonal(curv_phi.(gs, y, b))
                        AWA = A'*W*A
                        xk = copy(x)
                        for inner = 1:ninner
                                xi = copy(x)
                                x, _ = ncg_phase(I(N), ∇q, xi; niter = 3, W = AWA + reg1 * T' * T, xhow = xhow)
                                z = soft.(T * x, reg2)
                        end
                        out[iter + 1] = fun(x, iter)
                end
        else
                throw("unknown curvhow")
        end
        return x, out
end
