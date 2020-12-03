using LinearAlgebra
"""
f(x, v, η; yi, ρ) = sum[(|vi|^2) - yi*log(|vi|^2) + ρ/2 * (|vi - ai'x + η|^2 - |η|^2)]
For bi = 0
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector

Optional Input:
`x0`: initial guess
`niter`: number of iterations
`ρ`: Lagrange penalty parameter; default = 1
`fun`: User-defined function to be evaluated with two arguments `(x,iter).
`xhow`: update x using real or complex formula, default = real.
It is evaluated at `(x0,0)` and then after each iteration.
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""

function poisson_admm_b0(A::AbstractMatrix{<:Number},
                      y::AbstractVector{<:Number};
                      x0::AbstractVector{<:Number} = nothing,
                      niter::Int = 100,
                      ρ::Real = 1,
                      xhow::Symbol = :real,
                      phow::Symbol = :constant,
                      fun::Function = (x,iter) -> undef)
        (M, N) = size(A)
        if isnothing(x0)
            if xhow == :real
                x0 = randn(N)
            elseif xhow == :complex
                x0 = sqrt(var/2) * (randn(N) + im * randn(N))
            else
                throw("unknown xhow")
            end
        end
        absv_func = (t, yi) -> (ρ*t + sqrt(ρ^2 * t^2 + 8 * yi * (2 + ρ))) / (2 * (2 + ρ))

        x = copy(x0)

        v = A * x
        η = zeros(M)
        out = Array{Any}(undef, niter+1)
        out[1] = fun(x,0)

        for iter = 1:niter
            old_v = v
            # For v update
            phase_v = sign.(A*x - η)
            abs_v = absv_func.(abs.(A*x - η), y)
            v = abs_v .* phase_v

            # For x update
            if xhow == :real
                x = real(A' * A) \ real(A' * v)
            elseif xhow == :complex
                x = (A' * A) \ (A' * v)
            else
                throw("unknown xhow")
            end

            # For η update
            η = η + (v - A * x)

            out[iter + 1] = fun(x, iter)

            if iter % 10 == 0 # examine AL penalty parameter every 10 iterations
                if phow == :constant
                    continue
                elseif phow == :adaptive
                    rk = v - A * x # primal residual
                    sk = ρ * A' * (v - old_v) # dual residual
                    if norm(rk) > 10 * norm(sk)
                        ρ = 2 * ρ
                    elseif norm(sk) > 10 * norm(rk)
                        ρ = ρ / 2
                    else
                        ρ = ρ
                    end
                else
                    throw("unknown phow")
                end
            end
        end
        return x, out
end
