using LinearAlgebra
using PolynomialRoots
"""
f(x, v, η; yi, bi, ρ) = sum[(|vi|^2 + bi) - yi*log(|vi|^2 + bi) + ρ/2 * (|vi - ai'x + η|^2 - |η|^2)]
For bi ≠ 0
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
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
function poisson_admm_bn0(A::AbstractMatrix{<:Number},
                      y::AbstractVector{<:Number},
                      b::AbstractVector{<:Number};
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

        x = copy(x0)

        v = A * x
        η = zeros(M)
        out = Array{Any}(undef, niter+1)
        out[1] = fun(x,0)


        for iter = 1:niter
            old_v = v
            # For v update
            phase_v = sign.(A*x - η)
            # absolute value of v update
            t = abs.(A*x - η)
            c1 = - (ρ .* t) ./ (2 + ρ)
            c2 = ((2 .* b) - (2 .* y) + (ρ .* b)) ./ (2 + ρ)
            c3 = - ((ρ .* b) .* t) ./ (2 + ρ)

            root_cubic = cubic.(c1, c2, c3)
            abs_v = select_root.(root_cubic, y, abs.(A * x), b, ρ * ones(M))
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

function select_root(roots, yi, abs_ai_t_xi, bi, ρ)
    cost_fun = (vi, yi, abs_ai_t_xi, bi) -> (abs2(vi) + bi) - yi * log(abs2(vi) + bi) + ρ/2 * (vi - abs_ai_t_xi)^2
    pos_roots = roots.>0
    if count(pos_roots) == 3
        v1 = cost_fun(roots[1], yi, abs_ai_t_xi, bi)
        v2 = cost_fun(roots[2], yi, abs_ai_t_xi, bi)
        v3 = cost_fun(roots[3], yi, abs_ai_t_xi, bi)
        return roots[argmin(v1, v2, v3)]
    else
        return maximum(roots[pos_roots])
    end
end

function cubic(a, b, c)
#
# This function extracts the real roots of the cubic equation
# x^3+a*x^2+b*x+c = 0.
#
# Check the discriminant condition.
#
   q = (a*a - 3.0 * b)/ 9.0
   r = (2.0 * a^3 - 9.0 * a * b + (3.0 * 9.0) * c)/ (9.0 * 6.0)
   if r^2 < q^3
#
# Three real roots.
#
      theta = acos(r / sqrt(q^3))
      return [-2.0 * sqrt(q) * cos(theta / 3.0) - a / 3.0
      -2.0 * sqrt(q) * cos((theta + 2.0 * pi) / 3.0) - a / 3.0
      -2.0 * sqrt(q) * cos((theta - 2.0 * pi) / 3.0) - a / 3.0]
   else
#
# A single real root.
#
      d = -sign(r) * cbrt(abs(r) + sqrt(r * r - q^3))
      if abs(d) <= 0.0
         e = 0.0
      else
         e = q/d
      end
      return [d + e - a / 3.0, NaN, NaN]
   end
end # function cubic
