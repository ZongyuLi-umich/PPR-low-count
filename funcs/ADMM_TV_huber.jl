# ADMM_TV_huber.jl
include("ncg_phase.jl")
"""
This code implements regularized ADMM method with TV regularizer,
approximated by a Huber function
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
Optional Input:
`x0`: initial estimate of x
`niter`: number of outer iterations, default is 100
`ρ`: AL penalty parameter
`reg1`: regulairzer parameter of the Huber function
`reg2`: regulairzer parameter of the Huber function
`xhow`: type of x, ":real" or ":complex"
`phow`: strategy of updating AL penalty parameter, ":constant" or ":adaptive"
`updatehow`: method of updating `x`, ":bs" or ":cg" (see our paper for details)
`fun`: User-defined function to be evaluated with two arguments `(x,iter)`
It is evaluated at `(x0,0)` and then after each iteration.
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function ADMM_TV_huber(A::AbstractMatrix{<:Number},
                      y::AbstractVector{<:Number},
                      b::AbstractVector{<:Number};
                      x0::AbstractVector{<:Number} = nothing,
                      niter::Int = 100,
                      ρ::Real = 16,
                      reg1::Real = 1,
                      reg2::Real = 1,
                      xhow::Symbol = :real,
                      phow::Symbol = :constant,
                      updatehow::Symbol = :bs,
                      fun::Function = (x,iter) -> undef)
        (M, N) = size(A)
        if isnothing(x0)
            if xhow === :real
                x0 = randn(N)
            elseif xhow === :complex
                x0 = sqrt(var/2) * (randn(N) + im * randn(N))
            else
                throw("unknown xhow")
            end
        end

        x = copy(x0)
        # T = spdiagm(0 => -ones(N-1), 1 => ones(N-1))[1:end-1,:]
        T = LinearMapAA(x -> diff(x), y -> TV_adj(y), (N-1, N); T=Float64)
        soft(v, reg) = sign(v) * max(abs(v) - reg, 0)
        curv_huber = (t, α) -> abs(t) > α ? α/abs(t) : 1
        v = A * x
        η = zeros(M)
        z = T * x
        out = Array{Any}(undef, niter+1)
        out[1] = fun(x,0)

        for iter = 1:niter
            Ax_old = A * x
            # For v update
            old_v = v
            phase_v = sign.(Ax_old - η)
            if iszero(b)
                absv_func = (t, yi) -> (ρ*t + sqrt(ρ^2 * t^2 + 8 * yi * (2 + ρ))) / (2 * (2 + ρ))
                abs_v = absv_func.(abs.(Ax_old - η), y)
            else
                # absolute value of v update
                t = abs.(Ax_old - η)
                c1 = - (ρ * t) / (2 + ρ)
                c2 = ((2 * b) - (2 * y) + (ρ * b)) / (2 + ρ)
                c3 = - ((ρ * b) .* t) / (2 + ρ)

                root_cubic = cubic.(c1, c2, c3)
                abs_v = select_root.(root_cubic, y, t, b, ρ, η)
            end

            v = abs_v .* phase_v

            # For x update, Replace by cg + L1
            ∇q = x -> ρ * A' * (A * x - v - η) + reg1 * T' * (grad_huber.(T * x, reg2))
            xk = copy(x)
            x, _ = ncg_phase(I(N), ∇q, xk;
                            niter = 3,
                            W = LinearMapAA(x -> ρ*A'*(A*x) + reg1 * T' * (Diagonal(curv_huber.(T*x, reg2)) * (T * x)), (N,N); T=ComplexF32),
                            xhow = xhow)

            Ax_new = A * x

            # For η update
            η = η + (v - Ax_new)
            out[iter + 1] = fun(x, iter)

            if iter % 10 == 0 # examine AL penalty parameter every 10 iterations
                if phow == :constant
                    continue
                elseif phow == :adaptive
                    rk = v - Ax_new # primal residual
                    sk = ρ * A' * (v - old_v) # dual residual
                    if norm(rk) > 10 * norm(sk)
                        ρ = 2 * ρ
                    elseif norm(sk) > 100 * ρ * norm(rk)
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

function select_root(roots, yi, ti, bi, ρ, etai)
    cost_fun = (vi) -> vi^2 + bi - yi * log(vi^2 + bi) + ρ / 2 * ((vi - ti)^2 - abs2(etai))
    if any(isnan.(roots))
        return roots[1]
    else
        return roots[argmin(cost_fun.(roots))]
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
