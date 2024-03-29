# Wirtinger_flow.jl
"""
This code implements unregularized WF method for Poisson phase retrieval
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: M x 1, background counts
Optional Input:
`x0`: initial estimate of x
`niter`: number of outer iterations, default is 100
`gradhow`: gradient descent method, ":poisson" or ":gaussian"
`sthow`: method of choosing step size, ":fisher" or ":lineser"
`xhow`: type of x, ":real" or ":complex"
`istrun`: if truncate the gradient
`mustep`: parameter for the line search method (for step size)
`mushrink`: parameter for the line search method (for step size)
`trunreg`: parameter for truncated WF
`fun` User-defined function to be evaluated with two arguments `(x,iter)`
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
                        mushrink = 2,
                        trunreg::Real = 25,
                        ref = nothing,
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

    if isnothing(ref)
        ref = zeros(eltype(x0), size(x0))
    end

    out[1] = fun(x0,0)
    x = copy(x0)

    Ax = A * (x + vec(ref))
    if gradhow ==:gaussian
        cost_fun = (Ax, y, b) -> norm(max.(y-b,0) - abs2.(Ax))^2
    elseif gradhow ==:poisson
        phi = (v, yi, bi) -> (abs2(v) + bi) - yi * log(abs2(v) + bi) # (v^2 + b) - yi * log(v^2 + b)
        cost_fun = (Ax, y, b) -> sum(phi.(Ax, y, b))
    else
        throw("unknown gradhow")
    end

    # num_idx = 0
    for iter = 1:niter
        # First compute A * x
        if istrun
            idx = (abs.(y - abs2.(Ax)) .<= trunreg * mean(abs.(y - abs2.(Ax))) * abs2.(Ax) / norm(x))
            # @show sum(idx)
            # num_idx += sum(idx)
            if isa(A, Array)
                B = (@view A[idx, :])
            else
                B = LinearMapAA(x -> (A * (x + vec(ref)))[idx], y -> A' * embed(y, idx), (sum(idx), N); T = ComplexF32)
            end
            Bx = B * (x + vec(ref))
            ∇f, fisher = cal_grad(B, Bx, y[idx], b[idx], gradhow, xhow)
        else
            ∇f, fisher = cal_grad(A, Ax, y, b, gradhow, xhow)
        end

        if istrun
            Bdk = B * ∇f
        else
            Adk = A * ∇f
        end

        if sthow === :fisher
            if istrun
                D = Diagonal(fisher.(Bx, b[idx]))
                μ = - norm(∇f)^2 / real(Bdk' * D * Bdk)
                if isnan(μ)
                    @warn("break due to nan!")
                    break
                end

                x += μ * ∇f
                Bx += μ * Bdk
            else
                D = Diagonal(fisher.(Ax, b))
                μ = - norm(∇f)^2 / real(Adk' * D * Adk)
                if isnan(μ)
                    @warn("break due to nan!")
                    break
                end

                x += μ * ∇f
                Ax += μ * Adk
            end
        elseif sthow === :lineser
            if istrun
                μ = 1
                x_old = copy(x)
                Bx_old = copy(Bx)
                cost_Bx_old = cost_fun(Bx_old, y[idx], b[idx])

                x_new = x - μ * ∇f
                Bx_new = Bx - μ * Bdk
                mu_grad_f = mustep * norm(∇f, 2)^2
                # cost fun should take A * x0 as input.
                # Tuning param input args
                while cost_fun(Bx_new, y[idx], b[idx]) > cost_Bx_old - μ * mu_grad_f
                    μ = μ / mushrink
                    x_new = x_old - μ * ∇f
                    Bx_new = Bx_old - μ * Bdk # Find a suitable μ
                end
                x = copy(x_new)
                Bx = B * (x + vec(ref))
            else
                μ = 1
                x_old = copy(x)
                Ax_old = copy(Ax)
                cost_Ax_old = cost_fun(Ax_old, y, b)

                x_new = x - μ * ∇f
                Ax_new = Ax - μ * Adk
                mu_grad_f = mustep * norm(∇f, 2)^2
                # cost fun should take A * x0 as input.
                # Tuning param input args
                while cost_fun(Ax_new, y, b) > cost_Ax_old - μ * mu_grad_f
                    μ = μ / mushrink
                    x_new = x_old - μ * ∇f
                    Ax_new = Ax_old - μ * Adk # Find a suitable μ
                end
                x = copy(x_new)
                Ax = A * (x + vec(ref))
            end

        elseif sthow === :empir
            if istrun
                μ = - min(1 - exp(- iter / 330), 0.4)
                x += μ * ∇f
                Bx += μ * Bdk
            else
                μ = - min(1 - exp(- iter / 330), 0.4)
                x += μ * ∇f
                Ax += μ * Adk
            end

        elseif sthow === :optim_gau
            # calculate coefficients for the cubic equation
            if istrun
                u = real(conj.(Bx .+ b[idx]) .* Bdk)
                r = abs2.(Bx .+ b[idx]) .- y[idx]
                c3 = sum(abs.(Bdk) .^ 4)
                c2 = -3 * sum(u .* abs2.(Bdk))
                c1 = sum(r .* abs2.(Bdk) .+ 2 * u .^ 2)
                c0 = - sum(u .* r)
                roots = - cubic(c2 / c3, c1 / c3, c0 / c3)
                μ = select_root(roots, cost_fun, Bx, Bdk, y[idx], b[idx])
                # @show μ
                x += μ * ∇f
                Bx += μ * Bdk
            else
                u = real(conj.(Ax .+ b) .* Adk)
                r = abs2.(Ax .+ b) .- y
                c3 = sum(abs.(Adk) .^ 4)
                c2 = -3 * sum(u .* abs2.(Adk))
                c1 = sum(r .* abs2.(Adk) .+ 2 * u .^ 2)
                c0 = - sum(u .* r)
                roots = - cubic(c2 / c3, c1 / c3, c0 / c3)
                μ = select_root(roots, cost_fun, Ax, Adk, y, b)
                # @show μ
                x += μ * ∇f
                Ax += μ * Adk
            end

        else
            throw("unknown sthow!")
        end

        out[iter + 1] = fun(x, iter)
    end

    return x, out
    # return x, num_idx/(M * niter), out
end

function cal_grad(sys, sysx, y, b, gradhow, xhow)
    if gradhow ===:gaussian
        fisher = (vi, bi) -> 16 * abs2(vi) * (abs2(vi) + bi)
        grad_phi = (v, yi, bi) -> (abs2(v) - max(yi - bi, 0)) * v
        if xhow ===:real
            ∇f = 4 * real(sys' * grad_phi.(sysx, y, b))
        elseif xhow ===:complex
            ∇f = 4 * sys' * grad_phi.(sysx, y, b)
        else
            throw("unknown xhow")
        end
    elseif gradhow ===:poisson
        fisher = (vi, bi) -> 4 * abs2(vi) / (abs2(vi) + bi)
        grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi))
        if xhow ===:real
            ∇f = real(sys' * grad_phi.(sysx, y, b))
        elseif xhow ===:complex
            ∇f = sys' * grad_phi.(sysx, y, b)
        else
            throw("unknown xhow")
        end
    else
        throw("unknown gradhow")
    end
    return ∇f, fisher
end

function select_root(roots, cost_fun::Function, Ax, Adk, y, b)
    if any(isnan.(roots))
        return roots[1]
    else
        Ax1 = Ax .+ roots[1] * Adk
        Ax2 = Ax .+ roots[2] * Adk
        Ax3 = Ax .+ roots[3] * Adk
        cf1 = cost_fun(Ax1, y, b)
        cf2 = cost_fun(Ax2, y, b)
        cf3 = cost_fun(Ax3, y, b)
        return roots[argmin([cf1, cf2, cf3])]
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
