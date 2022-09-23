# Wirtinger_flow_inplace.jl
"""
In-place version of unregularized Wirtinger flow algorithm
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

mutable struct WFState{Tx, Tm}
    x::Tx
    x_old::Tx
    Ax_old::Tx
    x_new::Tx
    Ax_new::Tx
    Ax::Tx
    work_Ax::Tx
    work_Ax1::Tx
    Agx::Tx
    g::Tx
    Adk::Tx
    fisher_vec::Tx
    cost_fun_vec::Tx
    fisher!::Function
    grad_phi!::Function
    phi!::Function
    cost_fun::Function
    mapxhow::Function
    niter::Int
    x_hist::Tm

    function WFState(x0::AbstractVector,
                     A, y, b;
                     niter = 200,
                     gradhow::Symbol = :poisson,
                     xhow::Symbol = :complex)

        n = length(x0)
        T = eltype(x0)
        Tx = typeof(x0)

        x = zeros(T, n)
        x_old = similar(x)
        x_new = similar(x)

        copyto!(x, x0)
        copyto!(x_old, x0)
        copyto!(x_new, x0)

        Ax = A * x

        work_Ax = similar(Ax)
        work_Ax1 = similar(Ax)

        Agx = similar(Ax)
        Ax_old = similar(Ax)
        Ax_new = similar(Ax)
        x_hist = zeros(T, niter, n)
        Tm = typeof(x_hist)

        g = similar(x)
        Adk = similar(Ax)
        fisher_vec = similar(Ax)
        cost_fun_vec = similar(Ax)

        function fisher!(o, work, v, b)
            if gradhow === :gaussian
                broadcast!(abs2, work, v)
                broadcast!(+, work, work, b)
                broadcast!(abs2, o, v)
                broadcast!(*, o, o, work)
                o .*= 16
                return o
            elseif gradhow === :poisson
                broadcast!(abs2, work, v)
                broadcast!(+, work, work, b)
                broadcast!(abs2, o, v)
                broadcast!(/, o, o, work)
                broadcast!(*, o, o, 4)
                return o
            else
                throw("unknown gradhow")
            end
        end

        function grad_phi!(o, work, v, y, b)
            if gradhow === :gaussian
                broadcast!(-, work, y, b)
                work .= max.(work, 0)
                broadcast!(abs2, o, v)
                broadcast!(-, o, o, work)
                broadcast!(*, o, o, v)
                o .*= 4
                return o
            elseif gradhow === :poisson
                broadcast!(abs2, work, v)
                broadcast!(+, work, work, b)
                broadcast!(/, work, y, work)
                broadcast!(-, work, 1, work)
                broadcast!(*, o, v, work)
                broadcast!(*, o, o, 2)
                return o
            else
                throw("unknown gradhow")
            end
        end

        function phi!(o, work, v, y, b)
            if gradhow === :gaussian
                broadcast!(-, o, y, b)
                o .= max.(o, 0)
                broadcast!(abs2, work, v)
                broadcast!(-, o, o, work)
                return o
            elseif gradhow === :poisson
                broadcast!(abs2, work, v)
                broadcast!(+, work, work, b)
                copyto!(o, work)
                work .= log.(work)
                broadcast!(*, work, work, y)
                broadcast!(-, o, o, work)
                return o
            else
                throw("unknown gradhow")
            end
        end

        function cost_fun(o, work, Ax, y, b)
            if gradhow === :gaussian
                # return norm(phi!(o, work, Ax, y, b))^2
                return sum(abs2, phi!(o, work, Ax, y, b))
            elseif gradhow === :poisson
                return sum(phi!(o, work, Ax, y, b))
            else
                throw("unknown gradhow")
            end
        end


        # if gradhow === :gaussian
        #     # fisher(vi, bi) = 16 * abs2(vi) * (abs2(vi) + bi)
        #     global function grad_phi!(o, work, v, y, b)
        #         broadcast!(-, work, y, b)
        #         work .= max.(work, 0)
        #         broadcast!(abs2, o, v)
        #         broadcast!(-, o, o, work)
        #         broadcast!(*, o, o, v)
        #         return o
        #     end
        #     # grad_phi(v, yi, bi) = (abs2(v) - max(yi - bi, 0)) * v
        #     global function phi!(o, work, v, y, b)
        #         broadcast!(-, o, y, b)
        #         o .= max.(o, 0)
        #         broadcast!(abs2, work, v)
        #         broadcast!(-, o, o, work)
        #         return o
        #     end
        #     # phi(v, yi, bi) = max(yi - bi, 0) - abs2.(v)
        #     global cost_fun(o, work, Ax, y, b) = norm(phi!(o, work, Ax, y, b))^2
        # elseif gradhow === :poisson
        #     global function fisher!(o, work, v, b)
        #         broadcast!(abs2, work, v)
        #         broadcast!(+, work, work, b)
        #         broadcast!(abs2, o, v)
        #         broadcast!(/, o, o, work)
        #         broadcast!(*, o, o, 4)
        #         return o
        #     end
        #     # fisher!(o, vi, bi) = copyto!(o, 4 * abs2(vi) / (abs2(vi) + bi))
        #     global function grad_phi!(o, work, v, y, b)
        #         broadcast!(abs2, work, v)
        #         broadcast!(+, work, work, b)
        #         broadcast!(/, work, y, work)
        #         broadcast!(-, work, 1, work)
        #         broadcast!(*, o, v, work)
        #         broadcast!(*, o, o, 2)
        #         return o
        #     end
        #     # grad_phi!(o, v, yi, bi) = copyto!(o, 2 * v * (1 - yi/(abs2(v) + bi)))
        #     global function phi!(o, work, v, y, b)
        #         broadcast!(abs2, work, v)
        #         broadcast!(+, work, work, b)
        #         copyto!(o, work)
        #         work .= log.(work)
        #         broadcast!(*, work, work, y)
        #         broadcast!(-, o, o, work)
        #         return o
        #     end
        #     # global phi!(o, v, yi, bi) = copyto!(o, (abs2(v) + bi) - yi * log(abs2(v) + bi))
        #     global cost_fun(o, work, Ax, y, b) = sum(phi!(o, work, Ax, y, b))
        # else
        #     throw("unknown gradhow")
        # end

        # grad_phi!(Agx, Ax, y, b)
        function mapxhow(x)
            if xhow === :real
                return real(x)
            elseif xhow === :complex
                return x
            else
                throw("unknown xhow")
            end
        end


        # if xhow === :real
        #      = real(x)
        # elseif xhow === :complex
        #     mapxhow(x) = x
        # else
        #     throw("unknown xhow")
        # end

        mul!(g, A', Agx)
        # g .= 4 * mapxhow(g)

        mul!(Adk, A, g)

        new{Tx, Tm}(x, x_old, Ax_old, x_new, Ax_new, Ax, work_Ax, work_Ax1,
                Agx, g, Adk, fisher_vec, cost_fun_vec, fisher!, grad_phi!,
                phi!, cost_fun, mapxhow, niter, x_hist)

    end
end


function update_state!(A, y, b,
                       iter::Int,
                       state::WFState;
                       sthow::Symbol = :fisher,
                       mustep = 0.01,
                       mushrink = 2,)

        state.grad_phi!(state.Agx, state.work_Ax, state.Ax, y, b) # 0 bytes
        mul!(state.g, A', state.Agx) # 16 bytes
        copyto!(state.g, state.mapxhow(state.g)) # 0 bytes
        mul!(state.Adk, A, state.g)# 0 bytes
        if sthow === :fisher
            state.fisher!(state.fisher_vec, state.work_Ax, state.Ax, b) # 0 bytes
            broadcast!(sqrt, state.fisher_vec, state.fisher_vec) # 0 bytes
            # state.fisher_vec .= sqrt.(state.fisher_vec)
            broadcast!(*, state.fisher_vec, state.fisher_vec, state.Adk) # 0 bytes
            # state.fisher_vec .*= state.Adk
            domin = norm(state.fisher_vec) # 16 bytes
            domin = domin^2 # 0 bytes
            numer = norm(state.g) # 16 bytes
            numer = numer^2 # 0 bytes
            μ = - numer / domin # 0 bytes
        elseif sthow === :lineser
            μ = 1
            copyto!(state.x_old, state.x)
            copyto!(state.Ax_old, state.Ax)
            cost_Ax_old = state.cost_fun(state.cost_fun_vec, state.work_Ax, state.Ax, y, b)

            broadcast!(-, state.x_new, state.x, μ * state.g)
            broadcast!(-, state.Ax_new, state.Ax, μ * state.Adk)

            mu_grad_f = mustep * norm(state.g)^2
            # cost fun should take A * x0 as input.
            # Tuning param input args
            while cost_fun(state.cost_fun_vec, state.Ax_new, y, b) > cost_Ax_old - μ * mu_grad_f
                μ = μ / mushrink
                broadcast!(-, state.Ax_new, state.Ax_old, μ * state.Adk)
            end
        elseif sthow === :empir
            μ = - min(1 - exp(- iter / 330), 0.4)
        elseif sthow === :optim_gau
            # calculate coefficients for the cubic equation
            broadcast!(+, state.work_Ax, state.Ax, b)
            state.work_Ax .= conj.(state.work_Ax)
            broadcast!(*, state.work_Ax, state.work_Ax, state.Adk)
            state.work_Ax .= real(state.work_Ax)
            # u = real(conj.(Ax .+ b) .* Adk)
            broadcast!(+, state.work_Ax1, state.Ax, b)
            broadcast!(abs2, state.work_Ax1, state.work_Ax1)
            broadcast!(-, state.work_Ax1, state.work_Ax1, y)
            # r = abs2.(Ax .+ b) .- y
            # These lines are allocating, tomorrow I will fix it
            broadcast!(abs, state.Ax_old, state.Adk)
            state.Ax_old .= state.Ax_old .^ 4
            c3 = sum(state.Ax_old)
            # c3 = sum(abs.(state.Adk) .^ 4)
            broadcast!(abs2, state.Ax_old, state.Adk)
            broadcast!(*, state.Ax_new, state.work_Ax, state.Ax_old)
            c2 = -3 * sum(state.Ax_new)
            # c2 = -3 * sum(state.work_Ax .* abs2.(state.Adk))
            broadcast!(*, state.Ax_old, state.Ax_old, state.work_Ax1)
            state.Ax_new .= state.work_Ax .^ 2
            state.Ax_new .*= 2
            state.Ax_new .+= state.Ax_old
            c1 = sum(state.Ax_new)
            # c1 = sum(state.work_Ax1 .* abs2.(state.Adk) .+ 2 * state.work_Ax .^ 2)
            broadcast!(*, state.work_Ax, state.work_Ax, state.work_Ax1)
            c0 = - sum(state.work_Ax)

            roots = - cubic(c2 / c3, c1 / c3, c0 / c3)
            μ = select_root(roots, state)
        end

        broadcast!(*, state.x_old, state.g, μ)
        broadcast!(*, state.Ax_old, state.Adk, μ)

        broadcast!(+, state.x, state.x, state.x_old)
        broadcast!(+, state.Ax, state.Ax, state.Ax_old)

        copyto!((@view state.x_hist[iter, :]), state.x) # 112 bytes

end


function select_root(roots, state)
    if any(isnan.(roots))
        return roots[1]
    else
        broadcast!(+, state.Ax_old, state.Ax, roots[1] * state.Adk)
        broadcast!(+, state.Ax_new, state.Ax, roots[2] * state.Adk)
        broadcast!(+, state.work_Ax1, state.Ax, roots[3] * state.Adk)

        cf1 = state.cost_fun(state.cost_fun_vec, state.work_Ax, state.Ax_old, y, b)
        cf2 = state.cost_fun(state.cost_fun_vec, state.work_Ax, state.Ax_new, y, b)
        cf3 = state.cost_fun(state.cost_fun_vec, state.work_Ax, state.work_Ax1, y, b)
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

function Wirtinger_flow_inplace!(A, y, b,
                                niter::Int,
                                state::WFState;
                                sthow::Symbol = :fisher,
                                mustep = 0.01,
                                mushrink = 2,)

        mul!(state.Ax, A, state.x)
        for iter = 1:niter
            update_state!(A, y, b, iter, state;
                          sthow = sthow,
                          mustep = mustep,
                          mushrink = mushrink,
                          )
        end

end
