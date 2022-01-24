# Wirtinger_flow_inplace1.jl
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
    Ax::Tx
    work_Ax::Tx
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

        copyto!(x, x0)

        Ax = A * x

        work_Ax = similar(Ax)

        Agx = similar(Ax)
        x_hist = zeros(T, niter, n)
        Tm = typeof(x_hist)

        g = similar(x)
        Adk = similar(Ax)
        fisher_vec = similar(Ax)
        cost_fun_vec = similar(Ax)

        function fisher_gau!(o, v, b)
            @. o = 16 * abs2(v) * (abs2(v) + b)
            return o
        end

        function fisher_poi!(o, v, b)
            @. o = 4 * abs2(v) / (abs2(v) + b)
            return o
        end

        fisher! = gradhow === :poisson ? fisher_poi! : fisher_gau!

        function grad_phi_gau!(o, v, y, b)
            @. o = 4 * (abs2(v) - max(y - b, 0)) * v
            return o
        end

        function grad_phi_poi!(o, v, y, b)
            @. o = 2 * v * (1 - y / (abs2(v) + b))
            return o
        end

        grad_phi! = gradhow === :poisson ? grad_phi_poi! : grad_phi_gau!

        function phi_gau!(o, v, y, b)
            @. o = max(y - b, 0) - abs2(v)
            return o
        end

        function phi_poi!(o, v, y, b)
            @. o = (abs2(v) + b) - y * log(abs2(v) + b)
            return o
        end

        phi! = gradhow === :poisson ? phi_poi! : phi_gau!

        function cost_fun_gau(o, Ax, y, b)
            return sum(abs2, phi_gau!(o, Ax, y, b))
        end

        function cost_fun_poi(o, Ax, y, b)
            sum(abs2, phi_poi!(o, Ax, y, b))
        end

        cost_fun = gradhow === :poisson ? cost_fun_poi : cost_fun_gau

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

        mul!(g, A', Agx)

        mul!(Adk, A, g)

        new{Tx, Tm}(x, Ax, work_Ax, Agx, g, Adk, fisher_vec, cost_fun_vec,
                    fisher!, grad_phi!, phi!, cost_fun, mapxhow, niter, x_hist)

    end
end


function update_state!(A, y, b,
                       iter::Int,
                       state::WFState;
                       sthow::Symbol = :fisher)

        state.grad_phi!(state.Agx, state.Ax, y, b) # 0 bytes
        mul!(state.g, A', state.Agx) # 16 bytes
        copyto!(state.g, state.mapxhow(state.g)) # 0 bytes
        mul!(state.Adk, A, state.g)# 0 bytes

        if sthow === :fisher
            state.fisher!(state.work_Ax, state.Ax, b) # 0 bytes
            @. state.fisher_vec = sqrt(state.work_Ax) * state.Adk
            μ = - sum(abs2, state.g) / sum(abs2, state.fisher_vec) # 0 bytes
        end

        @. state.x += μ * state.g
        @. state.Ax += μ * state.Adk

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

function Wirtinger_flow_inplace1!(A, y, b,
                                  niter::Int,
                                  state::WFState;
                                  sthow::Symbol = :fisher)

        mul!(state.Ax, A, state.x)
        for iter = 1:niter
            update_state!(A, y, b, iter, state;
                          sthow = sthow,
                          )
        end

end
