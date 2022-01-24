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
