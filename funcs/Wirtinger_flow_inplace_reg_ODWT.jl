# Wirtinger_flow_inplace_reg.jl
"""
In-place version of regularized Poisson Wirtinger flow algorithm

"""

mutable struct WFState_ODWT{Tx, Cx, Tm}
    x::Tx
    Ax::Cx
    work_Ax::Tx
    Rx::Cx
    work_Rx::Tx
    g::Tx
    Ag::Cx
    Rg::Cx
    fisher_vec1::Cx
    fisher_vec2::Cx
    fisher!::Function
    grad!::Function
    mapxhow::Function
    niter::Int
    x_hist::Tm
    β::Real
    timer::Tx
    cost_vec::Tx

    function WFState_ODWT(x0::AbstractVector,
                     grad!::Function,
                     A;
                     β = 2,
                     R = I,
                     niter = 200,
                     xhow::Symbol = :complex)

        n = length(x0)
        sn = Int(sqrt(n))
        T = eltype(x0)
        Tx = typeof(x0)

        x = zeros(T, n)

        copyto!(x, x0)

        Ax = A * x
        Cx = typeof(Ax)

        Rx = vec(R * reshape(x, sn, sn)) # for ODWT only
        work_Ax = similar(real(Ax))
        work_Rx = similar(real(Rx))

        function fisher!(o, v, b)
            @. o = 4 * abs2(v) / (abs2(v) + b)
            return o
        end

        function mapxhow(x)
            if xhow === :real
                return real(x)
            elseif xhow === :complex
                return x
            else
                throw("unknown xhow")
            end
        end

        g = similar(x)
        grad!(g, Ax, Rx)
        copyto!(g, mapxhow(g))
        x_hist = zeros(T, niter, n)
        Tm = typeof(x_hist)

        Ag = similar(Ax)
        Rg = similar(Rx)
        mul!(Ag, A, g)
        Rg = vec(R * reshape(g, sn, sn)) # for ODWT
        fisher_vec1 = similar(Ax)
        fisher_vec2 = similar(Rx)
        timer = zeros(T, niter + 1)
        cost_vec = zeros(T, niter + 1)


        new{Tx, Cx, Tm}(x, Ax, work_Ax, Rx, work_Rx, g, Ag,
                    Rg, fisher_vec1, fisher_vec2, fisher!,
                    grad!, mapxhow, niter, x_hist, β, timer, cost_vec)

    end
end


function update_state!(A, b, iter::Int,
                       state::WFState_ODWT;
                       R = I)

        state.grad!(state.g, state.Ax, state.Rx) # 0 bytes
        # mul!(state.Rx, R, x)
        copyto!(state.g, state.mapxhow(state.g)) # 0 bytes
        mul!(state.Ag, A, state.g) # 0 bytes
        n = length(state.g)
        sn = Int(sqrt(n))
        # mul!(state.Rg, R, state.g) # for TV
        state.Rg = vec(R * reshape(state.g, sn, sn))

        state.fisher!(state.work_Ax, state.Ax, b) # 0 bytes

        @. state.work_Rx = 1
        @. state.fisher_vec1 = sqrt(state.work_Ax) * state.Ag
        @. state.fisher_vec2 = sqrt(state.work_Rx) * state.Rg
        μ = - sum(abs2, state.g) / (sum(abs2, state.fisher_vec1) + state.β * sum(abs2, state.fisher_vec2)) # 0 bytes

        @. state.x += μ * state.g
        @. state.Ax += μ * state.Ag
        @. state.Rx += μ * state.Rg

        copyto!((@view state.x_hist[iter, :]), state.x) # 112 bytes

end


function Wirtinger_flow_inplace_reg!(A, b, niter::Int, state::WFState_ODWT;
                                    R = I)

        # mul!(state.Ax, A, state.x)
        state.timer[1] = time()
        for iter = 1:niter
            update_state!(A, b, iter, state; R = R)
            state.timer[iter + 1] = time()
        end

end
