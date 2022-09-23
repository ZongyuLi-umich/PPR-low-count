# Wirtinger_flow_inplace_reg.jl
"""
In-place version of regularized Poisson Wirtinger flow algorithm

"""

mutable struct WFState_TV{Tx, Cx, Tm}
    x::Tx
    Ax::Cx
    work_Ax::Tx
    Rx::Tx
    work_Rx::Tx
    g::Tx
    Ag::Cx
    Rg::Tx
    fisher_vec1::Cx
    fisher_vec2::Tx
    fisher!::Function
    grad!::Function
    curv_reg!::Function
    mapxhow::Function
    niter::Int
    x_hist::Tm
    β::Real
    α::Real
    timer::Tx
    cost_vec::Tx

    function WFState_TV(x0::AbstractVector,
                     grad!::Function,
                     A;
                     β = 2,
                     α = 0.5,
                     R = I,
                     niter = 200,
                     ref = nothing,
                     xhow::Symbol = :complex)

        if isnothing(ref)
            ref = zeros(eltype(x0), size(x0))
        end
        n = length(x0)
        sn = Int(sqrt(n))
        T = eltype(x0)
        Tx = typeof(x0)

        x = zeros(T, n)

        copyto!(x, x0)

        Ax = A * (x + vec(ref))
        Cx = typeof(Ax)
        Rx = R * x
        work_Ax = similar(real(Ax))
        work_Rx = similar(real(Rx))

        function fisher!(o, v, b)
            @. o = 4 * abs2(v) / (abs2(v) + b)
            return o
        end

        function curv_reg!(t, α)
            if abs(t) > α
                return α / abs(t)
            else
                return 1
            end
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
        mul!(Rg, R, g) # for TV

        fisher_vec1 = similar(Ax)
        fisher_vec2 = similar(Rx)
        timer = zeros(T, niter + 1)
        cost_vec = zeros(T, niter + 1)


        new{Tx, Cx, Tm}(x, Ax, work_Ax, Rx, work_Rx, g, Ag,
                    Rg, fisher_vec1, fisher_vec2, fisher!,
                    grad!, curv_reg!, mapxhow, niter, x_hist,
                    β, α, timer, cost_vec)

    end
end


function update_state!(A, b, iter::Int,
                       state::WFState_TV;
                       R = I)

        state.grad!(state.g, state.Ax, state.Rx) # 0 bytes
        # mul!(state.Rx, R, x)
        copyto!(state.g, state.mapxhow(state.g)) # 0 bytes
        mul!(state.Ag, A, state.g) # 0 bytes
        mul!(state.Rg, R, state.g) # for TV

        state.fisher!(state.work_Ax, state.Ax, b) # 0 bytes
        @. state.work_Rx = state.curv_reg!(state.Rx, state.α)
        @. state.fisher_vec1 = sqrt(state.work_Ax) * state.Ag
        @. state.fisher_vec2 = sqrt(state.work_Rx) * state.Rg
        μ = - sum(abs2, state.g) / (sum(abs2, state.fisher_vec1) + state.β * sum(abs2, state.fisher_vec2)) # 0 bytes

        @. state.x += μ * state.g
        @. state.Ax += μ * state.Ag
        @. state.Rx += μ * state.Rg

        copyto!((@view state.x_hist[iter, :]), state.x) # 112 bytes

end


function Wirtinger_flow_inplace_reg!(A, b, niter::Int, state::WFState_TV;
                                    R = I)

        # mul!(state.Ax, A, state.x)
        state.timer[1] = time()
        for iter = 1:niter
            update_state!(A, b, iter, state; R = R)
            state.timer[iter + 1] = time()
        end

end
