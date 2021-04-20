"""
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: background counts
`LAA`: Lipschitz constant of A'A
Optional Input:
`x0`: initial guess
`niter`: number of iterations, default = 100.
`ninner`: number of inner iterations.
`curvhow`: curvature for the quadratic majorizer
`reg`: regulairzer parameter β
`xhow`: x is real or complex
`fun` User-defined function to be evaluated with two arguments `(x,iter).
It is evaluated at `(x0,0)` and then after each iteration.
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function LSMM_ODWT(A::AbstractMatrix{<:Number},
                y::AbstractVector{<:Number},
                b::AbstractVector{<:Number},
                LAA::Real;
                curvhow::Symbol = :max,
                xhow::Symbol = :real,
                x0 = nothing,
                reg = 1,
                niter = 100,
                ninner = 5,
                fun::Function = (x,iter) -> undef)

        M, N2 = size(A)
        N = Int(sqrt(N2))
        T, scales, mfun = Aodwt((N, N)) # T is real
        cost_out = zeros(niter + 1)
        out = Array{Any}(undef, niter+1)
        if isnothing(x0)
                if xhow == :real
                        x0 = vec(randn(N, N))
                elseif xhow == :complex
                        x0 = vec(sqrt(var/2) * (randn(N, N) + im * randn(N, N)))
                else
                        throw("unknown xhow")
                end
        end
        grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi)) # Here we know x is real
        curv_phi = (v, yi, bi) -> 2 + 2 * yi * (abs2(v) - bi) / (abs2(v) + bi)^2
        out[1] = fun(x0,0)

        soft = (z, c) -> sign(z) * max(abs(z) - c, 0)
        g_prox = (x, c) -> vec(T' * reshape(soft.(vec(T * reshape(x,N,N)), c * reg * (scales[:] .!= 0)), N, N))

        x = copy(x0)
        xk = copy(x0)

        if curvhow === :max
                W = I(M) * (2 .+ y ./ (4 * b)) # precompute W
                W = min.(W, mean(W) + 3 * std(W))
                L = maximum(W) * LAA
                if xhow === :real
                        ∇q = x -> real(A'*(Diagonal(W)*(A*(x - xk))) + A' * grad_phi.(A * xk, y, b))
                elseif xhow === :complex
                        ∇q = x -> A'*(Diagonal(W)*(A*(x - xk))) + A' * grad_phi.(A * xk, y, b)
                else
                        throw("unknown xhow")
                end
                for iter = 1:niter
                        xk = copy(x)
                        x, _ = pogm_restart(xk, Fcost, ∇q, L; mom =:pogm, niter = ninner,
                                                restart = :gr, g_prox = g_prox)
                        out[iter + 1] = fun(x, iter)
                end
        elseif curvhow === :imp
                Fcost = (x) -> 0
                for iter = 1:niter
                        s = A * x
                        gs = (b .+ sqrt.(b.^2 + b .* abs2.(s))) ./ abs.(s)
                        W = I(M) * curv_phi.(gs, y, b)
                        W = min.(W, mean(W) + 3 * std(W))
                        L = maximum(W) * LAA
                        xk = copy(x)
                        if xhow === :real
                                ∇q = x -> real(A'*(Diagonal(W)*(A*(x - xk))) + A' * grad_phi.(A * xk, y, b))
                        elseif xhow === :complex
                                ∇q = x -> A'*(Diagonal(W)*(A*(x - xk))) + A' * grad_phi.(A * xk, y, b)
                        else
                                throw("unknown xhow")
                        end
                        x, _ = pogm_restart(xk, Fcost, ∇q, L; mom =:pogm, niter = ninner,
                                                restart = :gr, g_prox = g_prox)
                        out[iter + 1] = fun(x, iter)
                end

        elseif curvhow === :fisher
                Fcost = (x) -> 0
                W = 4 * ones(M)
                L = 4 * LAA
                for iter = 1:niter
                        s = A * x
                        xk = copy(x)
                        if xhow === :real
                                ∇q = x -> real(A'*(Diagonal(W)*(A*(x - xk))) + A' * grad_phi.(A * xk, y, b))
                        elseif xhow === :complex
                                ∇q = x -> A'*(Diagonal(W)*(A*(x - xk))) + A' * grad_phi.(A * xk, y, b)
                        else
                                throw("unknown xhow")
                        end
                        x, _ = pogm_restart(xk, Fcost, ∇q, L; mom =:pogm, niter = ninner,
                                                restart = :gr, g_prox = g_prox)
                        out[iter + 1] = fun(x, iter)
                end
        else
                throw("unknown curvhow")
        end
        return x, out
end
