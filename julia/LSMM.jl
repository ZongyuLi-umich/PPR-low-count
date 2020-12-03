using LinearAlgebra
"""
Input:
`A`: M x N complex matrix, where the ith row is a_i'
`y`: M x 1, real measurement vector
`b`: background counts
Optional Input:
`x0`: initial guess
`niter`: number of iterations, default = 100.
`fun` User-defined function to be evaluated with two arguments `(x,iter).
It is evaluated at `(x0,0)` and then after each iteration.
This minimizes the cost function: f = ||Ax - diag(sqrt(y))*c||_2^2
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function LSMM(A::AbstractMatrix{<:Number},
                y::AbstractVector{<:Number},
                b::AbstractVector{<:Number};
                curvhow::Symbol = :max,
                xhow::Symbol = :real,
                x0 = nothing,
                niter = 100,
                fun::Function = (x,iter) -> undef)

        M, N = size(A)
        cost_out = zeros(niter + 1)
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
        grad_phi = (v, yi, bi) -> 2 * v * (1 - yi/(abs2(v) + bi)) # Here we know x is real
        curv_phi = (v, yi, bi) -> 2 + 2 * yi * (abs2(v) - bi) / (abs2(v) + bi)^2
        out[1] = fun(x0,0)
        if curvhow == :max
                W = I(M) * (2 .+ y ./ (4 * b))
                for iter = 1:niter
                        if xhow == :real
                                x0 = x0 - real(A' * Diagonal(W) * A) \ real(A' * grad_phi.(A * x0, y, b))
                        elseif xhow == :complex
                                x0 = x0 - (A' * Diagonal(W) * A) \ (A' * grad_phi.(A * x0, y, b))
                        else
                                throw("unknown xhow")
                        end
                        out[iter + 1] = fun(x0, iter)
                end
        elseif curvhow == :imp
                for iter = 1:niter
                        s = A * x0
                        gs = (b .+ sqrt.(b.^2 + b .* abs2.(s))) ./ abs.(s)
                        W = I(M) * curv_phi.(gs, y, b)
                        if xhow == :real
                                x0 = x0 - real(A' * Diagonal(W) * A) \ real(A' * grad_phi.(A * x0, y, b))
                        elseif xhow == :complex
                                x0 = x0 - (A' * Diagonal(W) * A) \ (A' * grad_phi.(A * x0, y, b))
                        else
                                throw("unknown xhow")
                        end

                        out[iter + 1] = fun(x0, iter)
                end
        else
                throw("unknown curvhow")

        end
        return x0, out

end


function ncg_inv_jf(B::AbstractVector{<:Any},
                gf::AbstractVector{<:Function},
                Lgf::AbstractVector{<:Real},
                x0::AbstractVector{<:Number};
                niter::Int = 50,
                ninner::Int = 10,
                P = I,
                betahow::Symbol = :dai_yuan,
                fun::Function = (x,iter) -> undef)
        out = Array{Any}(undef, niter + 1)
        out[1] = fun(x0, 0)
        J = length(B)
        x = x0
        dir = []
        grad_old = []
        grad_new = []
        Bx = [B[j] * x for j=1:J]
        grad = (Bx) -> sum([B[j]' * gf[j](Bx[j]) for j = 1:J ])
        for iter = 1:niter
                grad_new = grad(Bx)
                npgrad = -(P * grad_new)
                if iter == 1
                        dir = npgrad
                else
                        if betahow == :dai_yuan
                                betaval = grad_new' * (P * grad_new) /
                                        ((grad_new - grad_old)' * dir)
                        else
                                throw("unknown beta choice")
                        end
                        dir = npgrad + betaval * dir
                end
                grad_old = grad_new
                Bd = [B[j] * dir for j = 1:J]
                dh = alf -> sum([Bd[j]' * gf[j](Bx[j] + alf * Bd[j]) for j=1:J])
                Ldh = sum(Lgf .* norm.(Bd).^2) # Lipschitz constant for dh
                (alf, ) = gd(dh, Ldh, 0, niter=ninner) # GD-based line search
                x += alf * dir
                Bx += alf * Bd
                out[iter + 1] = fun(x, iter)
        end
        return x, out
end

function gd(g::Function, L::Real,
        x0::Union{Number,AbstractVector{<:Number}} ;
        niter::Int=100,
        fun::Function = (x,iter) -> undef)

        out = Array{Any}(undef, niter + 1)
        out[1] = fun(x0, 0)
        mu = 1 / L
        x = x0
        for iter = 1:niter
                x -= mu * g(x)
                out[iter + 1] = fun(x, iter)
        end
        return x, out
end
