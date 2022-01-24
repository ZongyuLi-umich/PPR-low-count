"""
Nonlinear preconditioned conjugate gradient algorithm to minimize
a general "inverse problem" cost function `f(Bx)`
where `f` is a quadratic function.
Input:
`B`: The Matrix in front of x
`gf`: functions for computing gradients of `f`
`x0`: initial guess
Optional Input:
`niter`: number of iterations
`xhow`: x is real or complex
`W`: Curvature matrix, can be a linear map
`P`: preconditioner
`betahow`: "beta" method for the search direction: default `:dai_yuan`
`fun` User-defined function to be evaluated with two arguments `(x,iter).
It is evaluated at `(x0,0)` and then after each iteration.
Output:
`x`: final iterate
`out`: `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""

function ncg_phase(
                B::AbstractMatrix{<:Number},
                gf::Function,
                x0::AbstractVector{<:Number} ;
                niter::Int = 50,
                xhow::Symbol = :real,
                P = I,
                W = I,
                betahow::Symbol = :dai_yuan,
                fun::Function = (x,iter) -> undef)
        out = Array{Any}(undef, niter+1)
        out[1] = fun(x0, 0)
        x = copy(x0)
        dir = []
        grad_old = []
        grad_new = []
        Bx = B * x
        grad = (Bx) -> B' * gf(Bx)
        for iter = 1:niter
                grad_new = grad(Bx) # gradient
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
                        dir = npgrad + betaval * dir # search direction
                end
                grad_old = grad_new
                Bd = B * dir
                alf_sol = - Bd' * gf(Bx) / (Bd' * (W * Bd))
                if xhow === :real
                        x += real(alf_sol * dir)
                        Bx += real(alf_sol * Bd)
                elseif xhow === :complex
                        x += alf_sol * dir
                        Bx += alf_sol * Bd
                else
                        throw("unknown xhow")
                end
                out[iter+1] = fun(x, iter)
        end
        return x, out
end
