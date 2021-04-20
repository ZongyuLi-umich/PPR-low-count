using LinearAlgebra
using LinearMapsAA
using Random: seed!
using Plots; default(markerstrokecolor=:auto)
using MIRT
using LaTeXStrings
using Distributions:Poisson
using Statistics
using Random: seed!
using JLD
using MAT
using SparseArrays
include("LSMM.jl")
include("Wirtinger_flow.jl")
include("Gerchberg_saxton.jl")
include("poisson_admm.jl")
include("LSMM_TV.jl")
include("LSMM_ODWT.jl")
include("poisson_admm_TV.jl")
include("poisson_admm_ODWT.jl")
lf = x -> log10(max(x,1e-16))
grab = (o,i) -> hcat(o...)[i,:]
lg = (o,i) -> lf.(grab(o,i))
grab_cost(x, r_time) = mean([grab(x[r], 1) for r in r_time])
grab_time(x, r_time) = mean([grab(x[r], 2) .- x[r][1][2] for r in r_time])
grab_nrmse(x, r_time) = mean([grab(x[r], 3) for r in r_time])
