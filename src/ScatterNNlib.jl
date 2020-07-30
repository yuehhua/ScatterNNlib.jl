module ScatterNNlib

using CUDA
using DataStructures: DefaultDict
using FillArrays: Fill
using Statistics: mean
using StaticArrays: StaticArray
using Zygote
using ZygoteRules

export
    # scatter
    scatter_add!,
    scatter_sub!,
    scatter_max!,
    scatter_min!,
    scatter_mul!,
    scatter_div!,
    scatter_mean!,
    scatter!,

    # gather
    gather

const IntOrTuple = Union{Integer,Tuple}
const ops = [:add, :sub, :mul, :div, :max, :min, :mean]
const name2op = Dict(:add => :+, :sub => :-, :mul => :*, :div => :/)

include("scatter/densearray.jl")
include("scatter/staticarray.jl")
include("interfaces.jl")

include("grad.jl")

include("gather.jl")

include("utils.jl")

if CUDA.functional()
    using CUDA: @cuda
    const MAX_THREADS = 1024

    include("cuda/cuarray.jl")
    include("cuda/grad.jl")
else
    @warn "CUDA unavailable, not loading CUDA support"
end

end
