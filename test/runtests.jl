using ScatterNNlib
using StaticArrays: @MMatrix, @MArray
using Zygote
using CUDA
using Test

cuda_tests = [
    "cuda/cuarray",
    "cuda/grad",
]

tests = [
    "scatter/densearray",
    "scatter/staticarray",
    "grad",
    "gather",
]

if CUDA.functional()
    append!(tests, cuda_tests)
else
    @warn "CUDA unavailable, not testing CUDA support"
end

@testset "ScatterNNlib.jl" begin
    for t in tests
        include("$(t).jl")
    end
end
