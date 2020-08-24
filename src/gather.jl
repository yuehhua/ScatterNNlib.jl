"""
Inverse operation of scatter
"""
function gather(input::AbstractArray{T,N}, index::AbstractArray{<:Integer,N}, dims::Integer) where {T,N}
    @assert dims <= N "Specified dimensions must lower or equal to the rank of input matrix."
    out = similar(index, T)
    @inbounds for x = CartesianIndices(out)
        tup = collect(Tuple(x))
        tup[dims] = index[x]
        view(out, x) .= view(input, tup...)
    end
    return out
end

function gather(input::Matrix{T}, index::Array{Int}) where T
    out = Array{T}(undef, size(input,1), size(index)...)
    @inbounds for ind = CartesianIndices(index)
        view(out, :, ind) .= view(input, :, index[ind])
    end
    return out
end

gather(input::Fill{T,2,<:Any}, index::Array{Int}) where T = gather(Matrix(input), index)

function gather_indices(X::Array{T}) where T
    Y = DefaultDict{T,Vector{CartesianIndex}}(CartesianIndex[])
    @inbounds for (ind, val) = pairs(X)
        push!(Y[val], ind)
    end
    Y
end
