## Scatter operations

for op = [:add, :sub, :mul, :div]
    fn = Symbol("scatter_$(op)!")
    @eval function $fn(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
        @simd for k = 1:length(xs)
            k = CartesianIndices(xs)[k]
            ys_v = view(ys, :, xs[k]...)
            us_v = view(us, :, k)
            @inbounds ys_v .= $(name2op[op]).(ys_v, us_v)
        end
        ys
    end
end

function scatter_max!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        ys_v = view(ys, :, xs[k]...)
        us_v = view(us, :, k)
        @inbounds view(ys, :, xs[k]...) .= max.(ys_v, us_v)
    end
    ys
end

function scatter_min!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        ys_v = view(ys, :, xs[k]...)
        us_v = view(us, :, k)
        @inbounds ys_v .= min.(ys_v, us_v)
    end
    ys
end

function scatter_mean!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    Ns = zero(ys)
    ys_ = zero(ys)
    scatter_add!(Ns, one.(us), xs)
    scatter_add!(ys_, us, xs)
    ys .+= save_div.(ys_, Ns)
    return ys
end


# Support different types of array
for op = ops
    fn = Symbol("scatter_$(op)!")
    @eval function $fn(ys::Array{T}, us::Array{S}, xs::Array{<:IntOrTuple}) where {T<:Real,S<:Real}
        PT = promote_type(T, S)
        $fn(PT.(ys), PT.(us), xs)
    end
    @eval function $fn(ys::StaticArray{<:Tuple,T}, us::StaticArray{<:Tuple,S},
                       xs::StaticArray{<:Tuple,<:IntOrTuple}) where {T<:Real,S<:Real}
        PT = promote_type(T, S)
        $fn(PT.(ys), PT.(us), xs)
    end
end
