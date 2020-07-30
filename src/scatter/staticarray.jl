## Scatter operations for StaticArray

for op = [:add, :sub, :mul, :div]
    fn = Symbol("scatter_$(op)!")
    @eval function $fn(ys::StaticArray{<:Tuple,T}, us::StaticArray{<:Tuple,T},
                       xs::StaticArray{<:Tuple,<:IntOrTuple}) where {T<:Real}
        @simd for k = 1:length(xs)
            k = CartesianIndices(xs)[k]
            ys_v = view(ys, :, xs[k]...)
            us_v = view(us, :, k)
            @inbounds ys_v .= $(name2op[op]).(ys_v, us_v)
        end
        ys
    end
end

function scatter_max!(ys::StaticArray{<:Tuple,T}, us::StaticArray{<:Tuple,T},
                      xs::StaticArray{<:Tuple,<:IntOrTuple}) where {T<:Real}
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        ys_v = view(ys, :, xs[k]...)
        us_v = view(us, :, k)
        @inbounds ys_v .= max.(ys_v, us_v)
    end
    ys
end

function scatter_min!(ys::StaticArray{<:Tuple,T}, us::StaticArray{<:Tuple,T},
                      xs::StaticArray{<:Tuple,<:IntOrTuple}) where {T<:Real}
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        ys_v = view(ys, :, xs[k]...)
        us_v = view(us, :, k)
        @inbounds ys_v .= min.(ys_v, us_v)
    end
    ys
end

function scatter_mean!(ys::StaticArray{<:Tuple,T}, us::StaticArray{<:Tuple,T},
                       xs::StaticArray{<:Tuple,<:IntOrTuple}) where {T<:Real}
    Ns = zero(ys)
    ys_ = zero(ys)
    scatter_add!(Ns, one.(us), xs)
    scatter_add!(ys_, us, xs)
    ys .+= save_div.(ys_, Ns)
    return ys
end
