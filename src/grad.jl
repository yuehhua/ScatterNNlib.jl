## Derivatives of scatter operations

@adjoint function scatter_add!(ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    ys_ = copy(ys)
    scatter_add!(ys_, us, xs)
    ys_, Δ -> (Δ, gather(Δ, xs), nothing)
end

@adjoint function scatter_sub!(ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    ys_ = copy(ys)
    scatter_sub!(ys_, us, xs)
    ys_, Δ -> (Δ, -gather(Δ, xs), nothing)
end

@adjoint function scatter_mul!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    ys_ = copy(ys)
    scatter_mul!(ys_, us, xs)
    ys_, function (Δ)
        Δy = Δ .+ zero(ys)
        scatter_mul!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = gather(ys, xs) .* gather(Δ, xs)
        @inbounds for ind = CartesianIndices(xs)
            inds = filter(x -> x != ind, rev_xs[xs[ind]])
            for i = 1:size(us, 1)
                Δu[i, ind] *= prod(j -> us[i, j], inds)
            end
        end
        (Δy, Δu, nothing)
    end
end

@adjoint function scatter_div!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    ys_ = copy(ys)
    scatter_div!(ys_, us, xs)
    ys_, function (Δ)
        Δy = Δ .+ zero(ys)
        scatter_div!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = - gather(ys, xs) .* gather(Δ, xs) ./ us.^2
        @inbounds for ind = CartesianIndices(xs)
            inds = filter(x -> x != ind, rev_xs[xs[ind]])
            for i = 1:size(us, 1)
                Δu[i, ind] /= prod(j -> us[i, j], inds)
            end
        end
        (Δy, Δu, nothing)
    end
end

@adjoint function scatter_max!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    max = copy(ys)
    scatter_max!(max, us, xs)
    max, function (Δ)
       Δy = (ys .== max) .* Δ
       Δu = (us .== gather(max, xs)) .* gather(Δ, xs)
       (Δy, Δu, nothing)
    end
end

@adjoint function scatter_min!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    min = copy(ys)
    scatter_min!(min, us, xs)
    min, function (Δ)
       Δy = (ys .== min) .* Δ
       Δu = (us .== gather(min, xs)) .* gather(Δ, xs)
       (Δy, Δu, nothing)
    end
end

@adjoint function scatter_mean!(ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    ys_ = copy(ys)
    scatter_mean!(ys_, us, xs)
    ys_, function (Δ)
        Δu = gather(Δ, xs)
        counts = zero.(xs)
        @inbounds for i = 1:size(ys, 2)
            counts += sum(xs.==i) * (xs.==i)
        end
        @inbounds for ind = CartesianIndices(counts)
            view(Δu, :, ind) ./= counts[ind]
        end
        (Δ, Δu, nothing)
    end
end
