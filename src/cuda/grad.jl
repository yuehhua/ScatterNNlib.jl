@adjoint function scatter_mul!(ys::CuArray{T}, us::CuArray{T}, xs::CuArray) where {T<:AbstractFloat}
    ys_ = copy(ys)
    scatter_mul!(ys_, us, xs)
    ys_, function (Δ)
        Δy = zero(ys) .+ Δ
        scatter_mul!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = gather(ys, xs) .* gather(zero(Δ)+Δ, xs)
        @inbounds for ind = CartesianIndices(xs)
            ind = Tuple(ind)
            inds = filter(x -> x != ind, rev_xs[xs[ind...]])
            for i = 1:size(us, 1)
                multiplier = one(T)
                for j = inds
                    multiplier *= us[i, j...]
                end
                Δu[i, ind...] *= multiplier
            end
        end
        (Δy, Δu, nothing)
    end
end

@adjoint function scatter_div!(ys::CuArray{T}, us::CuArray{T}, xs::CuArray) where {T<:AbstractFloat}
    ys_ = copy(ys)
    scatter_div!(ys_, us, xs)
    ys_, function (Δ)
        Δy = zero(ys) .+ Δ
        scatter_div!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = - gather(ys, xs)
        Δu .*= gather(zero(Δ)+Δ, xs)
        Δu ./= us.^2
        @inbounds for ind = CartesianIndices(xs)
            ind = Tuple(ind)
            inds = filter(x -> x != ind, rev_xs[xs[ind...]])
            for i = 1:size(us, 1)
                denom = one(T)
                for j = inds
                    denom *= us[i, j...]
                end
                Δu[i, ind...] /= denom
            end
        end
        (Δy, Δu, nothing)
    end
end

@adjoint function scatter_max!(ys::CuArray{T}, us::CuArray{T}, xs::CuArray) where {T<:AbstractFloat}
    max = copy(ys)
    scatter_max!(max, us, xs)
    max, function (Δ)
       Δy = numerical_cmp(ys, max) .* Δ
       Δu = gather(max, xs)
       Δu = numerical_cmp(us, Δu)
       Δu .*= gather(zero(Δ)+Δ, xs)
       (Δy, Δu, nothing)
    end
end

@adjoint function scatter_min!(ys::CuArray{T}, us::CuArray{T}, xs::CuArray) where {T<:AbstractFloat}
    min = copy(ys)
    scatter_min!(min, us, xs)
    min, function (Δ)
       Δy = numerical_cmp(ys, min) .* Δ
       Δu = gather(min, xs)
       Δu = numerical_cmp(us, Δu)
       Δu .*= gather(zero(Δ)+Δ, xs)
       (Δy, Δu, nothing)
    end
end
