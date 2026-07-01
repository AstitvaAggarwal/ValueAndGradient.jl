module ValueAndGradientReverseDiffExt

using ValueAndGradient: ValueAndGradient
using ReverseDiff: ReverseDiff
using ADTypes: AutoReverseDiff

_vdot(ȳ::Number, y::Number) = real(conj(ȳ) * y)
_vdot(ȳ::AbstractArray, y::AbstractArray) = real(sum(conj.(ȳ) .* y))
_vdot(ȳ::Tuple, y::Tuple) = sum(_vdot(ȳi, yi) for (ȳi, yi) in zip(ȳ, y))
_vdot(ȳ::NamedTuple{K}, y::NamedTuple{K}) where {K} = sum(_vdot(ȳ[k], y[k]) for k in K)

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoReverseDiff,
    x::AbstractArray;
    ad_cache = nothing,
    normalise_tangents = false,
    kwargs...,
) where {F}
    y = f(x)
    if ad_cache !== nothing
        ∂x = similar(x)
        ReverseDiff.gradient!(∂x, ad_cache, x)
        return y, normalise_tangents ? ValueAndGradient._normalise(x, ∂x, backend) : ∂x
    else
        x̄ = ReverseDiff.gradient(xi -> _vdot(ȳ, f(xi)), x)
        return y, normalise_tangents ? ValueAndGradient._normalise(x, x̄, backend) : x̄
    end
end

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoReverseDiff,
    x1::AbstractArray,
    x2::AbstractArray,
    xrest::AbstractArray...;
    ad_cache = nothing,
    normalise_tangents = false,
    kwargs...,
) where {F}
    xs = (x1, x2, xrest...)
    N = length(xs)
    y = f(xs...)
    if ad_cache !== nothing
        ∂xs = map(similar, xs)
        ReverseDiff.gradient!(∂xs, ad_cache, xs)
        return y, normalise_tangents ? ValueAndGradient._normalise(xs, ∂xs, backend) : ∂xs
    else
        x̄s = ntuple(N) do k
            ReverseDiff.gradient(
                xk -> _vdot(ȳ, f(ntuple(i -> i == k ? xk : xs[i], Val(N))...)),
                xs[k],
            )
        end
        return y, normalise_tangents ? ValueAndGradient._normalise(xs, x̄s, backend) : x̄s
    end
end

end
