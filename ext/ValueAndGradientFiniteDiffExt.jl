module ValueAndGradientFiniteDiffExt

using ValueAndGradient: ValueAndGradient
using FiniteDiff: FiniteDiff
using ADTypes: AutoFiniteDiff

_vdot(ȳ::Number, y::Number) = real(conj(ȳ) * y)
_vdot(ȳ::AbstractArray, y::AbstractArray) = real(sum(conj.(ȳ) .* y))
_vdot(ȳ::Tuple, y::Tuple) = sum(_vdot(ȳi, yi) for (ȳi, yi) in zip(ȳ, y))
_vdot(ȳ::NamedTuple{K}, y::NamedTuple{K}) where {K} = sum(_vdot(ȳ[k], y[k]) for k in K)

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoFiniteDiff,
    x::AbstractArray;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pullback = nothing,
    kwargs...,
) where {F}
    y = f(x)
    h = xi -> _vdot(ȳ, f(xi))
    if ad_cache !== nothing
        ∂x = similar(x)
        FiniteDiff.finite_difference_gradient!(∂x, h, x, ad_cache)
        return y,
        ValueAndGradient._apply_norm(
            x,
            ∂x,
            backend,
            normalise_tangents,
            normalise_pullback,
        )
    else
        x̄ = FiniteDiff.finite_difference_gradient(h, x)
        return y,
        ValueAndGradient._apply_norm(x, x̄, backend, normalise_tangents, normalise_pullback)
    end
end

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoFiniteDiff,
    x1::AbstractArray,
    x2::AbstractArray,
    xrest::AbstractArray...;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pullback = nothing,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoFiniteDiff does not support ad_cache for multi-arg pullback; it will be ignored."
    xs = (x1, x2, xrest...)
    N = length(xs)
    y = f(xs...)
    x̄s = ntuple(N) do k
        FiniteDiff.finite_difference_gradient(
            xk -> _vdot(ȳ, f(ntuple(i -> i == k ? xk : xs[i], Val(N))...)),
            xs[k],
        )
    end
    return y,
    ValueAndGradient._apply_norm(xs, x̄s, backend, normalise_tangents, normalise_pullback)
end

function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ::AbstractArray,
    backend::AutoFiniteDiff,
    x::AbstractArray;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pushforward = nothing,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoFiniteDiff does not support ad_cache for pushforward; it will be ignored."
    y = f(x)
    ẏ = FiniteDiff.finite_difference_jvp(f, x, ẋ)
    return y,
    ValueAndGradient._apply_norm(y, ẏ, backend, normalise_tangents, normalise_pushforward)
end

function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ,
    backend::AutoFiniteDiff,
    x1::AbstractArray,
    x2::AbstractArray,
    xrest::AbstractArray...;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pushforward = nothing,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoFiniteDiff does not support ad_cache for pushforward; it will be ignored."
    xs = (x1, x2, xrest...)
    N = length(xs)
    y = f(xs...)
    ε = cbrt(eps(float(eltype(x1))))
    g_plus = f(ntuple(i -> xs[i] .+ ε .* ẋ[i], Val(N))...)
    g_minus = f(ntuple(i -> xs[i] .- ε .* ẋ[i], Val(N))...)
    ẏ = (g_plus .- g_minus) ./ (2 * ε)
    return y,
    ValueAndGradient._apply_norm(y, ẏ, backend, normalise_tangents, normalise_pushforward)
end

end
