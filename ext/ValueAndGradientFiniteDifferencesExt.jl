module ValueAndGradientFiniteDifferencesExt

using ValueAndGradient: ValueAndGradient
using FiniteDifferences: jvp, j′vp
using ADTypes: AutoFiniteDifferences

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoFiniteDifferences,
    xs...;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pullback = nothing,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoFiniteDifferences does not support ad_cache; it will be ignored."
    fdm = backend.fdm
    y = f(xs...)
    x̄s = j′vp(fdm, f, ȳ, xs...)
    if length(xs) == 1
        x̄ = only(x̄s)
        return y,
        ValueAndGradient._apply_norm(
            only(xs),
            x̄,
            backend,
            normalise_tangents,
            normalise_pullback,
        )
    else
        t = Tuple(x̄s)
        return y,
        ValueAndGradient._apply_norm(
            xs,
            t,
            backend,
            normalise_tangents,
            normalise_pullback,
        )
    end
end

function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ,
    backend::AutoFiniteDifferences,
    xs...;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pushforward = nothing,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoFiniteDifferences does not support ad_cache; it will be ignored."
    fdm = backend.fdm
    y = f(xs...)
    N = length(xs)
    if N == 1
        ẏ = jvp(fdm, f, (only(xs), ẋ))
    else
        ẏ = jvp(fdm, f, ntuple(i -> (xs[i], ẋ[i]), N)...)
    end
    return y,
    ValueAndGradient._apply_norm(y, ẏ, backend, normalise_tangents, normalise_pushforward)
end

end
