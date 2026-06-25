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
    canonical_tangents = false,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoFiniteDifferences does not support ad_cache; it will be ignored."
    fdm = backend.fdm
    y = f(xs...)
    x̄s = j′vp(fdm, f, ȳ, xs...)
    if length(xs) == 1
        x̄ = only(x̄s)
        return y, canonical_tangents ? ValueAndGradient._canonicalize(only(xs), x̄, backend) : x̄
    else
        t = Tuple(x̄s)
        return y, canonical_tangents ? ValueAndGradient._canonicalize(xs, t, backend) : t
    end
end

function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ,
    backend::AutoFiniteDifferences,
    xs...;
    ad_cache = nothing,
    canonical_tangents = false,
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
    return y, canonical_tangents ? ValueAndGradient._canonicalize(y, ẏ, backend) : ẏ
end

end
