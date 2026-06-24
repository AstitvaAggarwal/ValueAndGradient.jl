module ValueAndGradientMooncakeExt

using ValueAndGradient: ValueAndGradient, DiffInput
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake, Config

function _pb_cache(f, xs...; config::Union{Config, Nothing})
    config === nothing && return Mooncake.prepare_pullback_cache(f, xs...)
    return Mooncake.prepare_pullback_cache(f, xs...; config)
end

function _deriv_cache(f, xs...; config::Union{Config, Nothing})
    config === nothing && return Mooncake.prepare_derivative_cache(f, xs...)
    return Mooncake.prepare_derivative_cache(f, xs...; config)
end

# Mooncake.Tangent wraps a NamedTuple of field tangents for struct outputs.
_mc_normalize(t) = t
_mc_normalize(t::Mooncake.Tangent) = t.fields

function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMooncake, x::DiffInput;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache : _pb_cache(f, x; config = backend.config)
    y, (_, x̄) = Mooncake.value_and_pullback!!(c, ȳ, f, x)
    return y, canonical_tangents ? ValueAndGradient._canonicalize(x, _mc_normalize(x̄)) : x̄
end

function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMooncake, x1::DiffInput, x2::DiffInput, xrest::DiffInput...;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache : _pb_cache(f, x1, x2, xrest...; config = backend.config)
    y, (_, x̄s...) = Mooncake.value_and_pullback!!(c, ȳ, f, x1, x2, xrest...)
    x̄s_norm = map(_mc_normalize, x̄s)
    return y, canonical_tangents ? ValueAndGradient._canonicalize((x1, x2, xrest...), x̄s_norm) : x̄s
end

function ValueAndGradient.value_and_pushforward!!(
        f::F, ẋ, backend::AutoMooncakeForward, x::DiffInput;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache : _deriv_cache(f, x; config = backend.config)
    df = Mooncake.zero_tangent(f)
    y, ẏ = Mooncake.value_and_derivative!!(c, (f, df), (x, ẋ))
    return y, canonical_tangents ? ValueAndGradient._canonicalize(y, _mc_normalize(ẏ)) : ẏ
end

function ValueAndGradient.value_and_pushforward!!(
        f::F, ẋ, backend::AutoMooncakeForward, x1::DiffInput, x2::DiffInput, xrest::DiffInput...;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache : _deriv_cache(f, x1, x2, xrest...; config = backend.config)
    df = Mooncake.zero_tangent(f)
    pairs = map((xi, ẋi) -> (xi, ẋi), (x1, x2, xrest...), ẋ)
    y, ẏ = Mooncake.value_and_derivative!!(c, (f, df), pairs...)
    return y, canonical_tangents ? ValueAndGradient._canonicalize(y, _mc_normalize(ẏ)) : ẏ
end

end
