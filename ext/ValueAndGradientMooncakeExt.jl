module ValueAndGradientMooncakeExt

using ValueAndGradient: ValueAndGradient, DiffInput
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake, Config

function _pb_cache(f, xs...; config::Union{Config,Nothing})
    config === nothing && return Mooncake.prepare_pullback_cache(f, xs...)
    return Mooncake.prepare_pullback_cache(f, xs...; config)
end

function _deriv_cache(f, xs...; config::Union{Config,Nothing})
    config === nothing && return Mooncake.prepare_derivative_cache(f, xs...)
    return Mooncake.prepare_derivative_cache(f, xs...; config)
end

ValueAndGradient._canonicalize(x, t::Mooncake.Tangent, ::Union{AutoMooncake,AutoMooncakeForward}) =
    ValueAndGradient._canonicalize(x, t.fields, nothing)

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoMooncake,
    x::DiffInput;
    ad_cache = nothing,
    canonical_tangents = false,
) where {F}
    c = ad_cache !== nothing ? ad_cache : _pb_cache(f, x; config = backend.config)
    y, (_, x̄) = Mooncake.value_and_pullback!!(c, ȳ, f, x)
    return y, canonical_tangents ? ValueAndGradient._canonicalize(x, x̄, backend) : x̄
end

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoMooncake,
    x1::DiffInput,
    x2::DiffInput,
    xrest::DiffInput...;
    ad_cache = nothing,
    canonical_tangents = false,
) where {F}
    c =
        ad_cache !== nothing ? ad_cache :
        _pb_cache(f, x1, x2, xrest...; config = backend.config)
    y, (_, x̄s...) = Mooncake.value_and_pullback!!(c, ȳ, f, x1, x2, xrest...)
    return y,
    canonical_tangents ? ValueAndGradient._canonicalize((x1, x2, xrest...), x̄s, backend) : x̄s
end

function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ,
    backend::AutoMooncakeForward,
    x::DiffInput;
    ad_cache = nothing,
    canonical_tangents = false,
) where {F}
    c = ad_cache !== nothing ? ad_cache : _deriv_cache(f, x; config = backend.config)
    df = Mooncake.zero_tangent(f)
    y, ẏ = Mooncake.value_and_derivative!!(c, (f, df), (x, ẋ))
    return y, canonical_tangents ? ValueAndGradient._canonicalize(y, ẏ, backend) : ẏ
end

function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ,
    backend::AutoMooncakeForward,
    x1::DiffInput,
    x2::DiffInput,
    xrest::DiffInput...;
    ad_cache = nothing,
    canonical_tangents = false,
) where {F}
    c =
        ad_cache !== nothing ? ad_cache :
        _deriv_cache(f, x1, x2, xrest...; config = backend.config)
    df = Mooncake.zero_tangent(f)
    pairs = map((xi, ẋi) -> (xi, ẋi), (x1, x2, xrest...), ẋ)
    y, ẏ = Mooncake.value_and_derivative!!(c, (f, df), pairs...)
    return y, canonical_tangents ? ValueAndGradient._canonicalize(y, ẏ, backend) : ẏ
end

end
