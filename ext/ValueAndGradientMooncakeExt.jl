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

ValueAndGradient._normalise(
    x,
    t::Mooncake.Tangent,
    ::Union{AutoMooncake,AutoMooncakeForward},
) = ValueAndGradient._normalise(x, t.fields, nothing)

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoMooncake,
    x::DiffInput;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pullback = nothing,
) where {F}
    c = ad_cache !== nothing ? ad_cache : _pb_cache(f, x; config = backend.config)
    y, (_, x̄) = Mooncake.value_and_pullback!!(c, ȳ, f, x)
    return y,
    ValueAndGradient._apply_norm(x, x̄, backend, normalise_tangents, normalise_pullback)
end

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoMooncake,
    x1::DiffInput,
    x2::DiffInput,
    xrest::DiffInput...;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pullback = nothing,
) where {F}
    c =
        ad_cache !== nothing ? ad_cache :
        _pb_cache(f, x1, x2, xrest...; config = backend.config)
    y, (_, x̄s...) = Mooncake.value_and_pullback!!(c, ȳ, f, x1, x2, xrest...)
    return y,
    ValueAndGradient._apply_norm(
        (x1, x2, xrest...),
        x̄s,
        backend,
        normalise_tangents,
        normalise_pullback,
    )
end

function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ,
    backend::AutoMooncakeForward,
    x::DiffInput;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pushforward = nothing,
) where {F}
    c = ad_cache !== nothing ? ad_cache : _deriv_cache(f, x; config = backend.config)
    df = Mooncake.zero_tangent(f)
    y, ẏ = Mooncake.value_and_derivative!!(c, (f, df), (x, ẋ))
    return y,
    ValueAndGradient._apply_norm(y, ẏ, backend, normalise_tangents, normalise_pushforward)
end

function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ,
    backend::AutoMooncakeForward,
    x1::DiffInput,
    x2::DiffInput,
    xrest::DiffInput...;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pushforward = nothing,
) where {F}
    c =
        ad_cache !== nothing ? ad_cache :
        _deriv_cache(f, x1, x2, xrest...; config = backend.config)
    df = Mooncake.zero_tangent(f)
    pairs = map((xi, ẋi) -> (xi, ẋi), (x1, x2, xrest...), ẋ)
    y, ẏ = Mooncake.value_and_derivative!!(c, (f, df), pairs...)
    return y,
    ValueAndGradient._apply_norm(y, ẏ, backend, normalise_tangents, normalise_pushforward)
end

end
