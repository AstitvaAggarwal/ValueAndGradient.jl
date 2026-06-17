module ValueAndGradientMooncakeExt

using ValueAndGradient: ValueAndGradient, DiffInput
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake, Config

_vg_config(backend::Union{AutoMooncake, AutoMooncakeForward}) = something(backend.config, Config())

function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMooncake, x::DiffInput;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache :
        Mooncake.prepare_pullback_cache(f, x; config = _vg_config(backend))
    y, (_, x̄) = Mooncake.value_and_pullback!!(c, ȳ, f, x)
    return y, x̄
end

function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMooncake, x1::DiffInput, x2::DiffInput, xrest::DiffInput...;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache :
        Mooncake.prepare_pullback_cache(f, x1, x2, xrest...; config = _vg_config(backend))
    y, (_, x̄s...) = Mooncake.value_and_pullback!!(c, ȳ, f, x1, x2, xrest...)
    return y, x̄s
end

function ValueAndGradient.value_and_pushforward!!(
        f::F, ẋ, backend::AutoMooncakeForward, x::DiffInput;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache :
        Mooncake.prepare_derivative_cache(f, x; config = _vg_config(backend))
    df = Mooncake.zero_tangent(f)
    y, ẏ = Mooncake.value_and_derivative!!(c, (f, df), (x, ẋ))
    return y, ẏ
end

function ValueAndGradient.value_and_pushforward!!(
        f::F, ẋ, backend::AutoMooncakeForward, x1::DiffInput, x2::DiffInput, xrest::DiffInput...;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache :
        Mooncake.prepare_derivative_cache(f, x1, x2, xrest...; config = _vg_config(backend))
    df = Mooncake.zero_tangent(f)
    pairs = map((xi, ẋi) -> (xi, ẋi), (x1, x2, xrest...), ẋ)
    y, ẏ = Mooncake.value_and_derivative!!(c, (f, df), pairs...)
    return y, ẏ
end

end
