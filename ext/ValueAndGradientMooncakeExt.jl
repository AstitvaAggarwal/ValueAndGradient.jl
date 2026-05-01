module ValueAndGradientMooncakeExt

using ValueAndGradient: ValueAndGradient
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake, Config

function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMooncake, x::Vararg{Any, 1},
    ) where {F}
    config = something(backend.config, Config())
    cache = Mooncake.prepare_pullback_cache(f, only(x); config)
    y, (_, x̄) = Mooncake.value_and_pullback!!(cache, ȳ, f, only(x))
    return y, x̄
end

function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMooncake, x::Vararg{Any, N},
    ) where {F, N}
    config = something(backend.config, Config())
    cache = Mooncake.prepare_pullback_cache(f, x...; config)
    y, (_, x̄s...) = Mooncake.value_and_pullback!!(cache, ȳ, f, x...)
    return y, x̄s
end

function ValueAndGradient.value_and_pushforward!!(
        f::F, ẋ, backend::AutoMooncakeForward, x::Vararg{Any, 1},
    ) where {F}
    config = something(backend.config, Config())
    cache = Mooncake.prepare_derivative_cache(f, only(x); config)
    df = Mooncake.zero_tangent(f)
    y, ẏ = Mooncake.value_and_derivative!!(cache, (f, df), (only(x), ẋ))
    return y, ẏ
end

function ValueAndGradient.value_and_pushforward!!(
        f::F, ẋ, backend::AutoMooncakeForward, x::Vararg{Any, N},
    ) where {F, N}
    config = something(backend.config, Config())
    cache = Mooncake.prepare_derivative_cache(f, x...; config)
    df = Mooncake.zero_tangent(f)
    pairs = ntuple(k -> (x[k], ẋ[k]), Val(N))
    y, ẏ = Mooncake.value_and_derivative!!(cache, (f, df), pairs...)
    return y, ẏ
end

end
