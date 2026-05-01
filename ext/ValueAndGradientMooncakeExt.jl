module ValueAndGradientMooncakeExt

using ValueAndGradient: ValueAndGradient, FWithCache
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake, Config

function ValueAndGradient.FWithCache(f::F, backend::AutoMooncake, x::Vararg{Any, N}) where {F, N}
    config = something(backend.config, Config())
    return FWithCache(f, Mooncake.prepare_pullback_cache(f, x...; config))
end

function ValueAndGradient.FWithCache(f::F, backend::AutoMooncakeForward, x::Vararg{Any, N}) where {F, N}
    config = something(backend.config, Config())
    return FWithCache(f, Mooncake.prepare_derivative_cache(f, x...; config))
end

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

function ValueAndGradient.value_and_pullback!!(
        fc::FWithCache, ȳ, ::AutoMooncake, x::Vararg{Any, 1},
    )
    y, (_, x̄) = Mooncake.value_and_pullback!!(fc.cache, ȳ, fc.f, only(x))
    return y, x̄
end

function ValueAndGradient.value_and_pullback!!(
        fc::FWithCache, ȳ, ::AutoMooncake, x::Vararg{Any, N},
    ) where {N}
    y, (_, x̄s...) = Mooncake.value_and_pullback!!(fc.cache, ȳ, fc.f, x...)
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

function ValueAndGradient.value_and_pushforward!!(
        fc::FWithCache, ẋ, ::AutoMooncakeForward, x::Vararg{Any, 1},
    )
    df = Mooncake.zero_tangent(fc.f)
    y, ẏ = Mooncake.value_and_derivative!!(fc.cache, (fc.f, df), (only(x), ẋ))
    return y, ẏ
end

function ValueAndGradient.value_and_pushforward!!(
        fc::FWithCache, ẋ, ::AutoMooncakeForward, x::Vararg{Any, N},
    ) where {N}
    df = Mooncake.zero_tangent(fc.f)
    pairs = ntuple(k -> (x[k], ẋ[k]), Val(N))
    y, ẏ = Mooncake.value_and_derivative!!(fc.cache, (fc.f, df), pairs...)
    return y, ẏ
end

end
