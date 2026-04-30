module ValueAndGradientMooncakeExt

using ValueAndGradient: ValueAndGradient, AbstractADCache, GradientOrder
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake, Config

struct MooncakePullbackCache{C} <: AbstractADCache
    inner::C
end

struct MooncakePushforwardCache{C} <: AbstractADCache
    inner::C
end

ValueAndGradient.gradient_order(::AutoMooncake) = GradientOrder{1}()
ValueAndGradient.gradient_order(::AutoMooncakeForward) = GradientOrder{1}()

function ValueAndGradient.prepare_pullback_cache(
        f::F, backend::AutoMooncake, x::Vararg{Any, N},
    ) where {F, N}
    config = something(backend.config, Config())
    return MooncakePullbackCache(Mooncake.prepare_pullback_cache(f, x...; config))
end

function ValueAndGradient.prepare_pushforward_cache(
        f::F, backend::AutoMooncakeForward, x::Vararg{Any, N},
    ) where {F, N}
    config = something(backend.config, Config())
    return MooncakePushforwardCache(Mooncake.prepare_derivative_cache(f, x...; config))
end

# Mooncake returns (y, (∂f, ∂x1, ..., ∂xN)); we drop ∂f.
function ValueAndGradient.value_and_pullback!!(
        cache::MooncakePullbackCache, f::F, ȳ, x::Vararg{Any, 1},
    ) where {F}
    y, (_, x̄) = Mooncake.value_and_pullback!!(cache.inner, ȳ, f, only(x))
    return y, x̄
end

function ValueAndGradient.value_and_pullback!!(
        cache::MooncakePullbackCache, f::F, ȳ, x::Vararg{Any, N},
    ) where {F, N}
    y, (_, x̄s...) = Mooncake.value_and_pullback!!(cache.inner, ȳ, f, x...)
    return y, x̄s
end

function ValueAndGradient.value_and_pushforward!!(
        cache::MooncakePushforwardCache, f::F, ẋ, x::Vararg{Any, 1},
    ) where {F}
    df = Mooncake.zero_tangent(f)
    y, ẏ = Mooncake.value_and_derivative!!(cache.inner, (f, df), (only(x), ẋ))
    return y, ẏ
end

function ValueAndGradient.value_and_pushforward!!(
        cache::MooncakePushforwardCache, f::F, ẋ, x::Vararg{Any, N},
    ) where {F, N}
    df = Mooncake.zero_tangent(f)
    pairs = ntuple(k -> (x[k], ẋ[k]), Val(N))
    y, ẏ = Mooncake.value_and_derivative!!(cache.inner, (f, df), pairs...)
    return y, ẏ
end

end
