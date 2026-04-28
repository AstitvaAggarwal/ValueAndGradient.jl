module ADInterfaceMooncakeExt

using ADInterface: ADInterface, AbstractGradientCache, GradientOrder
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake, Config

# ── Reverse-mode cache types ───────────────────────────────────────────────────

# Single differentiable argument — wraps Mooncake's native gradient cache.
struct MooncakeGradientCache{C} <: AbstractGradientCache
    inner::C
end

# Multiple differentiable arguments — wraps Mooncake's pullback cache.
# Stores the output type T so we can construct the correct cotangent seed one(T).
struct MooncakePullbackCache{C, T} <: AbstractGradientCache
    inner::C
    output_type::Type{T}
end

# Stores output element type and length (0 = scalar) so we construct the right cotangent.
# Scalar output → single pullback with one(ElY); vector output → m pullbacks with basis vectors.
struct MooncakeJacobianCache{C, ElY} <: AbstractGradientCache
    inner::C
    out_length::Int  # 0 for scalar outputs
end

# ── Forward-mode cache types ───────────────────────────────────────────────────

# Wraps a Mooncake ForwardCache; used for both single- and multi-arg gradient.
struct MooncakeForwardGradientCache{C} <: AbstractGradientCache
    inner::C
end

# ForwardCache for Jacobians: n JVPs (one per input dim) give Jacobian columns.
struct MooncakeForwardJacobianCache{C, ElX} <: AbstractGradientCache
    inner::C
    in_length::Int   # n = length(x)
    out_length::Int  # 0 = scalar, m = vector
end

# ── Capability ─────────────────────────────────────────────────────────────────

ADInterface.gradient_order(::AutoMooncake) = GradientOrder{1}()
ADInterface.gradient_order(::AutoMooncakeForward) = GradientOrder{1}()

# ── prepare_gradient_cache — reverse mode ─────────────────────────────────────

function ADInterface.prepare_gradient_cache(
        f::F, backend::AutoMooncake, x::Vararg{Any, 1},
    ) where {F}
    config = something(backend.config, Config())
    return MooncakeGradientCache(Mooncake.prepare_gradient_cache(f, only(x); config))
end

function ADInterface.prepare_gradient_cache(
        f::F, backend::AutoMooncake, x::Vararg{Any, N},
    ) where {F, N}
    config = something(backend.config, Config())
    T = typeof(f(x...))
    return MooncakePullbackCache(Mooncake.prepare_pullback_cache(f, x...; config), T)
end

# ── prepare_gradient_cache — forward mode ─────────────────────────────────────

# Forward mode handles all arities uniformly via ForwardCache.
function ADInterface.prepare_gradient_cache(
        f::F, backend::AutoMooncakeForward, x::Vararg{Any, N},
    ) where {F, N}
    config = something(backend.config, Config())
    return MooncakeForwardGradientCache(Mooncake.prepare_derivative_cache(f, x...; config))
end

# ── prepare_jacobian_cache ─────────────────────────────────────────────────────

function ADInterface.prepare_jacobian_cache(
        f::F, backend::AutoMooncake, x::Vararg{Any, 1},
    ) where {F}
    config = something(backend.config, Config())
    y = f(only(x))
    ElY = y isa Number ? typeof(y) : eltype(y)
    out_length = y isa Number ? 0 : length(y)
    inner = Mooncake.prepare_pullback_cache(f, only(x); config)
    return MooncakeJacobianCache{typeof(inner), ElY}(inner, out_length)
end

function ADInterface.prepare_jacobian_cache(
        f::F, backend::AutoMooncakeForward, x::Vararg{Any, 1},
    ) where {F}
    config = something(backend.config, Config())
    xonly = only(x)
    y = f(xonly)
    ElX = eltype(xonly)
    out_length = y isa Number ? 0 : length(y)
    inner = Mooncake.prepare_derivative_cache(f, xonly; config)
    return MooncakeForwardJacobianCache{typeof(inner), ElX}(inner, length(xonly), out_length)
end

# ── value_and_gradient!! — reverse mode ───────────────────────────────────────

function ADInterface.value_and_gradient!!(
        cache::MooncakeGradientCache, f::F, x::Vararg{Any, 1},
    ) where {F}
    y, (_, g) = Mooncake.value_and_gradient!!(cache.inner, f, only(x))
    return y, g
end

function ADInterface.value_and_gradient!!(
        cache::MooncakePullbackCache{C, T}, f::F, x::Vararg{Any, N},
    ) where {C, T, F, N}
    y, (_, gs...) = Mooncake.value_and_pullback!!(cache.inner, one(T), f, x...)
    return y, gs
end

# ── value_and_gradient!! — forward mode ───────────────────────────────────────

function ADInterface.value_and_gradient!!(
        cache::MooncakeForwardGradientCache, f::F, x::Vararg{Any, 1},
    ) where {F}
    y, (_, g) = Mooncake.value_and_gradient!!(cache.inner, f, only(x))
    return y, g
end

function ADInterface.value_and_gradient!!(
        cache::MooncakeForwardGradientCache, f::F, x::Vararg{Any, N},
    ) where {F, N}
    y, (_, gs...) = Mooncake.value_and_gradient!!(cache.inner, f, x...)
    return y, gs
end

# ── value_and_jacobian!! — reverse mode ───────────────────────────────────────

function ADInterface.value_and_jacobian!!(
        cache::MooncakeJacobianCache{C, ElY}, f::F, x::Vararg{Any, 1},
    ) where {C, ElY, F}
    xonly = only(x)
    if cache.out_length == 0
        # Scalar output: single pullback, gradient has same shape as input
        y, (_, dx) = Mooncake.value_and_pullback!!(cache.inner, one(ElY), f, xonly)
        return y, dx
    else
        # Vector output: one pullback per output dimension to build the full m×n Jacobian
        m = cache.out_length
        n = length(xonly)
        J = Matrix{ElY}(undef, m, n)
        local y
        for i in 1:m
            ȳ = zeros(ElY, m)
            ȳ[i] = one(ElY)
            yi, (_, dx) = Mooncake.value_and_pullback!!(cache.inner, ȳ, f, xonly)
            i == 1 && (y = yi)
            J[i, :] .= dx
        end
        return y, J
    end
end

# ── value_and_jacobian!! — forward mode ───────────────────────────────────────

function ADInterface.value_and_jacobian!!(
        cache::MooncakeForwardJacobianCache{C, ElX}, f::F, x::Vararg{Any, 1},
    ) where {C, ElX, F}
    xonly = only(x)
    if cache.out_length == 0
        # Scalar: forward-mode gradient is a natural single-pass computation
        y, (_, dx) = Mooncake.value_and_gradient!!(cache.inner, f, xonly)
        return y, dx
    else
        # Vector: n JVPs with standard basis tangents → columns of the m×n Jacobian
        n = cache.in_length
        m = cache.out_length
        J = Matrix{ElX}(undef, m, n)
        df = Mooncake.zero_tangent(f)
        local y
        for j in 1:n
            ẋ = zeros(ElX, n)
            ẋ[j] = one(ElX)
            yi, ẏ = Mooncake.value_and_derivative!!(
                cache.inner, (f, df), (xonly, ẋ)
            )
            j == 1 && (y = yi)
            J[:, j] .= ẏ
        end
        return y, J
    end
end

end
