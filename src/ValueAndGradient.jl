module ValueAndGradient

using ADTypes: ADTypes, AbstractADType

const DiffScalar = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}
const DiffArray  = AbstractArray{<:DiffScalar}
const DiffLeaf   = Union{DiffScalar, DiffArray}
const DiffInput  = Union{DiffLeaf, Tuple{Vararg{DiffLeaf}}}

struct GradientOrder{K}
    function GradientOrder{K}() where {K}
        _K = Int(K)
        _K ≥ 0 || throw(ArgumentError("GradientOrder requires K ≥ 0, got $_K"))
        return new{_K}()
    end
end

GradientOrder(K::Integer) = GradientOrder{Int(K)}()

Base.isless(::GradientOrder{J}, ::GradientOrder{K}) where {J, K} = J < K

gradient_order(::AbstractADType) = nothing

abstract type AbstractADCache end

function prepare_pullback_cache end
function prepare_pushforward_cache end

"""
    value_and_pullback!!(f, ȳ, backend, x...) -> (y, x̄)
    value_and_pullback!!(cache, f, ȳ, x...) -> (y, x̄)

Returns `y = f(x...)` and the VJP `x̄ = (∂f/∂x)ᵀ ȳ`.
Single argument: `x̄` has the same structure as `x`.
Multiple arguments: `x̄` is a tuple of per-argument cotangents.
"""
function value_and_pullback!! end

"""
    value_and_pushforward!!(f, ẋ, backend, x...) -> (y, ẏ)
    value_and_pushforward!!(cache, f, ẋ, x...) -> (y, ẏ)

Returns `y = f(x...)` and the JVP `ẏ = ∂f/∂x * ẋ`.
Single argument: `ẋ` has the same structure as `x`.
Multiple arguments: `ẋ` is a tuple of per-argument tangents.
"""
function value_and_pushforward!! end

# non-caching forms build a cache then delegate

function value_and_pullback!!(f::F, ȳ, backend::AbstractADType, x::Vararg{Any, N}) where {F, N}
    cache = prepare_pullback_cache(f, backend, x...)
    return value_and_pullback!!(cache, f, ȳ, x...)
end

function value_and_pushforward!!(f::F, ẋ, backend::AbstractADType, x::Vararg{Any, N}) where {F, N}
    cache = prepare_pushforward_cache(f, backend, x...)
    return value_and_pushforward!!(cache, f, ẋ, x...)
end

# error fallbacks

function prepare_pullback_cache(::Any, ::T, ::Vararg{Any, N}) where {T <: AbstractADType, N}
    throw(ArgumentError("`ValueAndGradient.prepare_pullback_cache` not implemented for backend `$T`."))
end

function prepare_pushforward_cache(::Any, ::T, ::Vararg{Any, N}) where {T <: AbstractADType, N}
    throw(ArgumentError("`ValueAndGradient.prepare_pushforward_cache` not implemented for backend `$T`."))
end

function value_and_pullback!!(::AbstractADCache, ::Any, ::Any, ::Vararg{Any, N}) where {N}
    throw(ArgumentError("`ValueAndGradient.value_and_pullback!!` not implemented for this cache type."))
end

function value_and_pushforward!!(::AbstractADCache, ::Any, ::Any, ::Vararg{Any, N}) where {N}
    throw(ArgumentError("`ValueAndGradient.value_and_pushforward!!` not implemented for this cache type."))
end

export GradientOrder,
    gradient_order,
    AbstractADCache,
    prepare_pullback_cache,
    prepare_pushforward_cache,
    value_and_pullback!!,
    value_and_pushforward!!

include("test_utils.jl")

end
