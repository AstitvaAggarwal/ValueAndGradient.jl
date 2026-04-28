module ADInterface

using ADTypes: ADTypes, AbstractADType

## Valid input types
#
# Restricted to IEEEFloat / Complex{IEEEFloat} / arrays / tuples thereof,
# so the interface can be fully specified and automatically tested via FiniteDifferences.

const DiffScalar = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}
const DiffArray  = AbstractArray{<:DiffScalar}
const DiffLeaf   = Union{DiffScalar, DiffArray}
const DiffInput  = Union{DiffLeaf, Tuple{Vararg{DiffLeaf}}}

# ── Capability trait ───────────────────────────────────────────────────────────

"""
    GradientOrder{K}

Trait indicating that an AD backend supports computing derivatives up to order `K`:

  - `GradientOrder{0}()`: primal evaluation only
  - `GradientOrder{1}()`: value + gradient / Jacobian
  - `GradientOrder{2}()`: value + gradient + Hessian

Backends declare their capability by implementing [`gradient_order`](@ref).
Consumers can compare orders: `GradientOrder{1}() ≤ GradientOrder{2}()`.
"""
struct GradientOrder{K}
    function GradientOrder{K}() where {K}
        _K = Int(K)
        _K ≥ 0 || throw(ArgumentError("GradientOrder requires K ≥ 0, got $_K"))
        return new{_K}()
    end
end

GradientOrder(K::Integer) = GradientOrder{Int(K)}()

Base.isless(::GradientOrder{J}, ::GradientOrder{K}) where {J, K} = J < K

"""
    gradient_order(backend::AbstractADType) -> GradientOrder{K} or nothing

Return the [`GradientOrder`](@ref) supported by `backend`, or `nothing` if the backend
does not implement the ADInterface gradient API.

Backends declare support by adding a method:

    ADInterface.gradient_order(::MyBackend) = GradientOrder{1}()
"""
gradient_order(::AbstractADType) = nothing

# ── Cache abstract type ────────────────────────────────────────────────────────

"""
    AbstractGradientCache

Abstract supertype for backend-specific derivative caches returned by
[`prepare_gradient_cache`](@ref) and [`prepare_jacobian_cache`](@ref).

Backends subtype this and implement the corresponding cached
[`value_and_gradient!!`](@ref) / [`value_and_jacobian!!`](@ref) methods.
"""
abstract type AbstractGradientCache end

# ── Cache preparation ──────────────────────────────────────────────────────────

"""
    prepare_gradient_cache(f, backend::AbstractADType, x...) -> AbstractGradientCache

Build a reusable cache for repeated [`value_and_gradient!!`](@ref) calls with the
same function `f` and input shapes `x...`.

The cache captures any compilation or tape-building work so that subsequent calls to
`value_and_gradient!!(cache, f, x...)` avoid that overhead.

Valid input types: `IEEEFloat`, `Complex{<:IEEEFloat}`, arrays thereof, or tuples of those.
Multiple differentiable arguments are supported.

# Interface

Backends implement:

    ADInterface.prepare_gradient_cache(f, ::MyBackend, x::Vararg{Any, N}) where {N} = MyGradientCache(...)

See also: [`prepare_jacobian_cache`](@ref), [`value_and_gradient!!`](@ref).
"""
function prepare_gradient_cache end

"""
    prepare_jacobian_cache(f, backend::AbstractADType, x...) -> AbstractGradientCache

Build a reusable cache for repeated [`value_and_jacobian!!`](@ref) calls with the
same function `f` and input shapes `x...`.

See also: [`prepare_gradient_cache`](@ref), [`value_and_jacobian!!`](@ref).
"""
function prepare_jacobian_cache end

# ── Interface functions ────────────────────────────────────────────────────────

"""
    value_and_gradient!!(f, backend::AbstractADType, x...)
    value_and_gradient!!(cache::AbstractGradientCache, f, x...)

Compute the primal value `y = f(x...)` and gradients `∇f` w.r.t. each argument.

For a single argument, returns `(y, g)` where `g` has the same structure as `x`.
For multiple arguments, returns `(y, (g1, g2, ...))` with one gradient per argument.

Valid input types: `IEEEFloat`, `Complex{<:IEEEFloat}`, arrays thereof, or tuples of those.

The `!!` signals that the backend may mutate internal cache state. The caller owns the
returned values: mutable components (e.g. gradient arrays) may be overwritten on the next
call with the same cache, so copy if you need to retain them.

The non-caching form is a one-shot convenience; for hot loops build a cache with
[`prepare_gradient_cache`](@ref) and use the cached form.

# Interface

Backends implement both forms:

    ADInterface.prepare_gradient_cache(f, ::MyBackend, x::Vararg{Any, N}) where {N} = MyGradientCache(...)
    ADInterface.value_and_gradient!!(cache::MyGradientCache, f, x::Vararg{Any, N}) where {N} = ...

and declare:

    ADInterface.gradient_order(::MyBackend) = GradientOrder{1}()

The non-caching form has a default that calls `prepare_gradient_cache` then the cached
form, so backends only need to implement the cached version.

See also: [`value_and_jacobian!!`](@ref), [`gradient_order`](@ref).
"""
function value_and_gradient!! end

"""
    value_and_jacobian!!(f, backend::AbstractADType, x...)
    value_and_jacobian!!(cache::AbstractGradientCache, f, x...)

Compute the primal value `y = f(x...)` and the Jacobian `∂f` w.r.t. each argument.

  - If `f` is scalar-valued, this is equivalent to [`value_and_gradient!!`](@ref).
  - If `f` is vector-valued (`f : ℝⁿ → ℝᵐ`), returns the full `m × n` Jacobian matrix.

For multiple arguments, returns `(y, (J1, J2, ...))` with one Jacobian per argument.

Valid input types: `IEEEFloat`, `Complex{<:IEEEFloat}`, arrays thereof, or tuples of those.

The `!!` signals that the backend may mutate internal cache state. The caller owns the
returned values.

The non-caching form has a default that calls `prepare_jacobian_cache` then the cached form.

# Interface

Backends implement:

    ADInterface.prepare_jacobian_cache(f, ::MyBackend, x::Vararg{Any, N}) where {N} = MyJacobianCache(...)
    ADInterface.value_and_jacobian!!(cache::MyJacobianCache, f, x::Vararg{Any, N}) where {N} = ...

See also: [`value_and_gradient!!`](@ref), [`gradient_order`](@ref).
"""
function value_and_jacobian!! end

# ── Default non-caching forms (delegate to cache API) ─────────────────────────

function value_and_gradient!!(f::F, backend::AbstractADType, x::Vararg{Any, N}) where {F, N}
    cache = prepare_gradient_cache(f, backend, x...)
    return value_and_gradient!!(cache, f, x...)
end

function value_and_jacobian!!(f::F, backend::AbstractADType, x::Vararg{Any, N}) where {F, N}
    cache = prepare_jacobian_cache(f, backend, x...)
    return value_and_jacobian!!(cache, f, x...)
end

# ── Error fallbacks ────────────────────────────────────────────────────────────

function prepare_gradient_cache(f::F, ::T, x::Vararg{Any, N}) where {F, T <: AbstractADType, N}
    throw(
        ArgumentError(
            "`ADInterface.prepare_gradient_cache` is not implemented for backend `$T`. " *
                "Add a method:\n    ADInterface.prepare_gradient_cache(f, ::$T, x::Vararg{Any,N}) where {N} = MyCache(...)\n" *
                "and implement:\n    ADInterface.value_and_gradient!!(cache::MyCache, f, x::Vararg{Any,N}) where {N} = ..."
        ),
    )
end

function prepare_jacobian_cache(f::F, ::T, x::Vararg{Any, N}) where {F, T <: AbstractADType, N}
    throw(
        ArgumentError(
            "`ADInterface.prepare_jacobian_cache` is not implemented for backend `$T`. " *
                "Add a method:\n    ADInterface.prepare_jacobian_cache(f, ::$T, x::Vararg{Any,N}) where {N} = MyCache(...)\n" *
                "and implement:\n    ADInterface.value_and_jacobian!!(cache::MyCache, f, x::Vararg{Any,N}) where {N} = ..."
        ),
    )
end

function value_and_gradient!!(::AbstractGradientCache, f::F, x::Vararg{Any, N}) where {F, N}
    throw(
        ArgumentError(
            "No `ADInterface.value_and_gradient!!` method found for this cache type. " *
                "Implement:\n    ADInterface.value_and_gradient!!(cache::MyCacheType, f, x::Vararg{Any,N}) where {N} = ..."
        ),
    )
end

function value_and_jacobian!!(::AbstractGradientCache, f::F, x::Vararg{Any, N}) where {F, N}
    throw(
        ArgumentError(
            "No `ADInterface.value_and_jacobian!!` method found for this cache type. " *
                "Implement:\n    ADInterface.value_and_jacobian!!(cache::MyCacheType, f, x::Vararg{Any,N}) where {N} = ..."
        ),
    )
end

export GradientOrder,
    gradient_order,
    AbstractGradientCache,
    prepare_gradient_cache,
    prepare_jacobian_cache,
    value_and_gradient!!,
    value_and_jacobian!!

include("test_utils.jl")

end
