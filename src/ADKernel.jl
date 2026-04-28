module ADKernel

using ADTypes: ADTypes, AbstractADType

# Input types are restricted to IEEEFloat / Complex{IEEEFloat} / arrays / tuples thereof
# so behavior can be fully specified and correctness checked via FiniteDifferences.

const DiffScalar = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}
const DiffArray  = AbstractArray{<:DiffScalar}
const DiffLeaf   = Union{DiffScalar, DiffArray}
const DiffInput  = Union{DiffLeaf, Tuple{Vararg{DiffLeaf}}}

"""
    GradientOrder{K}

Tracks the highest derivative order a backend supports.
`0` for primal only, `1` for gradients/Jacobians, `2` for Hessians.

Declare support with `ADKernel.gradient_order(::MyBackend) = GradientOrder{1}()`.
Orders can be compared: `GradientOrder{0}() < GradientOrder{1}()`.
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
    gradient_order(backend) -> GradientOrder{K} or nothing

Returns the highest derivative order supported by `backend`, or `nothing` if the
backend does not implement this interface.
"""
gradient_order(::AbstractADType) = nothing

"""
    AbstractGradientCache

Supertype for caches returned by `prepare_gradient_cache` and `prepare_jacobian_cache`.
Subtype this and implement the cached `value_and_gradient!!` / `value_and_jacobian!!`.
"""
abstract type AbstractGradientCache end

"""
    prepare_gradient_cache(f, backend, x...) -> AbstractGradientCache

Build a cache for repeated `value_and_gradient!!` calls. Any compilation or
tape-building cost is paid once here, so subsequent calls are cheap.

Input types: `IEEEFloat`, `Complex{<:IEEEFloat}`, arrays thereof, or tuples of those.

Backends implement:

    ADKernel.prepare_gradient_cache(f, ::MyBackend, x::Vararg{Any, N}) where {N} = MyCache(...)

See also: [`prepare_jacobian_cache`](@ref), [`value_and_gradient!!`](@ref).
"""
function prepare_gradient_cache end

"""
    prepare_jacobian_cache(f, backend, x...) -> AbstractGradientCache

Build a cache for repeated `value_and_jacobian!!` calls.

See also: [`prepare_gradient_cache`](@ref), [`value_and_jacobian!!`](@ref).
"""
function prepare_jacobian_cache end

"""
    value_and_gradient!!(f, backend, x...)
    value_and_gradient!!(cache, f, x...)

Returns `(f(x...), gradient)`. For a single argument the gradient has the same structure
as `x`; for multiple arguments it is a tuple of per-argument gradients.

`!!` means the backend may write into the cache. The caller owns the returned values;
copy them if you need to keep them past the next call.

For repeated calls, build a cache with `prepare_gradient_cache` first. The non-cached
form calls through automatically so backends only need to implement the cached version.

See also: [`value_and_jacobian!!`](@ref), [`gradient_order`](@ref).
"""
function value_and_gradient!! end

"""
    value_and_jacobian!!(f, backend, x...)
    value_and_jacobian!!(cache, f, x...)

Returns `(f(x...), Jacobian)`. For scalar-valued `f` this is the same as
`value_and_gradient!!`. For vector-valued `f` it returns the full m-by-n Jacobian matrix.

Which backend to use depends on the shape of the Jacobian:
- Scalar output: use `AutoMooncake` (one reverse pass, cheapest).
- More inputs than outputs (n > m): use `AutoMooncake` (m reverse passes vs n forward passes).
- More outputs than inputs (m > n): use `AutoMooncakeForward` (n forward passes vs m reverse passes).

See also: [`value_and_gradient!!`](@ref), [`gradient_order`](@ref).
"""
function value_and_jacobian!! end

# non-caching forms delegate to the cache API

function value_and_gradient!!(f::F, backend::AbstractADType, x::Vararg{Any, N}) where {F, N}
    cache = prepare_gradient_cache(f, backend, x...)
    return value_and_gradient!!(cache, f, x...)
end

function value_and_jacobian!!(f::F, backend::AbstractADType, x::Vararg{Any, N}) where {F, N}
    cache = prepare_jacobian_cache(f, backend, x...)
    return value_and_jacobian!!(cache, f, x...)
end

# error fallbacks

function prepare_gradient_cache(f::F, ::T, x::Vararg{Any, N}) where {F, T <: AbstractADType, N}
    throw(
        ArgumentError(
            "`ADKernel.prepare_gradient_cache` is not implemented for backend `$T`. " *
                "Add a method:\n    ADKernel.prepare_gradient_cache(f, ::$T, x::Vararg{Any,N}) where {N} = MyCache(...)\n" *
                "and implement:\n    ADKernel.value_and_gradient!!(cache::MyCache, f, x::Vararg{Any,N}) where {N} = ..."
        ),
    )
end

function prepare_jacobian_cache(f::F, ::T, x::Vararg{Any, N}) where {F, T <: AbstractADType, N}
    throw(
        ArgumentError(
            "`ADKernel.prepare_jacobian_cache` is not implemented for backend `$T`. " *
                "Add a method:\n    ADKernel.prepare_jacobian_cache(f, ::$T, x::Vararg{Any,N}) where {N} = MyCache(...)\n" *
                "and implement:\n    ADKernel.value_and_jacobian!!(cache::MyCache, f, x::Vararg{Any,N}) where {N} = ..."
        ),
    )
end

function value_and_gradient!!(::AbstractGradientCache, f::F, x::Vararg{Any, N}) where {F, N}
    throw(
        ArgumentError(
            "No `ADKernel.value_and_gradient!!` method found for this cache type. " *
                "Implement:\n    ADKernel.value_and_gradient!!(cache::MyCacheType, f, x::Vararg{Any,N}) where {N} = ..."
        ),
    )
end

function value_and_jacobian!!(::AbstractGradientCache, f::F, x::Vararg{Any, N}) where {F, N}
    throw(
        ArgumentError(
            "No `ADKernel.value_and_jacobian!!` method found for this cache type. " *
                "Implement:\n    ADKernel.value_and_jacobian!!(cache::MyCacheType, f, x::Vararg{Any,N}) where {N} = ..."
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
