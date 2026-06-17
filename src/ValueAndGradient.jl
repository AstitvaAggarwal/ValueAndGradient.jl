module ValueAndGradient

using ADTypes: AbstractADType

const DiffScalar = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}
const DiffArray = AbstractArray{<:DiffScalar}
const DiffLeaf = Union{DiffScalar, DiffArray}
const DiffInput = Union{DiffLeaf, Tuple{Vararg{DiffLeaf}}}

"""
    value_and_pullback!!(f, ȳ, backend, x...; ad_cache=nothing, canonical_tangents=false) -> (y, x̄)

Returns `y = f(x...)` and the VJP `x̄ = (∂f/∂x)ᵀ ȳ`.
`ȳ` must match the output type of `f`: scalar, array, or tuple thereof.
Single argument: `x̄` has the same structure as `x`.
Multiple arguments: `x̄` is a tuple of per-argument cotangents.

Pass a backend-specific `ad_cache` to reuse it across repeated calls.
If `nothing`, the backend builds one internally (convenience path — not efficient for loops).

`canonical_tangents=false` (default): output tangents are normalised to standard Julia types.
`canonical_tangents=true`: raw backend tangent types are returned. Only has effect when
`ad_cache=nothing`; if `ad_cache` is provided, the output type is determined by the cache.
"""
function value_and_pullback!! end

"""
    value_and_pushforward!!(f, ẋ, backend, x...; ad_cache=nothing, canonical_tangents=false) -> (y, ẏ)

Returns `y = f(x...)` and the JVP `ẏ = ∂f/∂x * ẋ`.
`ẏ` matches the output type of `f`: scalar, array, or tuple thereof.
Single argument: `ẋ` has the same structure as `x`.
Multiple arguments: `ẋ` is a tuple of per-argument tangents.

See `value_and_pullback!!` for `ad_cache` and `canonical_tangents` semantics.
"""
function value_and_pushforward!! end

"""
    test_pullback(f, ȳ, backend, xs...; rtol=1e-5, atol=1e-5)

Check `value_and_pullback!!` against finite differences.
Requires `FiniteDifferences` to be loaded.
"""
function test_pullback end

"""
    test_pushforward(f, ẋ, backend, xs...; rtol=1e-5, atol=1e-5)

Check `value_and_pushforward!!` against finite differences.
Requires `FiniteDifferences` to be loaded.
"""
function test_pushforward end

function value_and_pullback!!(f, ȳ, backend::AbstractADType, xs...; kwargs...)
    throw(
        ArgumentError(
            "value_and_pullback!! is not supported for $(typeof(backend)). " *
                "Use AutoMooncake, or use value_and_pushforward!! for forward-mode backends.",
        ),
    )
end

function value_and_pushforward!!(f, ẋ, backend::AbstractADType, xs...; kwargs...)
    throw(
        ArgumentError(
            "value_and_pushforward!! is not supported for $(typeof(backend)). " *
                "Use AutoMooncakeForward, or use value_and_pullback!! for reverse-mode backends.",
        ),
    )
end

export value_and_pullback!!,
    value_and_pushforward!!,
    test_pullback,
    test_pushforward

end
