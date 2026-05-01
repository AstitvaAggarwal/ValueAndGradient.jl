module ValueAndGradient

using ADTypes: AbstractADType

const DiffScalar = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}
const DiffArray  = AbstractArray{<:DiffScalar}
const DiffLeaf   = Union{DiffScalar, DiffArray}
const DiffInput  = Union{DiffLeaf, Tuple{Vararg{DiffLeaf}}}

"""
    value_and_pullback!!(f, ȳ, backend, x...) -> (y, x̄)
    value_and_pullback!!(fc::FWithCache, ȳ, backend, x...) -> (y, x̄)

Returns `y = f(x...)` and the VJP `x̄ = (∂f/∂x)ᵀ ȳ`.
`ȳ` must match the output type of `f`: scalar, array, or tuple thereof.
Single argument: `x̄` has the same structure as `x`.
Multiple arguments: `x̄` is a tuple of per-argument cotangents.

Pass an `FWithCache` as the first argument to reuse a pre-built backend cache.
"""
function value_and_pullback!! end

"""
    value_and_pushforward!!(f, ẋ, backend, x...) -> (y, ẏ)
    value_and_pushforward!!(fc::FWithCache, ẋ, backend, x...) -> (y, ẏ)

Returns `y = f(x...)` and the JVP `ẏ = ∂f/∂x * ẋ`.
`ẏ` matches the output type of `f`: scalar, array, or tuple thereof.
Single argument: `ẋ` has the same structure as `x`.
Multiple arguments: `ẋ` is a tuple of per-argument tangents.

Pass an `FWithCache` as the first argument to reuse a pre-built backend cache.
"""
function value_and_pushforward!! end

"""
    FWithCache(f, backend, x...)

Pairs `f` with a pre-built backend cache for repeated calls.
Build once with the backend and a representative input, then pass in place of `f`:

    fc = FWithCache(f, AutoMooncake(config=nothing), x)
    y, x̄ = value_and_pullback!!(fc, ȳ, AutoMooncake(config=nothing), x)
"""
struct FWithCache{F, C}
    f::F
    cache::C
end

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

export value_and_pullback!!,
    value_and_pushforward!!,
    FWithCache,
    test_pullback,
    test_pushforward

end
