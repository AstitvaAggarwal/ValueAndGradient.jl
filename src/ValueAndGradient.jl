module ValueAndGradient

using ADTypes: AbstractADType

const DiffScalar = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}
const DiffArray  = AbstractArray{<:DiffScalar}
const DiffLeaf   = Union{DiffScalar, DiffArray}
const DiffInput  = Union{DiffLeaf, Tuple{Vararg{DiffLeaf}}}

"""
    value_and_pullback!!(f, ȳ, backend, x...) -> (y, x̄)

Returns `y = f(x...)` and the VJP `x̄ = (∂f/∂x)ᵀ ȳ`.
`ȳ` must match the output type of `f`: scalar, array, or tuple thereof.
Single argument: `x̄` has the same structure as `x`.
Multiple arguments: `x̄` is a tuple of per-argument cotangents.
"""
function value_and_pullback!! end

"""
    value_and_pushforward!!(f, ẋ, backend, x...) -> (y, ẏ)

Returns `y = f(x...)` and the JVP `ẏ = ∂f/∂x * ẋ`.
`ẏ` matches the output type of `f`: scalar, array, or tuple thereof.
Single argument: `ẋ` has the same structure as `x`.
Multiple arguments: `ẋ` is a tuple of per-argument tangents.
"""
function value_and_pushforward!! end

export value_and_pullback!!, value_and_pushforward!!

include("test_utils.jl")

end
