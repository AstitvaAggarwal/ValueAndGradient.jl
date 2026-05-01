# ValueAndGradient.jl

A minimal, backend-agnostic Julia interface for VJPs and JVPs.

## Why this exists

SciML, Turing, and Lux all need to call VJPs and JVPs, and each ships its own wrappers around Mooncake, Enzyme, Zygote, etc. This package defines a single shared interface (`value_and_pullback!!` and `value_and_pushforward!!`) so backends implement it once and callers depend on it instead of on any specific AD package.

The input/output scope is intentionally narrow (scalars, arrays, tuples of floats) so correctness can be checked via FiniteDifferences.

## Input types

Supported differentiable inputs:

- `Float32`, `Float64` (and other `IEEEFloat` types)
- `Complex{Float32}`, `Complex{Float64}`
- Arrays of any of the above
- Tuples of any of the above
- Multiple differentiable arguments (passed as extra positional args)

## API

```julia
# VJP: returns (y, x̄) where x̄ = (∂f/∂x)ᵀ ȳ
y, x̄  = value_and_pullback!!(f, ȳ, backend, x)
y, x̄s = value_and_pullback!!(f, ȳ, backend, x1, x2)  # multiple args: x̄s is a tuple

# JVP: returns (y, ẏ) where ẏ = ∂f/∂x * ẋ
y, ẏ = value_and_pushforward!!(f, ẋ, backend, x)
y, ẏ = value_and_pushforward!!(f, (ẋ1, ẋ2), backend, x1, x2)
```

The caller controls the seed `ȳ` in `value_and_pullback!!`. SciML passes adjoint state, Turing passes importance weights, Lux passes cotangents from the layer above. Gradients (seed = 1) and full Jacobians are both derivable by the caller.

## Backends

| Backend | Type | ADTypes struct |
|---------|------|----------------|
| [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) | Reverse-mode (pullback) | `AutoMooncake` |
| [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) | Forward-mode (pushforward) | `AutoMooncakeForward` |

```julia
using ValueAndGradient, ADTypes, Mooncake

# Reverse-mode VJP
y, x̄ = value_and_pullback!!(x -> sum(x .^ 2), 1.0, AutoMooncake(config=nothing), [1.0, 2.0, 3.0])

# Forward-mode JVP
y, ẏ = value_and_pushforward!!(x -> x .^ 2, [1.0, 0.0, 0.0], AutoMooncakeForward(config=nothing), [1.0, 2.0, 3.0])
```

## Implementing a new backend

```julia
function ValueAndGradient.value_and_pullback!!(f, ȳ, ::MyBackend, x::Vararg{Any,N}) where {N}
    # compute and return (f(x...), cotangents)
end

function ValueAndGradient.value_and_pushforward!!(f, ẋ, ::MyBackend, x::Vararg{Any,N}) where {N}
    # compute and return (f(x...), tangent)
end
```

## Testing your backend

```julia
using ValueAndGradient.TestUtils

test_pullback(x -> sum(x .^ 2), 1.0, MyBackend(), [1.0, 2.0, 3.0])
test_pullback(x -> x .^ 2, [2.0, -1.0, 3.0], MyBackend(), [1.0, 2.0, 3.0])
test_pushforward(x -> x .^ 2, [1.0, 0.0, 0.0], MyBackend(), [1.0, 2.0, 3.0])
```

Correctness is checked against finite differences.
