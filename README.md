# ValueAndGradient.jl

A minimal, backend-agnostic Julia interface for VJPs and JVPs.

## Why this exists

SciML, Turing, and Lux all need to call VJPs and JVPs, and each ships its own wrappers around Mooncake, Enzyme, Zygote, etc. This package defines a single shared interface (`value_and_pullback!!` and `value_and_pushforward!!`) so backends implement it once and callers depend on it instead of on any specific AD package.

The input/output scope is narrow (scalars, arrays, tuples of floats) so correctness can be checked via FiniteDifferences.

## Input and output types

Supported differentiable inputs and outputs:

- `Float32`, `Float64` (and other `IEEEFloat` types)
- `Complex{Float32}`, `Complex{Float64}`
- Arrays of any of the above
- Tuples of any of the above
- Multiple differentiable arguments (passed as extra positional args)

`ȳ` in `value_and_pullback!!` must match the output type of `f`. `ẏ` returned by `value_and_pushforward!!` has the same structure as `f`'s output.

## API

```julia
# VJP: returns (y, x̄) where x̄ = (∂f/∂x)ᵀ ȳ
y, x̄  = value_and_pullback!!(f, ȳ, backend, x)
y, x̄s = value_and_pullback!!(f, ȳ, backend, x1, x2)  # multiple args: x̄s is a tuple

# JVP: returns (y, ẏ) where ẏ = ∂f/∂x * ẋ
y, ẏ = value_and_pushforward!!(f, ẋ, backend, x)
y, ẏ = value_and_pushforward!!(f, (ẋ1, ẋ2), backend, x1, x2)
```

The caller controls the seed `ȳ` in `value_and_pullback!!`. SciML passes adjoint state, Turing passes importance weights, Lux passes cotangents from the layer above. Gradients (seed = 1) and full Jacobians follow from this.

## Caching

For repeated calls, build the backend cache once and pass it via the `cache` keyword:

```julia
cache = Mooncake.prepare_pullback_cache(f, x)   # compile/prepare once
y, x̄ = value_and_pullback!!(f, ȳ, backend, x; cache)   # cheap from here on
```

Cache preparation is the backend's responsibility — backends use it if provided and fall back to building one internally if not. Stateless backends (e.g. finite differences) ignore the keyword entirely.

## Backends

| Backend | Type | ADTypes struct |
|---------|------|----------------|
| [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) | Reverse-mode (pullback) | `AutoMooncake` |
| [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) | Forward-mode (pushforward) | `AutoMooncakeForward` |

```julia
using ValueAndGradient, ADTypes, Mooncake

# one-shot
f = x -> sum(x .^ 2)
y, x̄ = value_and_pullback!!(f, 1.0, AutoMooncake(config=nothing), [1.0, 2.0, 3.0])

# cached (build rule once, reuse across calls)
cache = Mooncake.prepare_pullback_cache(f, [1.0, 2.0, 3.0])
y, x̄ = value_and_pullback!!(f, 1.0, AutoMooncake(config=nothing), [1.0, 2.0, 3.0]; cache)
```

## Implementing a new backend

```julia
function ValueAndGradient.value_and_pullback!!(f, ȳ, ::MyBackend, x::Vararg{Any,N}; cache=nothing) where {N}
    c = cache !== nothing ? cache : build_my_pullback_cache(f, x...)
    # compute and return (f(x...), cotangents) using c
end

function ValueAndGradient.value_and_pushforward!!(f, ẋ, ::MyBackend, x::Vararg{Any,N}; cache=nothing) where {N}
    c = cache !== nothing ? cache : build_my_derivative_cache(f, x...)
    # compute and return (f(x...), tangent) using c
end
```

## Testing your backend

Load `FiniteDifferences` and `Test` to enable the test utilities:

```julia
using ValueAndGradient, FiniteDifferences, Test

test_pullback(x -> sum(x .^ 2), 1.0, MyBackend(), [1.0, 2.0, 3.0])
test_pullback(x -> x .^ 2, [2.0, -1.0, 3.0], MyBackend(), [1.0, 2.0, 3.0])
test_pullback(x -> (x[1]^2, x[2]^2), (1.0, 1.0), MyBackend(), [1.0, 2.0])
test_pushforward(x -> x .^ 2, [1.0, 0.0, 0.0], MyBackend(), [1.0, 2.0, 3.0])
test_pushforward(x -> (x[1]^2, x[2]^2), [1.0, 1.0], MyBackend(), [1.0, 2.0])
```

Correctness is checked against finite differences.
