# ValueAndGradient.jl

A minimal, backend-agnostic Julia interface for VJPs and JVPs.

## Why this exists

SciML, Turing, and Lux all need to call VJPs and JVPs, and each has historically shipped its own thin wrappers around Mooncake, Enzyme, Zygote, etc. This package defines a single shared interface — `value_and_pullback!!` and `value_and_pushforward!!` — so backends implement it once and callers depend on it instead of on any specific AD package.

The input/output scope is intentionally narrow (scalars, arrays, tuples of floats) so that correctness can be fully verified automatically via FiniteDifferences.

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

# Cached forms (amortise compilation cost over repeated calls)
cache = prepare_pullback_cache(f, backend, x)
y, x̄ = value_and_pullback!!(cache, f, ȳ, x)

cache = prepare_pushforward_cache(f, backend, x)
y, ẏ = value_and_pushforward!!(cache, f, ẋ, x)

# Capability query
gradient_order(backend)  # returns GradientOrder{1} or nothing
```

The `!!` means the backend may write into the cache. Copy returned values if you need them past the next call.

The caller controls the seed `ȳ` in `value_and_pullback!!` — this is the key design choice. SciML passes adjoint state, Turing passes importance weights, Lux passes cotangents from the layer above. `value_and_gradient!!` (seed = 1) and full Jacobians are both derivable from `value_and_pullback!!` by the caller.

## Backends

| Backend | Type | ADTypes struct |
|---------|------|----------------|
| [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) | Reverse-mode (pullback) | `AutoMooncake` |
| [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) | Forward-mode (pushforward) | `AutoMooncakeForward` |

```julia
using ValueAndGradient, ADTypes, Mooncake

# Reverse-mode VJP
backend = AutoMooncake(config=nothing)
y, x̄ = value_and_pullback!!(x -> sum(x .^ 2), 1.0, backend, [1.0, 2.0, 3.0])

# Forward-mode JVP
fwd = AutoMooncakeForward(config=nothing)
y, ẏ = value_and_pushforward!!(x -> x .^ 2, [1.0, 0.0, 0.0], fwd, [1.0, 2.0, 3.0])
```

## Implementing a new backend

```julia
# 1. Declare capability
ValueAndGradient.gradient_order(::MyBackend) = GradientOrder{1}()

# 2. Build caches
struct MyPullbackCache <: ValueAndGradient.AbstractADCache ... end
ValueAndGradient.prepare_pullback_cache(f, ::MyBackend, x::Vararg{Any,N}) where {N} = MyPullbackCache(...)

struct MyPushforwardCache <: ValueAndGradient.AbstractADCache ... end
ValueAndGradient.prepare_pushforward_cache(f, ::MyBackend, x::Vararg{Any,N}) where {N} = MyPushforwardCache(...)

# 3. Implement the cached calls
ValueAndGradient.value_and_pullback!!(cache::MyPullbackCache, f, ȳ, x::Vararg{Any,N}) where {N} = ...
ValueAndGradient.value_and_pushforward!!(cache::MyPushforwardCache, f, ẋ, x::Vararg{Any,N}) where {N} = ...
```

Non-cached forms call through automatically.

## Testing your backend

```julia
using ValueAndGradient.TestUtils

backend = MyBackend()

# Check VJP with seed ȳ=1.0
test_pullback(x -> sum(x .^ 2), 1.0, backend, [1.0, 2.0, 3.0])

# Check VJP with non-unit seed (verifies ȳ is actually used)
test_pullback(x -> x .^ 2, [2.0, -1.0, 3.0], backend, [1.0, 2.0, 3.0])

# Check JVP
test_pushforward(x -> x .^ 2, [1.0, 0.0, 0.0], backend, [1.0, 2.0, 3.0])
```

Correctness is checked against finite differences, including the cached form.
