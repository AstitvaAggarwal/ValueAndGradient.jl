# ADKernel.jl

A minimal Julia interface for computing gradients and Jacobians across AD backends.

## Why this exists

[DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl) is the general solution. ADKernel is narrower: inputs are restricted to scalars, arrays, and tuples of floats, which means behavior can be fully pinned down and correctness verified automatically via FiniteDifferences. The payoff is a bundled `TestUtils` module that backend authors can run to check their implementation, similar to `Mooncake.test_rule`.

## Input types

Supported inputs for differentiable arguments:

- `Float32`, `Float64` (and other `IEEEFloat` types)
- `Complex{Float32}`, `Complex{Float64}`
- Arrays of any of the above (e.g. `Vector{Float64}`, `Matrix{ComplexF32}`)
- Tuples of any of the above

Multiple differentiable arguments are supported (pass them as extra positional args).

```julia
DiffScalar = Union{IEEEFloat, Complex{<:IEEEFloat}}
DiffInput  = Union{DiffScalar, AbstractArray{<:DiffScalar}, Tuple{Vararg{DiffLeaf}}}
```

## API

```julia
# derivative order
gradient_order(backend)  # returns GradientOrder{0}, GradientOrder{1}, or nothing

# One-shot (builds a fresh cache each call)
y, g  = value_and_gradient!!(f, backend, x)
y, gs = value_and_gradient!!(f, backend, x, y)   # multiple args: returns tuple of gradients
y, J  = value_and_jacobian!!(f, backend, x)

# Cached (amortises compilation cost over repeated calls)
cache = prepare_gradient_cache(f, backend, x)
y, g  = value_and_gradient!!(cache, f, x)

cache = prepare_jacobian_cache(f, backend, x)
y, J  = value_and_jacobian!!(cache, f, x)
```

The `!!` means the backend may write into the cache. The caller owns the returned values; copy them if you need to keep them past the next call.

## Backends

| Backend | Type | Order |
|---------|------|-------|
| [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) | Reverse-mode | `GradientOrder{1}` |
| [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) | Forward-mode | `GradientOrder{1}` |

Load via weak dependency:

```julia
using ADKernel, ADTypes, Mooncake

backend = AutoMooncake(config=nothing)
y, g = value_and_gradient!!(x -> sum(x .^ 2), backend, [1.0, 2.0, 3.0])
```

## Which backend to use

For gradients, always use `AutoMooncake` (one reverse pass regardless of input size).

For Jacobians, it depends on the shape of `f`:

| Case | Best choice | Cost |
|------|-------------|------|
| Scalar output | `AutoMooncake` | 1 reverse pass |
| More inputs than outputs (n > m) | `AutoMooncake` | m reverse passes |
| More outputs than inputs (m > n) | `AutoMooncakeForward` | n forward passes |

## Implementing a backend

```julia
# 1. Declare capability
ADKernel.gradient_order(::MyBackend) = GradientOrder{1}()

# 2. Build a cache
struct MyGradientCache <: ADKernel.AbstractGradientCache ... end
ADKernel.prepare_gradient_cache(f, ::MyBackend, x::Vararg{Any,N}) where {N} = MyGradientCache(...)

# 3. Implement the cached call
ADKernel.value_and_gradient!!(cache::MyGradientCache, f, x::Vararg{Any,N}) where {N} = ...
```

The non-cached forms call through automatically, so backends only need to implement the cached versions.

## Testing your backend

```julia
using ADKernel.TestUtils, ADTypes

backend = MyBackend()
test_value_and_gradient(x -> sum(x .^ 2), backend, [1.0, 2.0, 3.0])
test_value_and_gradient((x, y) -> sum(x .* y), backend, [1.0, 2.0], [3.0, 4.0])
test_value_and_jacobian(x -> x .^ 2, backend, [1.0, 2.0, 3.0])
```

Correctness is checked against finite differences, including the cached form and repeated calls.
