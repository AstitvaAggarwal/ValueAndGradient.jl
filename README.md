# ValueAndGradient.jl

A minimal, backend-agnostic Julia interface for first-order automatic differentiation.

## Why this exists

SciML, Turing, and Lux all need to call VJPs and JVPs, and each ships its own wrappers around Mooncake, Enzyme, Zygote, etc. This package defines a single shared interface so backends implement it once and callers depend on it instead of any specific AD package.

The design principle is a thin wrapper: if the backend exposes the operation as public API, VG.jl calls it directly. It never reaches into backend internals, which is why it does not break on every backend release.

## Operations

Five first-order operations are implemented:

| Operation | Returns | Description | Status |
|---|---|---|---|
| `value_and_pullback!!(f, ȳ, backend, x...)` | `(y, x̄)` | VJP — `x̄ = ȳᵀ · ∂f/∂x` | ✅ |
| `value_and_pushforward!!(f, ẋ, backend, x...)` | `(y, ẏ)` | JVP — `ẏ = ∂f/∂x · ẋ` | ✅ |
| `value_and_gradient!!(f, backend, x...)` | `(y, ∇f)` | Gradient — `f` must be scalar-valued | ✅ |
| `value_and_jacobian!!(f, backend, x...)` | `(y, (J1,...))` | Full Jacobian; always returns tuple of per-arg Jacobians | ✅ |
| `value_and_derivative!!(f, backend, x)` | `(y, ẏ)` | Scalar-input derivative — `x::DiffScalar`; forward-mode only | ✅ |

All operations accept an optional `ad_cache` keyword (see Caching below) and `canonical_tangents` keyword.

## Input types

VG.jl constrains **inputs only**, not outputs:

```julia
DiffScalar = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}
DiffArray  = AbstractArray{<:DiffScalar}
DiffLeaf   = Union{DiffScalar, DiffArray}
DiffInput  = Union{DiffLeaf, Tuple{Vararg{DiffLeaf}}}
```

Single-argument calls accept any `DiffInput` (scalar, array, tuple of scalars/arrays). Multi-argument calls accept two or more `DiffLeaf` arguments. `value_and_jacobian!!` accepts any `DiffArray` input (real or complex, any shape); multi-arg `value_and_jacobian!!` accepts two or more `DiffLeaf` arguments and returns a tuple of per-argument Jacobians.

## Output types

No constraint. `f` may return scalars, arrays, tuples, NamedTuples, structs, complex values — VG.jl passes everything through to the backend unchanged. The only exception is `value_and_gradient!!`, where the backend (Mooncake) enforces a scalar `IEEEFloat` output.

## API

```julia
using ValueAndGradient, ADTypes, Mooncake

f  = x -> sum(x .^ 2)
x  = [1.0, 2.0, 3.0]

# Pullback (VJP)
y, x̄  = value_and_pullback!!(f, 1.0, AutoMooncake(config=nothing), x)

# Multi-arg pullback
g = (x, y) -> sum(x .* y)
y, (x̄, ȳ) = value_and_pullback!!(g, 1.0, AutoMooncake(config=nothing), [1.0,2.0], [3.0,4.0])

# Pushforward (JVP)
y, ẏ = value_and_pushforward!!(f, ones(3), AutoMooncakeForward(config=nothing), x)

# Gradient
y, ∇f = value_and_gradient!!(f, AutoMooncake(config=nothing), x)

# Jacobian (single-arg: always returns (y, (J,)) — note the tuple destructuring)
h  = x -> [x[1]^2 + x[2], x[2]^2 - x[1]]
y, (J,) = value_and_jacobian!!(h, AutoMooncake(config=nothing), [2.0, 3.0])

# Multi-arg jacobian
g2 = (x, y) -> x .* y
y2, (Jx, Jy) = value_and_jacobian!!(g2, AutoMooncake(config=nothing), [1.0, 2.0], [3.0, 4.0])

# Derivative (forward-mode only; x must be a scalar)
y, ẏ = value_and_derivative!!(x -> x^3, AutoMooncakeForward(config=nothing), 2.0)
```

## Caching

For repeated calls (training loops, solvers), build the backend cache once outside the loop:

```julia
cache = Mooncake.prepare_pullback_cache(f, x)
for _ in 1:10_000
    y, x̄ = value_and_pullback!!(f, 1.0, AutoMooncake(config=nothing), x; ad_cache=cache)
end
```

`ad_cache=nothing` (default) builds a fresh backend cache on every call — convenient for one-off use, not suitable for hot paths. When `ad_cache` is provided, VG.jl passes it directly to the backend with zero overhead.

## Backends

## Jacobian paths

`value_and_jacobian!!` uses two dispatch layers:

- **Layer 1 (native):** `AbstractVector{<:IEEEFloat}` with `AutoMooncake`/`AutoMooncakeForward` → calls `Mooncake.value_and_jacobian!!` directly. No warnings, minimum overhead.
- **Layer 2 (derived):** everything else (matrix inputs, complex arrays, multi-arg, other backends) → VG.jl builds J from repeated primitive calls. Emits a `@warn` once per call with pass count and n/m dimensions; includes a tip if the other mode direction would be more efficient.
  - Forward mode: n × `value_and_pushforward!!` (one column per input dim, basis tangent shaped like `x`)
  - Reverse mode: 1 forward eval + m × `value_and_pullback!!` (one row per output dim)

For hot loops, pass `ad_cache` to avoid rebuilding the backend cache on every call.

| Backend | ADTypes struct | Mode | Ops |
|---|---|---|---|
| Mooncake.jl | `AutoMooncake` | reverse | pullback, gradient, jacobian |
| Mooncake.jl | `AutoMooncakeForward` | forward | pushforward, jacobian, derivative |
| ForwardDiff.jl | `AutoForwardDiff` | forward | *(planned)* |
| Zygote.jl | `AutoZygote` | reverse | *(planned)* |
| Enzyme.jl | `AutoEnzyme` | both | *(planned)* |
| FiniteDifferences.jl | `AutoFiniteDifferences` | numerical | *(planned)* |

## Backend API reference (verified from source)

Each backend's native first-order API, for reference when implementing new extensions:

| Op | ForwardDiff | Zygote | Enzyme | Mooncake | FiniteDifferences |
|---|---|---|---|---|---|
| Pullback | ❌ | `pullback(f,x)→(y,back)` | `autodiff(Reverse,...)` | `value_and_pullback!!` (@public) | `j′vp` |
| Pushforward | ❌ named | `pushforward(f,x)→callable` | `autodiff(Forward,...)` | `value_and_derivative!!` (general JVP) | `jvp` |
| Gradient | `gradient` | `gradient` | `gradient` | `value_and_gradient!!` | `grad` |
| Jacobian | `jacobian` | `jacobian` | `jacobian` | `value_and_jacobian!!` | `jacobian` |
| Derivative | `derivative` (scalar) | ❌ | ❌ | ⚠️ general JVP only | ⚠️ unnamed |

## Implementing a new backend

Define methods for the operations your backend supports:

```julia
# In MyPackage/ext/ValueAndGradientMyBackendExt.jl

using ValueAndGradient: ValueAndGradient, DiffInput, DiffLeaf

function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMyBackend, x::DiffInput;
        ad_cache=nothing, canonical_tangents=false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache : build_cache(f, x)
    y, x̄ = my_vjp(c, ȳ, f, x)
    return y, x̄
end

function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMyBackend, x1::DiffLeaf, x2::DiffLeaf, xrest::DiffLeaf...;
        ad_cache=nothing, canonical_tangents=false,
    ) where {F}
    xs = (x1, x2, xrest...)
    c  = ad_cache !== nothing ? ad_cache : build_cache(f, xs...)
    y, x̄s = my_vjp(c, ȳ, f, xs...)
    return y, x̄s
end
```

## Testing a new backend

Load `FiniteDifferences` and `Test` to enable test utilities. All helpers compare AD results against independent finite-difference computations:

```julia
using ValueAndGradient, FiniteDifferences, Test

backend = AutoMyBackend()

test_pullback(x -> sum(x .^ 2), 1.0, backend, [1.0, 2.0, 3.0])
test_pullback(x -> x .^ 2, [2.0, -1.0, 3.0], backend, [1.0, 2.0, 3.0])
test_pullback(x -> (a=sum(x.^2), b=x[1]), (a=1.0, b=1.0), backend, [1.0, 2.0])  # NamedTuple output
test_pullback((x, y) -> sum(x .* y), 1.0, backend, [1.0, 2.0], [3.0, 4.0])      # multi-arg

test_pushforward(x -> x .^ 2, [1.0, 0.0, 0.0], backend, [1.0, 2.0, 3.0])
test_pushforward(x -> (a=sum(x.^2), b=x[1]), ones(3), backend, [1.0, 2.0, 3.0]) # NamedTuple output

test_gradient(x -> sum(x .^ 2), backend, [1.0, 2.0, 3.0])
test_jacobian(x -> [x[1]^2 + x[2], x[2]^2 - x[1]], backend, [2.0, 3.0])
```
