# ValueAndGradient.jl

A minimal, backend-agnostic Julia interface for first-order automatic differentiation.

## Why this exists

SciML, Turing, and Lux all need to call VJPs and JVPs, and each ships its own wrappers around Mooncake, Enzyme, Zygote, etc. VG.jl defines a single shared interface so backends implement it once and callers depend on it instead of any specific AD package.

The design principle is a thin wrapper: if the backend exposes the operation as public API, VG.jl calls it directly. It never reaches into backend internals, which is why it does not break on every backend release.

## Operations

Two primitive first-order operations:

| Operation | Returns | Description |
|---|---|---|
| `value_and_pullback!!(f, ȳ, backend, x...)` | `(y, x̄)` | VJP — `x̄ = (∂f/∂x)ᵀ ȳ` |
| `value_and_pushforward!!(f, ẋ, backend, x...)` | `(y, ẏ)` | JVP — `ẏ = ∂f/∂x · ẋ` |

Both accept optional keyword arguments:
- `ad_cache=nothing` — pass a pre-built backend cache to avoid rebuilding it on every call (see [Caching](#caching))
- `canonical_tangents=false` — when `true`, normalises backend-specific tangent types to standard Julia types (see [Canonical tangents](#canonical-tangents))

## Input types

VG.jl constrains **inputs only**, not outputs:

```julia
DiffScalar = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}
DiffArray  = AbstractArray{<:DiffScalar}
DiffLeaf   = Union{DiffScalar, DiffArray}
DiffInput  = Union{DiffLeaf, Tuple{Vararg{DiffLeaf}}}
```

Single-argument calls accept any `DiffInput` (scalar, array, or tuple of scalars/arrays).
Multi-argument calls accept two or more `DiffLeaf` arguments.

## Output types

No constraint on `f`'s return type. Scalars, arrays, tuples, NamedTuples, structs, complex values — VG.jl and most backends pass them through. The `_vdot` helper used internally for derived fallback paths handles `Number`, `AbstractArray`, `Tuple`, and `NamedTuple` outputs.

## API

```julia
using ValueAndGradient, ADTypes, Mooncake

# Pullback (VJP) — scalar input
f = x -> x^2
y, x̄ = value_and_pullback!!(f, 1.0, AutoMooncake(config=nothing), 3.0)
# y = 9.0, x̄ = 6.0

# Pullback (VJP) — array input
f = x -> sum(x .^ 2)
y, x̄ = value_and_pullback!!(f, 1.0, AutoMooncake(config=nothing), [1.0, 2.0, 3.0])
# y = 14.0, x̄ = [2.0, 4.0, 6.0]

# Pullback — array output (ȳ must match output shape)
f = x -> x .^ 2
y, x̄ = value_and_pullback!!(f, [2.0, -1.0, 3.0], AutoMooncake(config=nothing), [1.0, 2.0, 3.0])

# Pullback — multi-arg
g = (x, y) -> sum(x .* y)
val, (x̄, ȳ) = value_and_pullback!!(g, 1.0, AutoMooncake(config=nothing), [1.0, 2.0], [3.0, 4.0])

# Pushforward (JVP)
f = x -> x .^ 2
y, ẏ = value_and_pushforward!!(f, ones(3), AutoMooncakeForward(config=nothing), [1.0, 2.0, 3.0])
# y = [1.0, 4.0, 9.0], ẏ = [2.0, 4.0, 6.0]

# Pushforward — multi-arg (ẋ is a tuple of per-arg tangents)
g = (x, y) -> sum(x .* y)
y, ẏ = value_and_pushforward!!(g, ([1.0, 0.0], [0.0, 1.0]), AutoMooncakeForward(config=nothing), [1.0, 2.0], [3.0, 4.0])

# Tuple output
f = x -> (x[1]^2, x[2]^2)
y, x̄ = value_and_pullback!!(f, (1.0, 1.0), AutoMooncake(config=nothing), [1.0, 2.0])

# NamedTuple output
f = x -> (a = sum(x .^ 2), b = x[1] + x[2])
y, x̄ = value_and_pullback!!(f, (a = 1.0, b = 1.0), AutoMooncake(config=nothing), [1.0, 2.0, 3.0])
```

## Caching

For repeated calls (training loops, solvers), build the backend cache once outside the loop:

```julia
# Mooncake pullback
cache = Mooncake.prepare_pullback_cache(f, x)
for _ in 1:10_000
    y, x̄ = value_and_pullback!!(f, 1.0, AutoMooncake(config=nothing), x; ad_cache=cache)
end

# Mooncake pushforward
cache = Mooncake.prepare_derivative_cache(f, x)
for _ in 1:10_000
    y, ẏ = value_and_pushforward!!(f, ẋ, AutoMooncakeForward(config=nothing), x; ad_cache=cache)
end

# ReverseDiff compiled tape (single-arg)
tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, x))
y, x̄ = value_and_pullback!!(f, 1.0, AutoReverseDiff(), x; ad_cache=tape)

# FiniteDiff gradient cache
cache = FiniteDiff.GradientCache(similar(x), x)
y, x̄ = value_and_pullback!!(f, 1.0, AutoFiniteDiff(), x; ad_cache=cache)
```

`ad_cache=nothing` (default) builds a fresh backend cache on every call — convenient for one-off use, not suitable for hot paths. Stateless backends (Zygote, FiniteDifferences, Tracker, ForwardDiff, Enzyme) ignore `ad_cache` with a warning.

## Canonical tangents

When `canonical_tangents=true`, VG.jl normalises backend-specific tangent types to standard Julia values before returning. Useful when downstream code should not need to know about `Mooncake.Tangent` or Zygote's `nothing` cotangents.

| Tangent type | Canonical form |
|---|---|
| `nothing` (Zygote unused-argument cotangent) | `zero(x)` |
| `Mooncake.Tangent{NT}` (struct output from Mooncake pushforward) | fields NamedTuple `NT`, then tries `T(values(NT)...)` |
| Struct `T` with a matching positional constructor `T(fields...)` | `T` instance |
| All other types | unchanged |

```julia
# Zygote: f ignores y → cotangent would be nothing → zero(y) with canonical_tangents=true
f = (x, y) -> sum(x .^ 2)
_, x̄s = value_and_pullback!!(f, 1.0, AutoZygote(), x, y; canonical_tangents=true)
# x̄s[2] == zero(y)  instead of nothing

# Mooncake pushforward with struct output → reconstructed struct
struct MyPair{T}; a::T; b::T; end
f = x -> MyPair(sum(x .^ 2), x[1] + x[2])
_, ẏ = value_and_pushforward!!(f, ones(3), AutoMooncakeForward(config=nothing), [1.0, 2.0, 3.0];
                                canonical_tangents=true)
# ẏ isa MyPair{Float64}
```

If `T` has no matching positional constructor, VG.jl falls back to returning the raw `NamedTuple` and emits a `@warn`.

## Backends

| Backend | ADTypes struct | Mode | Pullback | Pushforward |
|---|---|---|---|---|
| Mooncake.jl | `AutoMooncake` | reverse | ✅ native | ⚠️ derived |
| Mooncake.jl | `AutoMooncakeForward` | forward | ⚠️ derived | ✅ native |
| ForwardDiff.jl | `AutoForwardDiff` | forward | ⚠️ derived | ✅ native |
| ReverseDiff.jl | `AutoReverseDiff` | reverse | ✅ native | ⚠️ derived |
| Tracker.jl | `AutoTracker` | reverse | ✅ native | ⚠️ derived |
| Zygote.jl | `AutoZygote` | reverse | ✅ native | ⚠️ derived |
| Enzyme.jl | `AutoEnzyme(mode=Enzyme.Reverse)` | reverse | ✅ native | — |
| Enzyme.jl | `AutoEnzyme(mode=Enzyme.Forward)` | forward | — | ✅ native |
| FiniteDifferences.jl | `AutoFiniteDifferences` | numerical | ✅ native | ✅ native |
| FiniteDiff.jl | `AutoFiniteDiff` | numerical | ✅ native | ✅ native |

When a backend only supports one direction natively, VG.jl falls back with a `@warn`. Pullback from a forward-mode backend runs one pushforward per input element; pushforward from a reverse-mode backend calls pullback once per output element.

Input support varies by backend:

| Backend | Scalar | Array | Complex | Multi-arg |
|---|---|---|---|---|
| `AutoMooncake` | ✅ | ✅ | ✅ (passthrough) | ✅ |
| `AutoMooncakeForward` | — | ✅ | ✅ (passthrough) | ✅ |
| `AutoForwardDiff` | ✅ | ✅ | — | ✅ |
| `AutoReverseDiff` | — | ✅ | — | ✅ |
| `AutoTracker` | — | ✅ | — | ✅ |
| `AutoZygote` | ✅ | ✅ | ✅ | ✅ |
| `AutoEnzyme` | — | ✅ | — | ✅ |
| `AutoFiniteDifferences` | ✅ | ✅ | ✅ | ✅ |
| `AutoFiniteDiff` | — | ✅ | — | ✅ |

## Structured array inputs

`AutoMooncake` supports structured arrays with `friendly_tangents=true`:

```julia
using LinearAlgebra

x = Symmetric([1.0 2.0; 2.0 3.0])
backend = AutoMooncake(config=Mooncake.Config(friendly_tangents=true))
y, x̄ = value_and_pullback!!(x -> sum(x .^ 2), 1.0, backend, x)
# x̄ is a plain Matrix{Float64}
```

| Type | `friendly_tangents=false` | `friendly_tangents=true` |
|---|---|---|
| `Symmetric` | Mooncake tangent type | `Matrix{T}` ✅ |
| `SymTridiagonal` | Mooncake tangent type | `Matrix{T}` ✅ |
| `Diagonal` | — | `@test_broken` (upstream Mooncake gap) |
| `Hermitian` | — | `@test_broken` (upstream Mooncake gap) |

## Testing utilities

Load `FiniteDifferences` and `Test` to enable test helpers. Each compares AD results against independent finite-difference computations:

```julia
using ValueAndGradient, FiniteDifferences, Test, ADTypes

backend = AutoMooncake(config=nothing)

# pullback: checks value == f(x) and x̄ agrees with FD
test_pullback(x -> sum(x .^ 2), 1.0, backend, [1.0, 2.0, 3.0])
test_pullback(x -> x .^ 2, [2.0, -1.0, 3.0], backend, [1.0, 2.0, 3.0])
test_pullback(x -> (a=sum(x.^2), b=x[1]), (a=1.0, b=1.0), backend, [1.0, 2.0])
test_pullback((x, y) -> sum(x .* y), 1.0, backend, [1.0, 2.0], [3.0, 4.0])

# pushforward: checks value == f(x) and ẏ agrees with FD
test_pushforward(x -> x .^ 2, [1.0, 0.0, 0.0], backend, [1.0, 2.0, 3.0])
test_pushforward(x -> (a=sum(x.^2), b=x[1]), ones(3), backend, [1.0, 2.0, 3.0])
```

Both helpers accept `rtol` and `atol` keyword arguments (default `1e-5`).

## Examples

The `examples/` directory contains five standalone scripts demonstrating `value_and_pullback!!` across the SciML ecosystem. Each runs multiple backends on the same loss function to show that swapping the backend is a one-argument change.

| Script | Problem | Validates against | Notes |
|---|---|---|---|
| `ode_param_estimation.jl` | ODE parameter estimation (`du/dt = -θu`) | finite differences + convergence | Mooncake/Zygote require SciMLSensitivity; FiniteDifferences works without it |
| `integrals_gradient.jl` | Numerical integral gradient (`∫₀¹ eˢˣ dx`) | analytical formula | |
| `linear_solve.jl` | Parameterised linear system (`A(θ)x = b`) | finite differences | |
| `lux_training.jl` | Lux MLP training loop | gradient check + loss decrease | |
| `neural_ode.jl` | Neural ODE (Lux inside ODE RHS) | finite differences + convergence | Mooncake/Zygote require SciMLSensitivity; FiniteDifferences works without it |

The two ODE scripts need SciMLSensitivity for Mooncake and Zygote to work — without it those backends fail at the solver boundary. Both scripts catch and report failures rather than crashing, so FiniteDifferences still runs.

Setup:

```julia
julia --project=examples/ -e '
    using Pkg
    Pkg.develop(path=".")
    Pkg.add([
        "OrdinaryDiffEq", "SciMLSensitivity",
        "Integrals", "LinearSolve",
        "Lux", "Optimisers",
        "Mooncake", "Zygote", "FiniteDifferences",
        "ADTypes", "Random",
    ])
'
```

Run any script with:

```
julia --project=examples/ examples/ode_param_estimation.jl
```

## Implementing a new backend

Define methods for the operations your backend supports natively. VG.jl will automatically handle the other direction via its derived fallback.

```julia
# In MyPackage/ext/ValueAndGradientMyBackendExt.jl
module ValueAndGradientMyBackendExt

using ValueAndGradient: ValueAndGradient, DiffInput, DiffLeaf
using ADTypes: AutoMyBackend

# Single-arg pullback
function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMyBackend, x::DiffInput;
        ad_cache=nothing, canonical_tangents=false, kwargs...) where {F}
    c = ad_cache !== nothing ? ad_cache : build_cache(f, x)
    y, x̄ = my_vjp(c, ȳ, f, x)
    return y, canonical_tangents ? ValueAndGradient._canonicalize(x, x̄) : x̄
end

# Multi-arg pullback
function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMyBackend,
        x1::DiffLeaf, x2::DiffLeaf, xrest::DiffLeaf...;
        ad_cache=nothing, canonical_tangents=false, kwargs...) where {F}
    xs = (x1, x2, xrest...)
    y, x̄s = my_vjp_multiarg(ȳ, f, xs...)
    return y, canonical_tangents ? ValueAndGradient._canonicalize(xs, x̄s) : x̄s
end

end
```

Register the extension in `Project.toml`:

```toml
[weakdeps]
MyBackend = "..."

[extensions]
ValueAndGradientMyBackendExt = "MyBackend"
```
