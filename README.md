# ValueAndGradient.jl

VJPs and JVPs across any Julia AD backend, with a single shared interface.

## Why

SciML, Turing, and Lux each ship their own thin wrappers around Mooncake, Enzyme, Zygote, etc. VG.jl is that wrapper, written once. Callers depend on VG.jl; backends implement it once. Swapping backends is a one-argument change.

The design is deliberately shallow: VG.jl calls each backend's public API directly and never reaches into internals, so it doesn't break on every backend release.

## Quick start

```julia
using ValueAndGradient, ADTypes, Mooncake

f = x -> sum(x .^ 2)
x = [1.0, 2.0, 3.0]

# VJP: returns (f(x), ∂f/∂x · ȳ)
y, x̄ = value_and_pullback!!(f, 1.0, AutoMooncake(config=nothing), x)
# y = 14.0, x̄ = [2.0, 4.0, 6.0]

# JVP: returns (f(x), ∂f/∂x · ẋ)
y, ẏ = value_and_pushforward!!(f, ones(3), AutoMooncakeForward(config=nothing), x)
# y = 14.0, ẏ = 12.0
```

Both functions have the same signature shape:

```julia
value_and_pullback!!(f, ȳ, backend, x...; ad_cache=nothing, canonical_tangents=false)
value_and_pushforward!!(f, ẋ, backend, x...; ad_cache=nothing, canonical_tangents=false)
```

`ȳ` must match the output type of `f`. For array outputs, pass an array of the same shape; for scalar outputs, pass a scalar. Multi-argument functions return a tuple of per-argument tangents:

```julia
g = (x, y) -> sum(x .* y)
val, (x̄, ȳ) = value_and_pullback!!(g, 1.0, AutoMooncake(config=nothing), [1.0, 2.0], [3.0, 4.0])
```

Tuple and NamedTuple outputs work too — pass a matching `ȳ`:

```julia
f = x -> (a = sum(x .^ 2), b = x[1] + x[2])
y, x̄ = value_and_pullback!!(f, (a=1.0, b=1.0), AutoMooncake(config=nothing), x)
```

## Input types

Inputs must be differentiable scalars or arrays:

```julia
DiffScalar = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}
DiffArray  = AbstractArray{<:DiffScalar}
DiffLeaf   = Union{DiffScalar, DiffArray}
```

There's no constraint on output types — structs, tuples, NamedTuples, arrays, scalars all pass through in principle. In practice, support depends on the backend; if something doesn't work, please open a PR.

## Caching

For hot paths (training loops, solvers), build the backend cache once outside the loop:

```julia
# Mooncake
cache = Mooncake.prepare_pullback_cache(f, x)
for _ in 1:10_000
    y, x̄ = value_and_pullback!!(f, 1.0, AutoMooncake(config=nothing), x; ad_cache=cache)
end

# ReverseDiff compiled tape
tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, x))
y, x̄ = value_and_pullback!!(f, 1.0, AutoReverseDiff(), x; ad_cache=tape)

# FiniteDiff
cache = FiniteDiff.GradientCache(similar(x), x)
y, x̄ = value_and_pullback!!(f, 1.0, AutoFiniteDiff(), x; ad_cache=cache)
```

Stateless backends (Zygote, FiniteDifferences, Tracker, ForwardDiff, Enzyme) ignore `ad_cache` with a warning.

## Canonical tangents

Different backends return different types for what is conceptually "zero gradient" or "struct gradient". Setting `canonical_tangents=true` normalises the known cases — it is not a guarantee of a fully shared common type across all backends, but for `DiffLeaf` inputs with scalar/array outputs all backends agree after normalisation:

| Raw tangent | Normalised to |
|---|---|
| `nothing` — Zygote's cotangent for an unused argument | `zero(x)` |
| `Mooncake.Tangent{NT}` — struct tangent from Mooncake pushforward | `NT` NamedTuple, then tries `T(values(NT)...)` |
| Any other type | unchanged |

```julia
# Zygote returns nothing for y since f doesn't use it
f = (x, y) -> sum(x .^ 2)
_, x̄s = value_and_pullback!!(f, 1.0, AutoZygote(), x, y; canonical_tangents=true)
# x̄s[2] == zero(y)  instead of nothing

# Mooncake pushforward: struct output comes back as a reconstructed struct
struct MyPair{T}; a::T; b::T; end
f = x -> MyPair(sum(x .^ 2), x[1] + x[2])
_, ẏ = value_and_pushforward!!(f, ones(3), AutoMooncakeForward(config=nothing), x; canonical_tangents=true)
# ẏ isa MyPair{Float64}
```

If `T` has no matching positional constructor, the raw `NamedTuple` is returned with a `@warn`.


## Backends

| Backend | ADTypes | Pullback | Pushforward |
|---|---|---|---|
| Mooncake.jl | `AutoMooncake` | ✅ native | ⚠️ derived |
| Mooncake.jl | `AutoMooncakeForward` | ⚠️ derived | ✅ native |
| ForwardDiff.jl | `AutoForwardDiff` | ⚠️ derived | ✅ native |
| ReverseDiff.jl | `AutoReverseDiff` | ✅ native | ⚠️ derived |
| Tracker.jl | `AutoTracker` | ✅ native | ⚠️ derived |
| Zygote.jl | `AutoZygote` | ✅ native | ⚠️ derived |
| Enzyme.jl | `AutoEnzyme` | ✅ native | ✅ native |
| FiniteDifferences.jl | `AutoFiniteDifferences` | ✅ native | ✅ native |
| FiniteDiff.jl | `AutoFiniteDiff` | ✅ native | ✅ native |

⚠️ derived: VG.jl implements the missing direction via a fallback — pullback from a forward-mode backend runs one pushforward per input element; pushforward from a reverse-mode backend runs one pullback per output element. A `@warn` is emitted.

**Enzyme:** `AutoEnzyme`'s `mode` field is ignored. `value_and_pullback!!` always uses `Enzyme.Reverse` internally; `value_and_pushforward!!` always uses `Enzyme.Forward`.

Input support varies:

| Backend | Scalar | Array | Complex | Multi-arg |
|---|---|---|---|---|
| `AutoMooncake` | ✅ | ✅ | ✅ | ✅ |
| `AutoMooncakeForward` | ✅ | ✅ | ✅ | ✅ |
| `AutoForwardDiff` | ✅ | ✅ | — | ✅ |
| `AutoReverseDiff` | — | ✅ | — | ✅ |
| `AutoTracker` | — | ✅ | — | ✅ |
| `AutoZygote` | ✅ | ✅ | ✅ | ✅ |
| `AutoEnzyme` | — | ✅ | — | ✅ |
| `AutoFiniteDifferences` | ✅ | ✅ | ✅ | ✅ |
| `AutoFiniteDiff` | — | ✅ | — | ✅ |

## Structured array inputs

`AutoMooncake` supports structured arrays via `friendly_tangents=true`, which converts Mooncake's internal tangent type back to a plain `Matrix`:

```julia
using LinearAlgebra
x = Symmetric([1.0 2.0; 2.0 3.0])
backend = AutoMooncake(config=Mooncake.Config(friendly_tangents=true))
y, x̄ = value_and_pullback!!(x -> sum(x .^ 2), 1.0, backend, x)
# x̄ isa Matrix{Float64}
```

| Type | `friendly_tangents=true` |
|---|---|
| `Symmetric` | `Matrix{T}` ✅ |
| `SymTridiagonal` | `Matrix{T}` ✅ |
| `Diagonal` | `@test_broken` — upstream Mooncake gap |
| `Hermitian` | `@test_broken` — upstream Mooncake gap |

## Testing utilities

Load `FiniteDifferences` and `Test` to get `test_pullback` and `test_pushforward`, which check AD results against finite differences:

```julia
using ValueAndGradient, FiniteDifferences, Test, ADTypes

backend = AutoMooncake(config=nothing)
test_pullback(x -> sum(x .^ 2), 1.0, backend, [1.0, 2.0, 3.0])
test_pushforward(x -> x .^ 2, [1.0, 0.0, 0.0], backend, [1.0, 2.0, 3.0])
```

Both accept `rtol` and `atol` (default `1e-5`).

## Examples

`examples/` has five standalone scripts showing `value_and_pullback!!` across the SciML ecosystem. Each runs multiple backends on the same problem so that swapping is visibly a one-argument change.

| Script | Problem |
|---|---|
| `linear_solve.jl` | Parameterised linear system `A(θ)x = b` |
| `integrals_gradient.jl` | Numerical integral gradient `∫₀¹ eˢˣ dx` |
| `lux_training.jl` | Lux MLP training loop |
| `ode_param_estimation.jl` | ODE parameter estimation `du/dt = -θu` |
| `neural_ode.jl` | Neural ODE (Lux inside ODE RHS) |

The two ODE scripts need SciMLSensitivity for Mooncake and Zygote to differentiate through the solver. Without it those backends fail at the solver boundary; FiniteDifferences still runs. Both scripts catch and report failures rather than crashing.

```julia
julia --project=examples/ -e '
    using Pkg; Pkg.develop(path=".")
    Pkg.add(["OrdinaryDiffEq", "SciMLSensitivity", "Integrals", "LinearSolve",
             "Lux", "Optimisers", "Mooncake", "Zygote", "FiniteDifferences",
             "ADTypes", "Random"])
'
julia --project=examples/ examples/linear_solve.jl
```

## Adding a new backend

Implement whichever of the two operations your backend supports natively — VG.jl will derive the other automatically. Register the extension in `Project.toml` and call `_canonicalize` at each return site if you want `canonical_tangents` support:

```julia
# ext/ValueAndGradientMyBackendExt.jl
module ValueAndGradientMyBackendExt

using ValueAndGradient: ValueAndGradient, DiffInput, DiffLeaf
using ADTypes: AutoMyBackend

function ValueAndGradient.value_and_pullback!!(
        f::F, ȳ, backend::AutoMyBackend, x::DiffInput;
        ad_cache=nothing, canonical_tangents=false, kwargs...) where {F}
    y, x̄ = my_vjp(f, ȳ, x)
    return y, canonical_tangents ? ValueAndGradient._canonicalize(x, x̄, backend) : x̄
end

end
```

```toml
# Project.toml
[weakdeps]
MyBackend = "..."

[extensions]
ValueAndGradientMyBackendExt = "MyBackend"
```
