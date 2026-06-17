module ValueAndGradient

import ADTypes
using ADTypes: AbstractADType, ForwardMode

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
    value_and_gradient!!(f, backend, x...; ad_cache=nothing, canonical_tangents=false) -> (y, x̄)

Returns `y = f(x...)` (which must be a scalar) and the gradient `x̄`.
Single argument: `x̄` has the same structure as `x`.
Multiple arguments: `x̄` is a tuple of per-argument gradients.

See `value_and_pullback!!` for `ad_cache` and `canonical_tangents` semantics.
"""
function value_and_gradient!! end

"""
    value_and_jacobian!!(f, backend, x; ad_cache=nothing, canonical_tangents=false) -> (y, (J,))
    value_and_jacobian!!(f, backend, x::Tuple{Vararg{DiffLeaf}}; ...) -> (y, (J,))
    value_and_jacobian!!(f, backend, x1, x2, ...; ad_cache=nothing, canonical_tangents=false) -> (y, (J1, J2, ...))

Returns `y = f(x)` and a tuple of Jacobian matrices, one per input argument.
`J[i,j] = ∂yᵢ/∂xⱼ` where `j` indexes into `vec(x)` for non-vector inputs.

**Input:** accepts all `DiffInput` types for single-arg calls:
- `DiffArray` — any `AbstractArray` with real or complex float elements
- `Tuple{Vararg{DiffLeaf}}` — `f` takes a single Tuple argument; returns `(y, (J,))`
  where `J` concatenates partial Jacobians for each Tuple element left-to-right.
  Column layout: the first `_n_params(t[1])` columns correspond to `t[1]`, the next
  `_n_params(t[2])` columns to `t[2]`, etc. To recover per-element Jacobians:
      ns = ValueAndGradient._n_params.(x)
      offsets = [0; cumsum(collect(ns))]
      Jᵢ = J[:, offsets[i]+1:offsets[i+1]]   # Jacobian w.r.t. t[i]
  Alternatively, pass Tuple elements as separate args to get `(J₁, J₂, …)` directly.
- `DiffScalar` — returns an `m×1` Jacobian; scalar-to-scalar gives a `1×1` matrix.
Multi-arg calls accept two or more `DiffInput` arguments and return one `Jᵢ` per arg.

**Native path (Layer 1):** single `AbstractVector{<:IEEEFloat}` with `AutoMooncake` or
`AutoMooncakeForward` — delegates directly to `Mooncake.value_and_jacobian!!`.

**Derived path (Layer 2):** everything else (matrix/complex inputs, Tuple inputs,
multi-arg calls, other backends) — builds the Jacobian column-by-column via repeated
pushforwards (forward-mode) or row-by-row via repeated pullbacks (reverse-mode).
Emits a `@warn` once per call reporting the number of AD passes and the n/m dimensions.
If a better mode exists (e.g. n>m for reverse, n<m for forward), the warning names it.

See `value_and_pullback!!` for `ad_cache` and `canonical_tangents` semantics.
"""
function value_and_jacobian!! end

"""
    value_and_derivative!!(f, backend, x; ad_cache=nothing, canonical_tangents=false) -> (y, ẏ)

Returns `y = f(x)` and the derivative `ẏ = df/dx`.
`x` must be a scalar (`DiffScalar` — real or complex). Output `y` is unconstrained.
For complex `x`, returns the Fréchet derivative in direction `one(x)` — equivalent to
`value_and_pushforward!!` with `ẋ = one(x)`. Only `AutoMooncakeForward` is supported.
For reverse-mode backends, use `value_and_pullback!!` with `ȳ = one(T)` explicitly.

See `value_and_pullback!!` for `ad_cache` and `canonical_tangents` semantics.
"""
function value_and_derivative!! end

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

"""
    test_gradient(f, backend, xs...; rtol=1e-5, atol=1e-5)

Check `value_and_gradient!!` against finite differences.
Requires `FiniteDifferences` to be loaded.
"""
function test_gradient end

"""
    test_derivative(f, backend, x; rtol=1e-5, atol=1e-5)

Check `value_and_derivative!!` against finite differences.
Requires `FiniteDifferences` to be loaded.
"""
function test_derivative end

"""
    test_jacobian(f, backend, x; rtol=1e-5, atol=1e-5)

Check `value_and_jacobian!!` against finite differences.
Requires `FiniteDifferences` to be loaded.
"""
function test_jacobian end

function value_and_derivative!!(f, backend::AbstractADType, x; kwargs...)
    throw(
        ArgumentError(
            "value_and_derivative!! is a forward-mode operation and is not supported for $(typeof(backend)). " *
                "Use AutoMooncakeForward, or use value_and_pullback!! with ȳ = one(T) for reverse-mode backends.",
        ),
    )
end

function value_and_pullback!!(f, ȳ, backend::AbstractADType, xs...; kwargs...)
    throw(
        ArgumentError(
            "value_and_pullback!! is a reverse-mode operation and is not supported for $(typeof(backend)). " *
                "Use AutoMooncake, or use value_and_pushforward!! for forward-mode backends.",
        ),
    )
end

function value_and_pushforward!!(f, ẋ, backend::AbstractADType, xs...; kwargs...)
    throw(
        ArgumentError(
            "value_and_pushforward!! is a forward-mode operation and is not supported for $(typeof(backend)). " *
                "Use AutoMooncakeForward, or use value_and_pullback!! for reverse-mode backends.",
        ),
    )
end

function value_and_gradient!!(f, backend::AbstractADType, xs...; kwargs...)
    throw(
        ArgumentError(
            "value_and_gradient!! is a reverse-mode operation and is not supported for $(typeof(backend)). " *
                "Use AutoMooncake.",
        ),
    )
end

_prepare_jac_cache_forward(f, backend, x) = nothing
_prepare_jac_cache_reverse(f, backend, x) = nothing

_zero_tangent(x::DiffScalar) = zero(x)
_zero_tangent(x::DiffArray) = zeros(eltype(x), size(x))
_zero_tangent(x::Tuple) = map(_zero_tangent, x)

_flatten(x::DiffScalar) = [x]
_flatten(x::Array) = vec(x)
_flatten(x::AbstractArray) = vec(collect(x))
_flatten(x::Tuple) = reduce(vcat, map(_flatten, x))

_n_params(x::DiffScalar) = 1
_n_params(x::DiffArray) = length(x)
_n_params(x::Tuple) = sum(_n_params, x)

_basis_tangents(x::DiffScalar) = (one(x),)
_basis_tangents(x::DiffArray) =
    (begin t = zeros(eltype(x), size(x)); t[k] = one(eltype(x)); t end
     for k in CartesianIndices(x))
_basis_tangents(x::Tuple) = (
    ntuple(k -> k == i ? ẋ : _zero_tangent(x[k]), length(x))
    for (i, xi) in enumerate(x) for ẋ in _basis_tangents(xi)
)

function _first_row_jacobian(row, m, n)
    J = Matrix{eltype(row)}(undef, m, n)
    J[1, :] = row
    J
end

function _cols_to_matrix(cols)
    J = Matrix{eltype(cols[1])}(undef, length(cols[1]), length(cols))
    for (k, col) in enumerate(cols); J[:, k] = col; end
    J
end

function _jac_warn(backend, n::Int, m::Int)
    if ADTypes.mode(backend) isa ForwardMode
        extra = m < n ? " Output has fewer dims — a reverse-mode backend would need only $m pullback calls." : ""
        @warn "value_and_jacobian!! derived path: $(typeof(backend)), $n pushforward calls (n=$n inputs, m=$m outputs).$extra"
    else
        extra = n < m ? " Input has fewer dims — a forward-mode backend would need only $n pushforward calls." : ""
        @warn "value_and_jacobian!! derived path: $(typeof(backend)), $m pullback calls (n=$n inputs, m=$m outputs).$extra"
    end
end

function value_and_jacobian!!(
        f::F,
        backend::AbstractADType,
        x::DiffInput;
        ad_cache = nothing,
        canonical_tangents = false,
    ) where {F}
    if ADTypes.mode(backend) isa ForwardMode
        return _jacobian_via_pushforward(f, backend, x; ad_cache, canonical_tangents)
    else
        return _jacobian_via_pullback(f, backend, x; ad_cache, canonical_tangents)
    end
end

function value_and_jacobian!!(
        f::F,
        backend::AbstractADType,
        x1::DiffInput,
        x2::DiffInput,
        xrest::DiffInput...;
        ad_cache = nothing,
        canonical_tangents = false,
    ) where {F}
    if ADTypes.mode(backend) isa ForwardMode
        return _jacobian_via_pushforward(f, backend, x1, x2, xrest...; ad_cache, canonical_tangents)
    else
        return _jacobian_via_pullback(f, backend, x1, x2, xrest...; ad_cache, canonical_tangents)
    end
end

function _jacobian_via_pushforward(
        f::F, backend, x::DiffInput;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache : _prepare_jac_cache_forward(f, backend, (x,))
    n = _n_params(x)
    basis = _basis_tangents(x)
    it = iterate(basis)
    it === nothing && return f(x), (zeros(0, n),)
    ẋ₁, state = it
    y, ẏ₁ = value_and_pushforward!!(f, ẋ₁, backend, x; ad_cache = c, canonical_tangents)
    flat₁ = _flatten(ẏ₁)
    m = length(flat₁)
    _jac_warn(backend, n, m)
    J = Matrix{eltype(flat₁)}(undef, m, n)
    J[:, 1] = flat₁
    col = 2
    it = iterate(basis, state)
    while it !== nothing
        ẋ, state = it
        _, ẏ = value_and_pushforward!!(f, ẋ, backend, x; ad_cache = c, canonical_tangents)
        J[:, col] = _flatten(ẏ)
        col += 1
        it = iterate(basis, state)
    end
    return y, (J,)
end

function _jacobian_via_pushforward(
        f::F, backend, x1::DiffInput, x2::DiffInput, xrest::DiffInput...;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    xs = (x1, x2, xrest...)
    c = ad_cache !== nothing ? ad_cache : _prepare_jac_cache_forward(f, backend, xs)
    ns = _n_params.(xs)
    n = sum(ns)
    y_out = Ref{Any}(nothing)
    cols_per_arg = [Vector{Vector}() for _ in xs]
    for (i, xi) in enumerate(xs)
        for ẋi in _basis_tangents(xi)
            ẋs = ntuple(j -> j == i ? ẋi : _zero_tangent(xs[j]), length(xs))
            y, ẏ = value_and_pushforward!!(f, ẋs, backend, xs...; ad_cache = c, canonical_tangents)
            flat = _flatten(ẏ)
            if isnothing(y_out[])
                y_out[] = y
                _jac_warn(backend, n, length(flat))
            end
            push!(cols_per_arg[i], flat)
        end
    end
    isnothing(y_out[]) && return f(xs...), Tuple(zeros(0, ns[i]) for i in eachindex(xs))
    Js = Tuple(_cols_to_matrix(cols_per_arg[i]) for i in eachindex(xs))
    return y_out[], Js
end

function _jacobian_via_pullback(
        f::F, backend, x::DiffInput;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    c = ad_cache !== nothing ? ad_cache : _prepare_jac_cache_reverse(f, backend, (x,))
    n = _n_params(x)
    y = f(x)
    m = y isa DiffScalar ? 1 : length(y)
    _jac_warn(backend, n, m)
    if y isa DiffScalar
        _, x̄ = value_and_pullback!!(f, one(y), backend, x; ad_cache = c, canonical_tangents)
        return y, (_first_row_jacobian(_flatten(x̄), 1, n),)
    end
    ȳ = zeros(eltype(y), size(y))
    ȳ[1] = one(eltype(y))
    _, x̄₁ = value_and_pullback!!(f, ȳ, backend, x; ad_cache = c, canonical_tangents)
    ȳ[1] = zero(eltype(y))
    J = _first_row_jacobian(_flatten(x̄₁), m, n)
    for j in 2:m
        ȳ[j] = one(eltype(y))
        _, x̄ = value_and_pullback!!(f, ȳ, backend, x; ad_cache = c, canonical_tangents)
        J[j, :] = _flatten(x̄)
        ȳ[j] = zero(eltype(y))
    end
    return y, (J,)
end

function _jacobian_via_pullback(
        f::F, backend, x1::DiffInput, x2::DiffInput, xrest::DiffInput...;
        ad_cache = nothing, canonical_tangents = false,
    ) where {F}
    xs = (x1, x2, xrest...)
    c = ad_cache !== nothing ? ad_cache : _prepare_jac_cache_reverse(f, backend, xs)
    ns = _n_params.(xs)
    n = sum(ns)
    y = f(xs...)
    m = y isa DiffScalar ? 1 : length(y)
    _jac_warn(backend, n, m)
    if y isa DiffScalar
        _, x̄s = value_and_pullback!!(f, one(y), backend, xs...; ad_cache = c, canonical_tangents)
        rows = map(_flatten, x̄s)
        return y, Tuple(_first_row_jacobian(rows[i], 1, ns[i]) for i in eachindex(xs))
    end
    ȳ = zeros(eltype(y), size(y))
    ȳ[1] = one(eltype(y))
    _, x̄s₁ = value_and_pullback!!(f, ȳ, backend, xs...; ad_cache = c, canonical_tangents)
    ȳ[1] = zero(eltype(y))
    rows₁ = map(_flatten, x̄s₁)
    Js = Tuple(_first_row_jacobian(rows₁[i], m, ns[i]) for i in eachindex(xs))
    for j in 2:m
        ȳ[j] = one(eltype(y))
        _, x̄s = value_and_pullback!!(f, ȳ, backend, xs...; ad_cache = c, canonical_tangents)
        ȳ[j] = zero(eltype(y))
        for (i, x̄i) in enumerate(x̄s); Js[i][j, :] = _flatten(x̄i); end
    end
    return y, Js
end

function value_and_jacobian!!(f, backend::AbstractADType, x; kwargs...)
    throw(
        ArgumentError(
            "value_and_jacobian!! requires DiffInput (DiffScalar, DiffArray, or Tuple{Vararg{DiffLeaf}}) inputs, got $(typeof(x)).",
        ),
    )
end


export value_and_pullback!!,
    value_and_pushforward!!,
    value_and_gradient!!,
    value_and_jacobian!!,
    value_and_derivative!!,
    test_pullback,
    test_pushforward,
    test_gradient,
    test_jacobian,
    test_derivative

end


# Slide 1 — Title
  
#   Title: ValueAndGradient.jl: A Unified AD API for the Mooncake Ecosystem

#   Subtitle: [Your name] · [Group name] · [Date]
  
#   Script:

#   ▎ "I'll talk about a Julia package I've been building as part of my thesis — ValueAndGradient.jl. It's a thin API layer that sits on top of Mooncake and provides a unified interface 
#   ▎ for the five core AD operations. The motivating question was: what should you actually call when you want a Jacobian in Julia today, and why does the existing answer keep 
#   ▎ breaking?"

#   ---
#   Slide 2 — The Problem

#   Heading: DifferentiationInterface.jl: the right goal, the wrong architecture

#   Left column — What DI set out to do:
#   One backend-agnostic AD API over ForwardDiff, Zygote, Enzyme, Mooncake, and others. The right goal.

#   Middle column — Why it broke: 
#   1. Extensions live in DI's repo — maintained by the wrong team
#   2. Calls Mooncake internal functions (prepare_pullback_cache, _copy_output, MinimalCtx)
#   3. Imports Mooncake's internal type system (CoDual, etc.)
#   4. Version-detection hacks (isdefined(Mooncake, :FriendlyTangentCache))
#   5. One Mooncake version cap cascades: Mooncake ≤ 0.5.24 → ComponentArrays blocked → all of SciML blocked
#   6. Enzyme activity states can't be represented in DI's abstraction at all

#   Right column — The ecosystem response:
#   - Turing left DI entirely (PR #1354) — moved to native backend APIs directly
#   - Lux rejected DI — AbstractArray constraint unworkable for ML
  
#   Bottom — prominent:
  
#   ▎ VG.jl is what DI should have been: a general AD interface for Julia, done correctly.
#   ▎ Extensions in VG.jl's repo. Only public backend APIs. Same principle as LogDensityProblemsAD.jl — the architecture Turing already trusted.

#   Script:   

#   ▎ "DI had the right idea — one interface over all AD backends in Julia. The problem is architectural. DI's extensions live in DI's repo, not the backend's repo, so every time 
#   ▎ Mooncake makes an internal change, DI breaks. It calls backend internals — private type system, functions that were never part of the public contract. When Mooncake releases a new 
#   ▎ version, DI can't keep up, and because DI pins an old Mooncake version, everything downstream — ComponentArrays, SciML — gets blocked too. Turing saw this and simply left DI. They 
#   ▎ now call Mooncake's public API directly. VG.jl formalises that as a reusable package. Extensions live in VG.jl's repo. We only ever call public APIs. If Mooncake changes internals 
#   ▎ tomorrow, VG.jl is unaffected."

#   ---
#   Slide 3 — The API: Five Operations

#   Heading: Five operations, one consistent interface

#   ┌─────────────────────────┬──────────────────┬─────────┬────────────────────────┐
#   │        Operation        │     Returns      │  Mode   │  Constraint on input   │
#   ├─────────────────────────┼──────────────────┼─────────┼────────────────────────┤
#   │ value_and_pullback!!    │ (y, x̄) — VJP     │ Reverse │ ȳ matches output shape │
#   ├─────────────────────────┼──────────────────┼─────────┼────────────────────────┤
#   │ value_and_pushforward!! │ (y, ẏ) — JVP     │ Forward │ ẋ matches input shape  │
#   ├─────────────────────────┼──────────────────┼─────────┼────────────────────────┤
#   │ value_and_gradient!!    │ (y, ∇f)          │ Reverse │ y must be scalar       │
#   ├─────────────────────────┼──────────────────┼─────────┼────────────────────────┤
#   │ value_and_derivative!!  │ (y, df/dx)       │ Forward │ x must be DiffScalar   │
#   ├─────────────────────────┼──────────────────┼─────────┼────────────────────────┤
#   │ value_and_jacobian!!    │ (y, (J₁, J₂, …)) │ Either  │ x must be DiffInput    │
#   └─────────────────────────┴──────────────────┴─────────┴────────────────────────┘
  
#   Below table:
#   - ad_cache and canonical_tangents kwargs available on all five
#   - Output of f is unconstrained on all five — scalar, array, tuple, struct, anything
#   - First-order only — HVP/Hessian excluded (require backend internals)

#   Script:

#   ▎ "The package exposes five functions. The first four are thin wrappers — they add type checking and a consistent ad_cache and canonical_tangents interface, then delegate directly to
#   ▎ Mooncake's public API. No logic, no transformation, zero overhead over calling Mooncake yourself. The interesting one is value_and_jacobian!!, which returns a tuple of Jacobian 
#   ▎ matrices — one per input argument — and has to do real work to support the full range of input types. That's what the next two slides cover. Note that we deliberately exclude 
#   ▎ Hessians and HVPs — those require close coupling with backend internals, which would put us right back in DI's position."

#   ---
#   Slide 4 — What is a DiffInput?

#   Heading: Supported input types: DiffInput

#   Type hierarchy:
#   DiffInput
#   ├── DiffLeaf
#   │   ├── DiffScalar  ─── Float32, Float64, ComplexF64, …
#   │   └── DiffArray   ─── AbstractArray{<:DiffScalar}
#   │                       (any shape, any real/complex float eltype)
#   └── Tuple{Vararg{DiffLeaf}}   ─── multi-argument functions

#   Examples:
#   # All valid inputs to value_and_jacobian!!:
#   2.0                          # DiffScalar
#   randn(Float32, 4)            # Vector{Float32}
#   randn(ComplexF64, 3, 3)      # Matrix{ComplexF64}
#   (randn(4), randn(2, 2))      # Tuple{DiffLeaf, DiffLeaf}
  
#   Note at bottom: Output of f is completely unconstrained — scalar, vector, matrix, NamedTuple, custom struct. VG.jl flattens it internally to build Jacobian rows.

#   Script:

#   ▎ "DiffInput is the constraint on what value_and_jacobian!! accepts as input. A DiffScalar is any real or complex IEEE float — Float32, Float64, ComplexF64. A DiffArray is any 
#   ▎ AbstractArray over those — vectors, matrices, arbitrary shapes. And you can pass a Tuple of DiffLeafs for multi-argument functions, in which case you get back one Jacobian per 
#   ▎ argument. Crucially, the output of f is completely unconstrained. VG.jl just calls _flatten on whatever comes back and uses that to build the rows or columns of the Jacobian. This 
#   ▎ is a deliberate design choice — we don't want to constrain what functions you can differentiate."

#   ---
#   Slide 5 — Architecture: Layer 1 vs Layer 2

#   Heading: Two-layer dispatch in value_and_jacobian!!

#   Flowchart:
#   value_and_jacobian!!(f, backend, x)
#              │
#       Is x AbstractVector{<:IEEEFloat}
#       AND backend ∈ {AutoMooncake, AutoMooncakeForward}?
#              │
#        ┌─────┴─────┐
#       YES          NO
#        │            │
#     Layer 1      Layer 2
#     (native)     (derived)
#        │            │
#   Mooncake.      ADTypes.mode(backend)?
#   value_and_          │
#   jacobian!!    ┌─────┴─────┐
#        │      Forward     Reverse
#     O(1) AD      │            │
#     passes   n pushforwards  m pullbacks
#              (1 per input   (1 per output
#               dimension)     dimension)

#   Below flowchart — two key points:
  
#   ad_cache threading: pass a pre-built cache to skip rule compilation on every call — critical in hot loops (ODE solvers, optimisers). Benchmark: forward cached 5× faster than
#   uncached. 

#   @warn on Layer 2: tells you the number of AD passes and flags if you've chosen the suboptimal mode for the shape you have.
  
#   Script:

#   ▎ "Layer 1 is the native Mooncake path — plain real vector input, we call Mooncake.value_and_jacobian!! directly. One compiled rule, one pass, nothing clever. Layer 2 is the derived 
#   ▎ path — everything else: matrix inputs, complex inputs, scalar inputs, multi-arg calls. We construct the Jacobian by looping over basis tangents or cotangents. Forward mode builds 
#   ▎ it column by column — one pushforward per input dimension, so cost is O(n). Reverse mode builds it row by row — one pullback per output dimension, so cost is O(m). Which is cheaper
#   ▎ depends entirely on the shape of your Jacobian. VG.jl emits a warning on Layer 2 that tells you exactly how many passes it made and whether a better mode exists for your problem. 
#   ▎ And if you're in a hot loop, pass ad_cache to avoid rebuilding Mooncake's rule on every call — that alone gives a 5× speedup in our benchmarks."

#   ---
#   Slide 6 — Remaining Work: SciML CPU

#   Heading: SciML CPU — 4 items remaining

#   ┌─────┬──────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
#   │  #  │                       Item                       │                                         Notes                                         │
#   ├─────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
#   │ 1   │ ComponentArray(::NamedTuple) rrule!!             │ No reverse-mode rule exists; only a ChainRules frule. Unblocks #4.                    │
#   ├─────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
#   │ 2   │ get_sampled_data rrule!!                         │ Returns gradient 0.0 today — floor division makes t invisible to reverse-mode tracing │
#   ├─────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
#   │ 3   │ RecursiveArrayTools VectorOfArray                │ Code written locally — evaluate whether real workflows hit it, then PR                │
#   ├─────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
#   │ 4   │ SciMLSensitivity #1426 — ComponentArray SubArray │ wsmoses says fixed enzyme-side (June 11) — needs re-test                              │
#   └─────┴──────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘
  
#   ┌─────┬───────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────┐
#   │  #  │                         Item                          │                                Notes                                 │            Function of interest            │
#   ├─────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
#   │ 1   │ ComponentArray(::NamedTuple) rrule!!                  │ No reverse-mode rule; only a ChainRules frule. Unblocks #4.          │ ComponentArrays.jl — componentarray.jl:96  │
#   ├─────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
#   │ 2   │ get_sampled_data rrule!!                              │ Returns 0.0 gradient — floor division makes t invisible to           │ MTKStdLib — sources.jl:603                 │
#   │     │                                                       │ reverse-mode                                                         │                                            │
#   ├─────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
#   │ 3   │ RecursiveArrayTools VectorOfArray                     │ Code written locally — evaluate whether real workflows hit it, then  │ RecursiveArrayTools.jl —                   │
#   │     │                                                       │ PR                                                                   │                          │
#   ├─────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
#   │ 4   │ SciMLSensitivity Issue #1425 — SVector immutability   │ MooncakeVJP in-place mutation incompatible with immutable SVector    │ SciMLSensitivity — issue #1425             │
#   ├─────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
#   │ 5   │ SciMLSensitivity Issue #1426 — ComponentArray         │ wsmoses says fixed enzyme-side (June 11) — needs re-test             │ ComponentArrays.jl —                       │
#   │  SubArray                                              │                                                                            │ array_interface.jl:163                     │
#   └─────┴───────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────┘


#   Script:

#   ▎ "Four SciML CPU items remain. The ComponentArray NamedTuple rule is the most load-bearing — there's no reverse-mode rule for constructing a ComponentArray from a NamedTuple, and it
#   ▎ also unblocks item 4. The get_sampled_data issue is a classic AD failure mode: a floor division inside the function makes the input invisible to tracing, so you get zero gradient 
#   ▎ silently. Both of those need writing from scratch. Item 3 is essentially done locally and just needs evaluation before a PR. Item 4 depends on an upstream Enzyme fix that may 
#   ▎ already be in — needs a re-test."

#   ---
#   Slide 7 — Remaining Work: GPU Rules

#   Heading: GPU rules — 5 root causes, covers all Lux + Flux GPU layers

#   ┌─────┬────────────────────────────────────────┬────────────────────────────────────────────────────────┐
#   │  #  │              Rule needed               │                        Unblocks                        │
#   ├─────┼────────────────────────────────────────┼────────────────────────────────────────────────────────┤
#   │ 5   │ Statistics.varm on CuArray             │ GroupNorm, InstanceNorm, LayerNorm (Lux + Flux)        │
#   ├─────┼────────────────────────────────────────┼────────────────────────────────────────────────────────┤
#   │ 6   │ Base.permutedims on CuArray            │ MultiHeadAttention (Lux + Flux)                        │
#   ├─────┼────────────────────────────────────────┼────────────────────────────────────────────────────────┤
#   │ 7   │ vcat/hcat/cat on CuArray               │ SkipConnection (Lux + Flux)                            │
#   ├─────┼────────────────────────────────────────┼────────────────────────────────────────────────────────┤
#   │ 8   │ LuxLib.Impl.batchnorm_cudnn!           │ BatchNorm GPU + BatchNorm CPU correctness (Lux + Flux) │
#   ├─────┼────────────────────────────────────────┼────────────────────────────────────────────────────────┤
#   │ 9   │ StatefulRecurrentCell setfield! on GPU │ RNN/LSTM/GRU GPU (Lux only — Flux already works)       │
#   └─────┴────────────────────────────────────────┴────────────────────────────────────────────────────────┘

#   ┌─────┬────────────────────────────────────┬─────────────────────────────────────────────────┬─────────────────────────────────────────┐
#   │  #  │            Rule needed             │                    Unblocks                     │          Function of interest           │
#   ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────┤
#   │ 5   │ Statistics.varm on CuArray         │ GroupNorm, InstanceNorm, LayerNorm (Lux + Flux) │ GPUArrays.jl — statistics.jl:3          │
#   ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────┤
#   │ 6   │ Base.permutedims on CuArray        │ MultiHeadAttention (Lux + Flux)                 │ GPUArrays.jl — linalg.jl:765            │
#   ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────┤
#   │ 7   │ vcat/hcat/cat on CuArray           │ SkipConnection (Lux + Flux)                     │ GPUArrays.jl — base.jl:175              │
#   ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────┤
#   │ 8   │ LuxLib.Impl.batchnorm_cudnn!       │ BatchNorm GPU (Lux + Flux)                      │ LuxLib.jl — LuxLibcuDNNExt/batchnorm.jl │
#   ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────┤
#   │ 9   │ StatefulRecurrentCell carry on GPU │ RNN/LSTM/GRU GPU (Lux only)                     │ Lux.jl — recurrent.jl:243               │
#   └─────┴────────────────────────────────────┴─────────────────────────────────────────────────┴─────────────────────────────────────────┘


#   Note at bottom: Flux GPU tests are permanently interface_only — Flux stores weights as CPU types regardless of device. Not a Mooncake problem; architectural ceiling in Flux.
  
#   Script:

#   ▎ "Five GPU rules cover the full remaining surface area for both Lux and Flux. Each one is a targeted rrule!! in MooncakeCUDAExt.jl. The varm rule is the highest coverage — one rule 
#   ▎ unblocks three layer types across two frameworks. The batchnorm_cudnn! rule is interesting because it also upgrades Lux BatchNorm on CPU from interface-only to full correctness — a
#   ▎ two-for-one. Worth noting: Flux GPU tests will always be interface-only regardless of what we fix, because Flux stores its weights as CPU types even when the model is on GPU. 
#   ▎ That's an architectural issue in Flux, not something Mooncake can solve."

#   ---
#   Slide 8 — Remaining Work: VG.jl

#   Heading: VG.jl — 6 items to complete the package

#   ┌─────┬────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
#   │  #  │                  Item                  │                                        Notes                                        │
#   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
#   │ 10  │ _canonicalize post-norm layer          │ Convert backend internal tangent types (e.g. Mooncake.Tangent) to plain Julia types │
#   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
#   │ 11  │ ForwardDiff backend                    │ True primitives: gradient, jacobian, derivative — rest derived automatically        │
#   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
#   │ 12  │ Zygote backend                         │ True primitives: pullback, gradient — rest derived                                  │
#   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
#   │ 13  │ FiniteDifferences pullback/pushforward │ j′vp/jvp exist in FiniteDifferences, just not wired in                              │
#   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
#   │ 14  │ Enzyme backend                         │ Complex activity states — hardest; lowest priority                                  │
#   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
#   │ 15  │ DynamicPPL VG.jl integration           │ Formalises PR #1354's ad-hoc backend calls into a proper protocol                   │
#   └─────┴────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘
  
# ┌─────┬────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
#   │  #  │                  Item                  │                                    Notes                                                             │
#   ├─────┼────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 10  │ _canonicalize post-norm layer          │ Convert backend tangent types (e.g. Mooncake.Tangent) to plain Julia types                           │
#   ├─────┼────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 11  │ ForwardDiff backend                    │ gradient(f, x::AbstractArray), jacobian(f, x::AbstractArray), derivative(f, x::Real)                 │
#   │     │                                        │ No pullback or pushforward API — both throw. Nothing else auto-derived.                              │
#   ├─────┼────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 12  │ Zygote backend                         │ pullback(f, args...), gradient(f, args...), jacobian(f, args...), pushforward(f, x...) → callable    │
#   │     │                                        │ pushforward needs normalisation (returns (ẋ...)->ẏ, not (y,ẏ))                                       │
#   │     │                                        │ jacobian: AbstractArray/Number x only; other types silently skipped                                  │
#   │     │                                        │ derivative throws (forward-mode only). jacobian also derivable via Layer 2 as fallback.              │
#   ├─────┼────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 13  │ FiniteDifferences backend              │ Current ext is test helpers only — no backend implemented yet                                        │
#   │     │                                        │ To wire in: grad→gradient, jacobian→jacobian, jvp→pushforward, j′vp→pullback (not yet used)         │
#   │     │                                        │ All accept arbitrary types via to_vec; fdm from backend.fdm                                          │
#   ├─────┼────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 14  │ Enzyme backend                         │ autodiff(Reverse,...) → pullback; autodiff(Forward, Duplicated(x,ẋ),...) → pushforward               │
#   │     │                                        │ gradient(Reverse,...) → gradient only (ȳ=1 sugar); gradient(Forward,...) → full gradient, not JVP   │
#   │     │                                        │ jacobian(Forward/Reverse,...) → jacobian                                                             │
#   │     │                                        │ Every arg needs activity annotation: Const, Active, Duplicated, BatchDuplicated, …                   │
#   │     │                                        │ Hardest to wrap — activity inference non-trivial; lowest priority                                     │
#   ├─────┼────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 15  │ DynamicPPL VG.jl integration           │ Formalises PR #1354's ad-hoc backend calls into a proper VG.jl protocol                              │
#   └─────┴────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘
#   Script:

#   ▎ "Six items remain for VG.jl. The _canonicalize layer is quality-of-life — right now custom struct tangents come back as Mooncake.Tangent and users have to access them via 
#   ▎ .fields.a. Canonicalisation converts those to plain Julia types automatically. The backend extensions are the bulk of the work — but because the derived-path mechanism is already 
#   ▎ in place, each backend only needs to implement its two true primitives; VG.jl derives the full five-operation suite automatically. FiniteDifferences is almost free — the functions 
#   ▎ exist, they just haven't been wired in. Enzyme is the hardest because of its activity state model. The DynamicPPL integration is the thesis payoff item — it formalises the 
#   ▎ architecture that Turing already trusts into a reusable protocol."



















#   Slide 1 — Title
  
#   Title: ValueAndGradient.jl: A Unified AD API for the Mooncake Ecosystem

#   Subtitle: [Your name] · [Group name] · [Date]
  
#   Script:

#   ▎ "I'll talk about a Julia package I've been building as part of my thesis — ValueAndGradient.jl. It's a thin API layer that sits on top of Mooncake and provides a unified interface 
#   ▎ for the five core AD operations. The motivating question was: what should you actually call when you want a Jacobian in Julia today, and why does the existing answer keep 
#   ▎ breaking?"

#   ---
#   Slide 2 — Why DifferentiationInterface.jl isn't enough

#   Left column — General DI design gaps (all backends):
#   - Mandatory AbstractVector input incompatible with fmap-based parameter structures (Lux issue #544)
#   - Enzyme activity states unsupported — multi-argument inputs not modelled (LPPAD issue #26, wsmoses)
#   - Convention mismatches papered over — e.g. Zygote's pushforward returns a callable, not (y, ẏ)
#   - Extensions in DI's repo — maintained by the wrong team for every backend

#   Right column — Mooncake-specific breakage:
#   - Calls Mooncake internals (CoDual, _copy_output, MinimalCtx)
#   - Version-detection hacks (isdefined(Mooncake, :FriendlyTangentCache))
#   - One version cap cascades: Mooncake ≤ 0.5.24 → ComponentArrays blocked → all of SciML blocked

#   Ecosystem verdict:
#   - Turing left DI (PR #1354) — back to native APIs
#   - Lux declined DI — mandatory AbstractVector input a poor fit (issue #544)

#   Bottom:
#   ▎ VG.jl: each backend implements two primitives. Everything else is derived. No internals touched.

#   ---
#   Script:

#   ▎ "DI has two classes of problems. The first is general — design gaps that affect every backend. The AbstractVector input requirement is a poor fit for ML frameworks like Lux, where 
#   ▎ parameters are fmap-based structures like NamedTuples — Lux's maintainer raised this directly in issue #544. Enzyme's activity states weren't supported in DI's abstraction — 
#   ▎ wsmoses flagged this in 2024, saying it would be a blocker until DI models multi-argument inputs properly. And convention mismatches between backends — Zygote's pushforward returns
#   ▎ a callable, not a value-tangent pair — get papered over rather than properly resolved. The second class is Mooncake-specific: DI reaches into Mooncake's private internals, uses 
#   ▎ functions that were never part of the public contract, and version-detects with isdefined hacks. When Mooncake changes, DI breaks, and one pinned version cascades through 
#   ▎ ComponentArrays and all of SciML. Turing saw this and left. VG.jl's answer is simple: each backend implements two primitives against its own public API. Everything else is derived 
#   ▎ from those. Nobody touches anyone's internals."


#   ---
#   Slide 3 — The API

#   Heading: Five operations, one consistent interface

#   ┌─────────────────────────┬──────────────────┬─────────┐
#   │        Operation        │     Returns      │  Mode   │
#   ├─────────────────────────┼──────────────────┼─────────┤
#   │ value_and_pullback!!    │ (y, x̄)           │ Reverse │
#   ├─────────────────────────┼──────────────────┼─────────┤
#   │ value_and_pushforward!! │ (y, ẏ)           │ Forward │
#   ├─────────────────────────┼──────────────────┼─────────┤
#   │ value_and_gradient!!    │ (y, ∇f)          │ Reverse │
#   ├─────────────────────────┼──────────────────┼─────────┤
#   │ value_and_derivative!!  │ (y, df/dx)       │ Forward │
#   ├─────────────────────────┼──────────────────┼─────────┤
#   │ value_and_jacobian!!    │ (y, (J₁, J₂, …)) │ Either  │
#   └─────────────────────────┴──────────────────┴─────────┘
  
#   - ad_cache and canonical_tangents on all five
#   - Output of f unconstrained throughout
#   - First-order only — HVP/Hessian excluded by design

#   Script:

#   ▎ "The package exposes five functions. The first four are thin wrappers — they add type checking and a consistent ad_cache and canonical_tangents interface, then delegate directly to
#   ▎ Mooncake's public API. No logic, no transformation, zero overhead over calling Mooncake yourself. The interesting one is value_and_jacobian!!, which returns a tuple of Jacobian 
#   ▎ matrices — one per input argument — and has to do real work to support the full range of input types. That's what the next two slides cover. Note that we deliberately exclude 
#   ▎ Hessians and HVPs — those require close coupling with backend internals, which would put us right back in DI's position."                       
   
#   ---
#   Slide 4 — DiffInput

#   Heading: Supported input types: DiffInput

#   DiffInput  ← single-argument calls
#   ├── DiffLeaf
#   │   ├── DiffScalar  ─── Float32, Float64, ComplexF64, …
#   │   └── DiffArray   ─── AbstractArray{<:DiffScalar}
#   │                       (any shape, any real/complex float eltype)
#   └── Tuple{Vararg{DiffLeaf}}  ─── f takes a tuple as its single input

#   Multi-arg calls  ← separate dispatch
#     value_and_jacobian!!(f, backend, x₁, x₂, …)
#     each xᵢ must be a DiffLeaf — returns one Jacobian per argument

#   # Single-arg examples:
#   2.0                        # DiffScalar
#   randn(Float32, 4)          # DiffArray
#   randn(ComplexF64, 3, 3)    # DiffArray
#   (randn(4), randn(2,2))     # Tuple{DiffLeaf,DiffLeaf} — f takes a tuple

#   # Multi-arg example:
#   value_and_jacobian!!(f, backend, randn(4), randn(3))  # → (y, (J₁, J₂))

#   Output of f is completely unconstrained — scalar, vector, matrix, struct. VG.jl flattens it internally to build the Jacobian.
  

#   Script:

#   ▎ "DiffInput is the constraint on what value_and_jacobian!! accepts as input. A DiffScalar is any real or complex IEEE float — Float32, Float64, ComplexF64. A DiffArray is any 
#   ▎ AbstractArray over those — vectors, matrices, arbitrary shapes. And you can pass a Tuple of DiffLeafs for multi-argument functions, in which case you get back one Jacobian per 
#   ▎ argument. Crucially, the output of f is completely unconstrained. VG.jl just calls _flatten on whatever comes back and uses that to build the rows or columns of the Jacobian. This 
#   ▎ is a deliberate design choice — we don't want to constrain what functions you can differentiate."

#   ---
#   Slide 5 — Architecture

#   Heading: Two-layer dispatch

#   value_and_jacobian!!(f, backend, x)
#              │
# AbstractVector{<:IEEEFloat} + Mooncake?
#              │
#        ┌─────┴──────┐
#       YES           NO
#        │             │
#     Layer 1       Layer 2
#     1 AD pass     Forward: n pushforwards
#                   Reverse: m pullbacks

#   - ad_cache: skip rule compilation on repeat calls — 5× speedup
#   - @warn on Layer 2: reports passes used, suggests cheaper mode

#   Script:

#   ▎ "Layer 1 is the native Mooncake path — plain real vector input, we call Mooncake.value_and_jacobian!! directly. One compiled rule, one pass. Layer 2 is the derived path — 
#   ▎ everything else: matrix inputs, complex inputs, scalar inputs, multi-arg calls. Forward mode builds the Jacobian column by column — one pushforward per input dimension, O(n). 
#   ▎ Reverse mode builds it row by row — one pullback per output dimension, O(m). Which is cheaper depends on the shape of your Jacobian. VG.jl emits a warning on Layer 2 that tells you
#   ▎ exactly how many passes it made and whether a better mode exists. And if you're in a hot loop, pass ad_cache — that alone gives a 5× speedup."

#   ---
#   Slide 6 — Remaining: SciML CPU
  
#   Heading: SciML CPU — 4 items remaining

# #   ┌─────┬──────────────────────────────────────┬────────────┐
# #   │  #  │                 Item                 │            │
# #   ├─────┼──────────────────────────────────────┼────────────┤
# #   │ 1   │ ComponentArray(::NamedTuple) rrule!! │ not done   │
# #   ├─────┼──────────────────────────────────────┼────────────┤
# #   │ 2   │ get_sampled_data rrule!!             │ not done   │
# #   ├─────┼──────────────────────────────────────┼────────────┤
# #   │ 3   │ RecursiveArrayTools VectorOfArray    │ in progress│
# #   ├─────┼──────────────────────────────────────┼────────────┤
# #   │ 4   │ SciMLSensitivity #1426               │ not done   │
# #   └─────┴──────────────────────────────────────┴────────────┘

#   Script:

#   ▎ "Four SciML CPU items remain. The ComponentArray NamedTuple rule is the most load-bearing — there's no reverse-mode rule for constructing a ComponentArray from a NamedTuple, and it
#   ▎ also unblocks item 4. The get_sampled_data issue is a classic AD failure mode: a floor division inside the function makes the input invisible to tracing, so you get zero gradient 
#   ▎ silently. Item 3 is essentially done locally, just needs evaluation before a PR. Item 4 depends on an upstream Enzyme fix that may already be in — needs a re-test."

#   ---
#   Slide 7 — Remaining: GPU Rules

#   Heading: GPU — 5 rules, covers all Lux + Flux GPU layers

# #   ┌─────┬─────────────────────────────────┬────────────────────────────────────┐
# #   │  #  │              Rule               │              Unblocks              │
# #   ├─────┼─────────────────────────────────┼────────────────────────────────────┤
# #   │ 5   │ Statistics.varm                 │ GroupNorm, InstanceNorm, LayerNorm │
# #   ├─────┼─────────────────────────────────┼────────────────────────────────────┤
# #   │ 6   │ Base.permutedims                │ MultiHeadAttention                 │
# #   ├─────┼─────────────────────────────────┼────────────────────────────────────┤
# #   │ 7   │ vcat/hcat/cat                   │ SkipConnection                     │
# #   ├─────┼─────────────────────────────────┼────────────────────────────────────┤
# #   │ 8   │ batchnorm_cudnn!                │ BatchNorm GPU + CPU correctness    │
# #   ├─────┼─────────────────────────────────┼────────────────────────────────────┤
# #   │ 9   │ StatefulRecurrentCell setfield! │ RNN/LSTM/GRU (Lux only)            │
# #   └─────┴─────────────────────────────────┴────────────────────────────────────┘

  
#   Script:

#   ▎ "Five GPU rules cover the full remaining surface area for both Lux and Flux. Each one is a targeted rrule!! in MooncakeCUDAExt.jl. The varm rule is the highest coverage — one rule 
#   ▎ unblocks three layer types across two frameworks. The batchnorm_cudnn! rule also upgrades Lux BatchNorm on CPU from interface-only to full correctness — a two-for-one. Worth 
#   ▎ noting: Flux GPU tests will always be interface-only regardless, because Flux stores its weights as CPU types even when the model is on GPU. That's an architectural issue in Flux, 
#   ▎ not something Mooncake can solve."

#   ---
#   Slide 8 — Remaining: VG.jl

#   Heading: VG.jl — 6 items


#   ┌─────┬────────────────────────────────────────┬─────┐
#   │  #  │                  Item                  │     │
#   ├─────┼────────────────────────────────────────┼─────┤
#   │ 10  │ _canonicalize post-norm layer          │ ❌  │
#   ├─────┼────────────────────────────────────────┼─────┤
#   │ 11  │ ForwardDiff backend                    │ ❌  │
#   ├─────┼────────────────────────────────────────┼─────┤
#   │ 12  │ Zygote backend                         │ ❌  │
#   ├─────┼────────────────────────────────────────┼─────┤
#   │ 13  │ FiniteDifferences pullback/pushforward │ ❌  │
#   ├─────┼────────────────────────────────────────┼─────┤
#   │ 14  │ Enzyme backend                         │ ❌  │
#   ├─────┼────────────────────────────────────────┼─────┤
#   │ 15  │ DynamicPPL VG.jl integration           │ ❌  │
#   └─────┴────────────────────────────────────────┴─────┘

# #   ┌─────┬────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
# #   │  #  │                  Item                  │                                        Notes                                        │
# #   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
# #   │ 10  │ _canonicalize post-norm layer          │ Convert backend internal tangent types (e.g. Mooncake.Tangent) to plain Julia types │
# #   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
# #   │ 14  │ DI backend                             │ DI Extension, integration                                                           │
# #   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
# #   │ 11  │ ForwardDiff backend                    │ True primitives: gradient, jacobian, derivative - rest derived automatically        │
# #   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
# #   │ 12  │ Zygote backend                         │ True primitives: pullback, gradient - rest derived                                  │
# #   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
# #   │ 13  │ FiniteDifferences pullback/pushforward │ j′vp/jvp exist in FiniteDifferences, just not wired in                              │
# #   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
# #   │ 14  │ Enzyme backend                         │ Complex activity states — hardest                                                   │
# #   ├─────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
# #   │ 15  │ DynamicPPL VG.jl integration           │ Formalises PR #1354's ad-hoc backend calls into a proper protocol                   │
# #   └─────┴────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘
#     ​

# xᵢ are DiffInputs, y is a Tuple of Matrices containing the Jacobians of f with respect to each xᵢ.
# ​
# Multi-arg dispatch:
# y = (J₁, J₂, …) = value_and_jacobian!!(f, backend, x₁, x₂, …)

# Single-arg dispatch:
# y = (J₁) = value_and_jacobian!!(f, backend, x₁)

# ​
# # Example:​

# value_and_jacobian!!(f, backend, randn(4), randn(3)) # (y, (J₁, J₂))​

# ​

# ​