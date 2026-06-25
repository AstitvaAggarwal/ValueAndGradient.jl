module ValueAndGradient

using ADTypes: AbstractADType, mode, ForwardMode, ReverseMode

"IEEE floats and their complex counterparts: `Float16/32/64` and `Complex{<:IEEEFloat}`."
const DiffScalar = Union{Base.IEEEFloat,Complex{<:Base.IEEEFloat}}

"Any `AbstractArray` whose element type is a `DiffScalar`."
const DiffArray = AbstractArray{<:DiffScalar}

"A single differentiable value: scalar or array of scalars (`DiffScalar | DiffArray`)."
const DiffLeaf = Union{DiffScalar,DiffArray}

"""
    DiffInput

The set of input types accepted by `value_and_pullback!!` and `value_and_pushforward!!`:
a single `DiffLeaf` (scalar or array), or a `Tuple` of `DiffLeaf`s for multi-input functions.
"""
const DiffInput = Union{DiffLeaf,Tuple{Vararg{DiffLeaf}}}

"""
    value_and_pullback!!(f, ȳ, backend, x...; ad_cache=nothing, canonical_tangents=false) -> (y, x̄)

Returns `y = f(x...)` and the VJP `x̄ = (∂f/∂x)ᵀ ȳ`.
`ȳ` must match the output type of `f`: scalar, array, or tuple thereof.
Single argument: `x̄` has the same structure as `x`.
Multiple arguments: `x̄` is a tuple of per-argument cotangents.

Pass a backend-specific `ad_cache` to reuse it across repeated calls.
If `nothing`, the backend builds one internally (convenience path — not efficient for loops).

When `canonical_tangents=true`, known backend-specific wrapper types are stripped before
returning. `nothing` (Zygote's cotangent for unused arguments) becomes `zero(x)`.
`Mooncake.Tangent` (struct output from Mooncake pushforward) is unwrapped to its fields
NamedTuple, then reconstructed as `T(values(nt)...)` if `T` has a matching positional
constructor — otherwise the NamedTuple is returned with a warning. Everything else passes
through unchanged. Default `false` returns whatever the backend produces.

For `DiffLeaf` inputs with scalar/array outputs all backends agree after normalisation.
Two known gaps: other forward-mode backends (e.g. Enzyme) don't normalise struct tangents,
and the derived pullback for forward-mode backends gives real tangents for complex inputs.
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

# internal helpers

_vdot(a::Number, b::Number) = real(conj(a) * b)
_vdot(a::AbstractArray, b::AbstractArray) =
    real(sum(conj(ai) * bi for (ai, bi) in zip(a, b)))
_vdot(a::Tuple, b::Tuple) = sum(_vdot(ai, bi) for (ai, bi) in zip(a, b))
_vdot(a::NamedTuple{K}, b::NamedTuple{K}) where {K} = sum(_vdot(a[k], b[k]) for k in K)

# Canonical tangent normalization (called by extension methods when canonical_tangents=true).
_zero_like(x::Number) = zero(real(x))
_zero_like(x::AbstractArray) = zero(x)
_zero_like(x::Tuple) = map(_zero_like, x)

_canonicalize(x, ::Nothing, backend) = _zero_like(x)
_canonicalize(x::DiffLeaf, t::DiffLeaf, backend) = t
_canonicalize(xs::Tuple, ts::Tuple, backend) =
    map((x, t) -> _canonicalize(x, t, backend), xs, ts)
# TODO: ChainRulesCore.NoTangent and ZeroTangent are not mapped to zero(x) here.
# For all 9 current backends this is a non-issue:
#   - Zygote converts AbstractZero → nothing via wrap_chainrules_output before returning,
#     so back(ȳ) only ever gives nothing or plain arrays for DiffLeaf inputs.
#   - All other backends produce plain arrays/scalars directly.
# A future backend that surfaces ChainRulesCore types directly to the caller would hit this.
# Fix if that arises: add
#   using ChainRulesCore: NoTangent, ZeroTangent
#   _canonicalize(x, ::Union{NoTangent,ZeroTangent}, backend) = _zero_like(x)
# TODO: struct outputs in pushforward — Mooncake normalises via its _canonicalize overload
# (Mooncake.Tangent → NamedTuple → struct). Other forward-mode backends (e.g. Enzyme)
# return their own shadow type for struct outputs; add a backend-specific overload here
# once Enzyme's exact shadow type for struct-returning f is confirmed.
function _canonicalize(x::T, t, backend) where {T}
    t isa NamedTuple || return t
    try
        return T(values(t)...)
    catch
        @warn "canonical_tangents=true: cannot reconstruct $(T) from tangent NamedTuple; " *
              "returning tangent as-is. Define a positional constructor for $(T) to enable reconstruction."
        return t
    end
end

# Derive ẏ = Jẋ by running pullback calls (one per output leaf element).
# Dispatches recursively on output type so Tuple and NamedTuple outputs are supported.
function _pushforward_via_pullback(f::F, ẋ, backend, xs...; kwargs...) where {F}
    y = f(xs...)
    return y, _pf_from_pb(f, y, ẋ, backend, xs...; kwargs...)
end

function _pf_from_pb(f::F, y::Number, ẋ, backend, xs...; kwargs...) where {F}
    _, x̄ = value_and_pullback!!(f, one(real(y)), backend, xs...; kwargs...)
    return typeof(real(y))(_vdot(x̄, ẋ))
end

function _pf_from_pb(f::F, y::AbstractArray, ẋ, backend, xs...; kwargs...) where {F}
    ẏ = similar(y, real(eltype(y)))
    for j in eachindex(y)
        eⱼ = zero(y)
        eⱼ[j] = one(eltype(y))
        _, x̄ = value_and_pullback!!(f, eⱼ, backend, xs...; kwargs...)
        ẏ[j] = _vdot(x̄, ẋ)
    end
    return ẏ
end

function _pf_from_pb(f::F, y::Tuple, ẋ, backend, xs...; kwargs...) where {F}
    ntuple(length(y)) do k
        fk = (args...) -> f(args...)[k]
        _pf_from_pb(fk, y[k], ẋ, backend, xs...; kwargs...)
    end
end

function _pf_from_pb(f::F, y::NamedTuple{Ks}, ẋ, backend, xs...; kwargs...) where {F,Ks}
    vals = ntuple(length(Ks)) do i
        k = Ks[i]
        fk = (args...) -> f(args...)[k]
        _pf_from_pb(fk, y[k], ẋ, backend, xs...; kwargs...)
    end
    return NamedTuple{Ks}(vals)
end

function _pf_from_pb(f::F, y, ẋ, backend, xs...; kwargs...) where {F}
    throw(
        ArgumentError(
            "Derived pushforward via pullback does not support output type $(typeof(y)). " *
            "Use a native forward-mode backend: AutoMooncakeForward, AutoForwardDiff, " *
            "AutoFiniteDiff, AutoFiniteDifferences, or AutoEnzyme(mode=Enzyme.Forward).",
        ),
    )
end

# Derive x̄ = Jᵀȳ by running n pushforward calls (one per input element).
# Reduces f to h = _vdot(ȳ, f(...)) before calling the backend so the backend
# always sees a scalar-returning function — works for any output type _vdot supports.
# TODO: similar(x, real(eltype(x))) strips the imaginary part for complex array inputs.
# Native reverse-mode backends return complex tangents for complex inputs; this derived
# path gives real tangents instead. Fix if complex + forward-mode-derived-pullback matters.
function _pullback_via_pushforward(f::F, ȳ, backend, xs...; kwargs...) where {F}
    N = length(xs)
    y = f(xs...)
    h = (args...) -> _vdot(ȳ, f(args...))
    if N == 1
        x = only(xs)
        if x isa Number
            _, dh = value_and_pushforward!!(h, one(real(x)), backend, x; kwargs...)
            return y, typeof(real(x))(dh)
        elseif x isa AbstractArray
            x̄ = similar(x, real(eltype(x)))
            for i in eachindex(x)
                eᵢ = zero(x)
                eᵢ[i] = one(eltype(x))
                _, x̄[i] = value_and_pushforward!!(h, eᵢ, backend, x; kwargs...)
            end
            return y, x̄
        else
            throw(
                ArgumentError(
                    "Derived pullback via pushforward requires scalar or AbstractArray input. Got $(typeof(x)).",
                ),
            )
        end
    else
        x̄s = ntuple(N) do k
            xk = xs[k]
            xk isa AbstractArray || throw(
                ArgumentError(
                    "Derived pullback via pushforward: multi-arg inputs must be AbstractArray. Got $(typeof(xk)).",
                ),
            )
            similar(xk, real(eltype(xk)))
        end
        for k = 1:N
            for i in eachindex(xs[k])
                ẋ = ntuple(N) do j
                    if j == k
                        e = zero(xs[j])
                        e[i] = one(eltype(xs[j]))
                        e
                    else
                        zero(xs[j])
                    end
                end
                _, x̄s[k][i] = value_and_pushforward!!(h, ẋ, backend, xs...; kwargs...)
            end
        end
        return y, x̄s
    end
end

# fallbacks

function value_and_pullback!!(f, ȳ, backend::AbstractADType, xs...; kwargs...)
    if mode(backend) isa ForwardMode
        @warn "value_and_pullback!! is not natively supported by $(typeof(backend)) " *
              "(a forward-mode backend). Falling back to $(sum(length, xs; init=0)) " *
              "pushforward call(s). Use value_and_pushforward!! for efficiency."
        return _pullback_via_pushforward(f, ȳ, backend, xs...; kwargs...)
    else
        throw(
            ArgumentError(
                "value_and_pullback!! is not supported for $(typeof(backend)). " *
                "Use a reverse-mode backend: AutoMooncake, AutoReverseDiff, AutoZygote, " *
                "AutoTracker, or AutoEnzyme(mode=Enzyme.Reverse).",
            ),
        )
    end
end

function value_and_pushforward!!(f, ẋ, backend::AbstractADType, xs...; kwargs...)
    if mode(backend) isa ReverseMode
        @warn "value_and_pushforward!! is not natively supported by $(typeof(backend)) " *
              "(a reverse-mode backend). Falling back to pullback call(s). " *
              "Use value_and_pullback!! for efficiency."
        return _pushforward_via_pullback(f, ẋ, backend, xs...; kwargs...)
    else
        throw(
            ArgumentError(
                "value_and_pushforward!! is not supported for $(typeof(backend)). " *
                "Use a forward-mode backend: AutoMooncakeForward, AutoForwardDiff, " *
                "AutoFiniteDiff, AutoFiniteDifferences, or AutoEnzyme(mode=Enzyme.Forward).",
            ),
        )
    end
end

export value_and_pullback!!, value_and_pushforward!!, test_pullback, test_pushforward

end
