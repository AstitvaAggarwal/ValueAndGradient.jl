module ValueAndGradientEnzymeExt

using ValueAndGradient: ValueAndGradient
using Enzyme: Enzyme
using ADTypes: AutoEnzyme

_vdot(ȳ::Number, y::Number) = real(conj(ȳ) * y)
_vdot(ȳ::AbstractArray, y::AbstractArray) = real(sum(conj.(ȳ) .* y))
_vdot(ȳ::Tuple, y::Tuple) = sum(_vdot(ȳi, yi) for (ȳi, yi) in zip(ȳ, y))
_vdot(ȳ::NamedTuple{K}, y::NamedTuple{K}) where {K} = sum(_vdot(ȳ[k], y[k]) for k in K)

# Reverse mode: gradient of dot(ȳ, f(x)) gives the VJP.
# Const(h) tells Enzyme the closure is read-only (ȳ and f are not differentiated).
function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoEnzyme,
    x::AbstractArray;
    ad_cache = nothing,
    normalise_tangents = false,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoEnzyme does not support ad_cache; it will be ignored."
    y = f(x)
    ∂x = zero(x)
    h = Enzyme.Const(xi -> _vdot(ȳ, f(xi)))
    Enzyme.autodiff(Enzyme.Reverse, h, Enzyme.Active, Enzyme.Duplicated(copy(x), ∂x))
    return y, normalise_tangents ? ValueAndGradient._normalise(x, ∂x, backend) : ∂x
end

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoEnzyme,
    x1::AbstractArray,
    x2::AbstractArray,
    xrest::AbstractArray...;
    ad_cache = nothing,
    normalise_tangents = false,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoEnzyme does not support ad_cache; it will be ignored."
    xs = (x1, x2, xrest...)
    y = f(xs...)
    ∂xs = map(zero, xs)
    h = Enzyme.Const((args...) -> _vdot(ȳ, f(args...)))
    dups = map((xi, ∂xi) -> Enzyme.Duplicated(copy(xi), ∂xi), xs, ∂xs)
    Enzyme.autodiff(Enzyme.Reverse, h, Enzyme.Active, dups...)
    return y, normalise_tangents ? ValueAndGradient._normalise(xs, ∂xs, backend) : ∂xs
end

# Forward mode: autodiff returns a 1-element result in Enzyme v0.13.
# result[1] is the shadow (tangent); call f separately for the primal.
function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ::AbstractArray,
    backend::AutoEnzyme,
    x::AbstractArray;
    ad_cache = nothing,
    normalise_tangents = false,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoEnzyme does not support ad_cache; it will be ignored."
    y = f(x)
    result = Enzyme.autodiff(
        Enzyme.Forward,
        f,
        Enzyme.Duplicated,
        Enzyme.Duplicated(copy(x), copy(ẋ)),
    )
    ẏ = result[1]
    return y, normalise_tangents ? ValueAndGradient._normalise(y, ẏ, backend) : ẏ
end

function ValueAndGradient.value_and_pushforward!!(
    f::F,
    ẋ,
    backend::AutoEnzyme,
    x1::AbstractArray,
    x2::AbstractArray,
    xrest::AbstractArray...;
    ad_cache = nothing,
    normalise_tangents = false,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoEnzyme does not support ad_cache; it will be ignored."
    xs = (x1, x2, xrest...)
    y = f(xs...)
    dups = map((xi, dxi) -> Enzyme.Duplicated(copy(xi), copy(dxi)), xs, ẋ)
    result = Enzyme.autodiff(Enzyme.Forward, f, Enzyme.Duplicated, dups...)
    ẏ = result[1]
    return y, normalise_tangents ? ValueAndGradient._normalise(y, ẏ, backend) : ẏ
end

end
