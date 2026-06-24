module ValueAndGradientForwardDiffExt

using ValueAndGradient: ValueAndGradient
using ForwardDiff: ForwardDiff
using ADTypes: AutoForwardDiff

_perturb(x::AbstractArray, ẋ, t) = x .+ t .* ẋ
_perturb(x::Number, ẋ, t) = x + t * ẋ
_perturb(x::Tuple, ẋ::Tuple, t) = map((xi, ẋi) -> _perturb(xi, ẋi, t), x, ẋ)

function ValueAndGradient.value_and_pushforward!!(
        f::F, ẋ, backend::AutoForwardDiff, xs...;
        ad_cache=nothing, canonical_tangents=false, kwargs...) where {F}
    ad_cache !== nothing && @warn "AutoForwardDiff does not support ad_cache; it will be ignored."
    y = f(xs...)
    N = length(xs)
    if N == 1
        x = only(xs)
        ẏ = ForwardDiff.derivative(t -> f(_perturb(x, ẋ, t)), zero(Float64))
    else
        ẏ = ForwardDiff.derivative(
            t -> f(ntuple(i -> _perturb(xs[i], ẋ[i], t), N)...),
            zero(Float64),
        )
    end
    return y, canonical_tangents ? ValueAndGradient._canonicalize(y, ẏ) : ẏ
end

end
