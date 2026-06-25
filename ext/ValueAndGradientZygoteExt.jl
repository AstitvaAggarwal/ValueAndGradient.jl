module ValueAndGradientZygoteExt

using ValueAndGradient: ValueAndGradient
using Zygote: Zygote
using ADTypes: AutoZygote

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoZygote,
    xs...;
    ad_cache = nothing,
    canonical_tangents = false,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoZygote does not support ad_cache; it will be ignored."
    y, back = Zygote.pullback(f, xs...)
    x̄s = back(ȳ)
    if length(xs) == 1
        x̄ = only(x̄s)
        return y, canonical_tangents ? ValueAndGradient._canonicalize(only(xs), x̄, backend) : x̄
    else
        return y, canonical_tangents ? ValueAndGradient._canonicalize(xs, x̄s, backend) : x̄s
    end
end

end
