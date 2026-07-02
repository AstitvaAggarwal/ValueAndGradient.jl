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
    normalise_tangents = false,
    normalise_pullback = nothing,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoZygote does not support ad_cache; it will be ignored."
    y, back = Zygote.pullback(f, xs...)
    x̄s = back(ȳ)
    if length(xs) == 1
        x̄ = only(x̄s)
        return y,
        ValueAndGradient._apply_norm(
            only(xs),
            x̄,
            backend,
            normalise_tangents,
            normalise_pullback,
        )
    else
        return y,
        ValueAndGradient._apply_norm(
            xs,
            x̄s,
            backend,
            normalise_tangents,
            normalise_pullback,
        )
    end
end

end
