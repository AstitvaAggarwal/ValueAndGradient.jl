module ValueAndGradientTrackerExt

using ValueAndGradient: ValueAndGradient
using Tracker: Tracker
using ADTypes: AutoTracker

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoTracker,
    x::AbstractArray;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pullback = nothing,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoTracker does not support ad_cache; it will be ignored."
    y_tracked, back = Tracker.forward(f, x)
    x̄s = back(ȳ)
    x̄ = Tracker.data(only(x̄s))
    return Tracker.data(y_tracked),
    ValueAndGradient._apply_norm(x, x̄, backend, normalise_tangents, normalise_pullback)
end

function ValueAndGradient.value_and_pullback!!(
    f::F,
    ȳ,
    backend::AutoTracker,
    x1::AbstractArray,
    x2::AbstractArray,
    xrest::AbstractArray...;
    ad_cache = nothing,
    normalise_tangents = false,
    normalise_pullback = nothing,
    kwargs...,
) where {F}
    ad_cache !== nothing &&
        @warn "AutoTracker does not support ad_cache; it will be ignored."
    xs = (x1, x2, xrest...)
    y_tracked, back = Tracker.forward(f, x1, x2, xrest...)
    x̄s = map(Tracker.data, back(ȳ))
    return Tracker.data(y_tracked),
    ValueAndGradient._apply_norm(xs, x̄s, backend, normalise_tangents, normalise_pullback)
end

end
