# Gradient through a numerical integral via ValueAndGradient.jl
#
# Integrand:  f(x, θ) = exp(θ·x)
# Integral:   I(θ) = ∫₀¹ exp(θx) dx = (exp(θ) - 1) / θ
# Gradient:   I'(θ) = ∫₀¹ x·exp(θx) dx  =  exp(θ)·(θ-1)/θ² + 1/θ²  (by Leibniz rule)
#
# Backends:   AutoForwardDiff, AutoFiniteDifferences
# Validation: analytical formula above
#
# Note: AutoMooncake requires Integrals v5+ (IntegralsMooncakeExt), but Integrals v5
# requires SciMLBase versions that conflict with Mooncake 0.5's compat bounds.
# With a compatible env, IntegralsMooncakeExt bridges via @from_chainrules from
# Zygote's ChainRules adjoint for __solvebp, intercepting above QuadGK.
# AutoZygote is excluded: segfaults on Julia 1.12 inside QuadGK's error-handling path.
#
# Setup:
#   julia> using Pkg; Pkg.add(["Integrals","ForwardDiff","FiniteDifferences"])

using Integrals
using ValueAndGradient
using ADTypes: AutoForwardDiff, AutoFiniteDifferences
using ForwardDiff
using FiniteDifferences: central_fdm
using Printf

# analytical gradient for comparison
analytical(θ) = exp(θ[1]) * (θ[1] - 1) / θ[1]^2 + 1 / θ[1]^2

function loss(θ)
    prob = IntegralProblem((x, p) -> exp(p[1] * x), (0.0, 1.0), θ)
    solve(prob, QuadGKJL()).u
end

θ₀ = [2.0]

backends = [
    "AutoForwardDiff"       => AutoForwardDiff(),
    "AutoFiniteDifferences" => AutoFiniteDifferences(fdm=central_fdm(5, 1)),
]

println("Integral gradient — ∂I/∂θ at θ=$(θ₀[1])\n")
println("  Analytical:               ∇θ = $(round(analytical(θ₀); digits=8))\n")

results = map(backends) do (name, backend)
    try
        y, dθ = value_and_pullback!!(loss, one(Float64), backend, θ₀)
        @printf("  %-24s  I(θ) = %.8f   ∇θ = %.8f\n", name, y, dθ[1])
        dθ
    catch e
        @printf("  %-24s  ERROR: %s\n", name, sprint(showerror, e) |> x -> first(x, 80))
        nothing
    end
end

println()
ref = analytical(θ₀)
any(!isnothing, results) || println("  (all backends failed — no comparison possible)")
for ((name, _), dθ) in zip(backends, results)
    dθ === nothing && continue
    match = isapprox(dθ[1], ref; rtol=1e-4)
    @printf("  %-24s  agrees with analytical: %s\n", name, match ? "✓" : "✗ (rtol=1e-4)")
end

println("\n  AutoMooncake: requires Integrals v5+ (conflicts with Mooncake 0.5 SciMLBase compat in this env).
  AutoZygote: segfaults on Julia 1.12 + QuadGK (upstream issue).")
