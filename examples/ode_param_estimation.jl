# Gradient of an ODE loss function via ValueAndGradient.jl
#
# Model:     du/dt = -θ·u,   u(0) = 1,   t ∈ [0, 1]
# Task:      compute ∂L/∂θ where L = MSE(simulated, observed)
# Backends:  AutoMooncake, AutoZygote, AutoFiniteDifferences — all via the same loss function
#
# SciMLSensitivity is loaded to provide efficient adjoint rules for solve.
# Without it, Mooncake traces through solver steps directly (correct but slower).
#
# Setup (from the repo root):
#   julia> using Pkg; Pkg.add(["OrdinaryDiffEq","SciMLSensitivity","Zygote","FiniteDifferences"])

using OrdinaryDiffEq
try
    using SciMLSensitivity
catch
    @warn "SciMLSensitivity not available; Mooncake will trace through solver internals directly (correct but slower)."
end
using ValueAndGradient
using ADTypes: AutoMooncake, AutoZygote, AutoFiniteDifferences
using Mooncake
using Zygote
using FiniteDifferences: central_fdm
using Printf

# --- problem ---
θ_true = [2.0]
tspan = (0.0, 1.0)
tsave = range(0.0, 1.0; length = 11)

ode_f(u, θ, t) = -θ .* u

data = let
    prob = ODEProblem(ode_f, [1.0], tspan, θ_true)
    Array(solve(prob, Tsit5(); saveat = tsave))
end

function loss(θ)
    prob = ODEProblem(ode_f, [1.0], tspan, θ)
    sol = solve(prob, Tsit5(); saveat = tsave, abstol = 1e-9, reltol = 1e-9)
    sum(abs2, Array(sol) .- data)
end

# --- gradient comparison ---
θ₀ = [1.5]

backends = [
    "AutoMooncake" => AutoMooncake(config = nothing),
    "AutoZygote" => AutoZygote(),
    "AutoFiniteDifferences" => AutoFiniteDifferences(fdm = central_fdm(5, 1)),
]

println("ODE parameter estimation — ∂L/∂θ at θ=$(θ₀[1])\n")
results = map(backends) do (name, backend)
    try
        y, dθ = value_and_pullback!!(loss, one(Float64), backend, θ₀)
        @printf("  %-24s  loss = %.6f   ∇θ = %.6f\n", name, y, dθ[1])
        dθ
    catch e
        @printf("  %-24s  FAILED: %s\n", name, sprint(showerror, e))
        nothing
    end
end

println()
successful = [(name, dθ) for ((name, _), dθ) in zip(backends, results) if dθ !== nothing]
if length(successful) >= 2
    ref = last(successful)[2]
    for (name, dθ) in successful
        match = isapprox(dθ, ref; rtol = 1e-4)
        @printf("  %-24s  agrees with ref: %s\n", name, match ? "✓" : "✗ (rtol=1e-4)")
    end
end

# --- short gradient descent to verify convergence ---
println("\nGradient descent (AutoMooncake, η=0.5, 20 steps):")
θ = copy(θ₀)
mc = AutoMooncake(config = nothing)
try
    for step = 1:20
        y, dθ = value_and_pullback!!(loss, one(Float64), mc, θ)
        θ .-= 0.5 .* dθ
    end
    @printf("  θ_recovered = %.4f  (true: %.1f)\n", θ[1], θ_true[1])
catch e
    @printf("  Gradient descent FAILED: %s\n", sprint(showerror, e))
end
