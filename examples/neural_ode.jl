# Neural ODE via ValueAndGradient.jl
#
# True dynamics:  dz/dt = A_true·z   (stable spiral: A_true = [[-0.1, -1], [1, -0.1]])
# Learned model:  dz/dt = nn(z; ps)  (Lux Dense(2→2, no bias) — learns A_true)
# Loss:           MSE between NN-ODE trajectory and true trajectory
#
# The ODE RHS calls the Lux model at each solver step.
# Mooncake traces through the ODE solver and the network jointly.
#
# Backends:  AutoMooncake, AutoZygote (gradient check before training)
#
# Setup:
#   julia> using Pkg; Pkg.add(["Lux","OrdinaryDiffEq","SciMLSensitivity","Optimisers",
#                               "Zygote","FiniteDifferences","Random"])

using Lux, OrdinaryDiffEq, Optimisers, Random
try
    ;
    using SciMLSensitivity;
catch
    ;
end   # optional: provides efficient ODE adjoints
using ValueAndGradient
using ADTypes: AutoMooncake, AutoZygote, AutoFiniteDifferences
using Mooncake
using Zygote
using FiniteDifferences: central_fdm
using Printf

rng = Random.default_rng()
Random.seed!(rng, 0)

# --- true dynamics: stable spiral ---
A_true = Float32[-0.1 -1.0; 1.0 -0.1]
u0 = Float32[1.0, 0.0]
tspan = (0.0f0, 3.0f0)
tsave = range(0.0f0, 3.0f0; length = 31)

target = let
    prob = ODEProblem((u, _, t) -> A_true * u, u0, tspan)
    Array(solve(prob, Tsit5(); saveat = tsave))
end

# --- neural ODE model: Dense(2→2, no bias) ---
model = Dense(2, 2; use_bias = false)
ps, st = Lux.setup(rng, model)

ps_flat, re = Optimisers.destructure(ps)   # Float32 vector (4 elements = 2×2 weight)

function loss(p)
    ps_nt = re(p)
    function rhs(u, _, t)
        y, _ = Lux.apply(model, u, ps_nt, st)
        return y
    end
    prob = ODEProblem(rhs, u0, tspan)
    sol = solve(prob, Tsit5(); saveat = tsave, abstol = 1e-6, reltol = 1e-6)
    sum(abs2, Array(sol) .- target)
end

# --- gradient check: Mooncake vs Zygote ---
println("Neural ODE gradient check (Mooncake vs Zygote):\n")

check_backends = [
    "AutoMooncake" => AutoMooncake(config = nothing),
    "AutoZygote" => AutoZygote(),
    "AutoFiniteDifferences" => AutoFiniteDifferences(fdm = central_fdm(5, 1)),
]

grads = map(check_backends) do (name, backend)
    try
        y, dp = value_and_pullback!!(loss, one(Float32), backend, ps_flat)
        @printf("  %-24s  loss = %.4f   ∇p = %s\n", name, y, string(round.(dp; digits = 4)))
        dp
    catch e
        @printf("  %-24s  ERROR: %s\n", name, first(sprint(showerror, e), 120))
        nothing
    end
end

println()
valid = [(n, g) for ((n, _), g) in zip(check_backends, grads) if !isnothing(g)]
if length(valid) >= 2
    ref = last(valid)[2]
    for (name, dp) in valid
        match = isapprox(dp, ref; rtol = 1e-3)
        @printf("  %-24s  agrees with FD: %s\n", name, match ? "✓" : "✗ (rtol=1e-3)")
    end
else
    isempty(valid) && println("  (all backends failed)")
end

# --- training loop with AutoMooncake ---
println("\nTraining neural ODE (AutoMooncake, η=0.05, 100 steps):")
mc = AutoMooncake(config = nothing)
p = copy(ps_flat)
η = 0.05f0
trained = false

try
    for step = 1:100
        y, dp = value_and_pullback!!(loss, one(Float32), mc, p)
        p .-= η .* dp
        step in (1, 25, 50, 100) && @printf("  step %3d   loss = %.4f\n", step, y)
    end
    global trained = true
catch e
    println("  Training failed: ", first(sprint(showerror, e), 120))
end

if trained
    learned_A = reshape(re(p).weight, 2, 2)
    println("\nLearned A:")
    display(learned_A)
    println("True A:")
    display(A_true)
end
