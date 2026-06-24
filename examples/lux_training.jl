# Lux model training via ValueAndGradient.jl
#
# Task:      learn y = sin(x₁) + cos(x₂) with a small MLP
# Interface: value_and_pullback!! replaces Zygote.gradient in the training loop
# Backends:  AutoMooncake (training), AutoZygote (gradient check)
#
# Parameters are flattened to a Vector{Float32} via Optimisers.destructure so they
# satisfy VG.jl's DiffInput constraint (AbstractArray{<:DiffScalar}).
#
# Setup:
#   julia> using Pkg; Pkg.add(["Lux","Optimisers","Zygote","FiniteDifferences","Random"])

using Lux, Optimisers, Random, Statistics
using ValueAndGradient
using ADTypes: AutoMooncake, AutoZygote, AutoFiniteDifferences
using Mooncake
using Zygote
using FiniteDifferences: central_fdm
using Printf

rng = Random.default_rng()
Random.seed!(rng, 42)

# --- synthetic data ---
n_data = 64
X = randn(rng, Float32, 2, n_data)
Y = sin.(X[1:1, :]) .+ cos.(X[2:2, :])

# --- model ---
model = Chain(Dense(2, 16, tanh), Dense(16, 1))
ps, st = Lux.setup(rng, model)

ps_flat, re = Optimisers.destructure(ps)   # Float32 vector

function loss(p)
    ps_nt = re(p)
    ŷ, _  = Lux.apply(model, X, ps_nt, st)
    mean(abs2, ŷ .- Y)
end

# --- gradient check: Mooncake vs Zygote on one step ---
println("Gradient check (one step, Mooncake vs Zygote):\n")

check_backends = [
    "AutoMooncake" => AutoMooncake(config=nothing),
    "AutoZygote"   => AutoZygote(),
]

grads = map(check_backends) do (name, backend)
    y, dp = value_and_pullback!!(loss, one(Float32), backend, ps_flat)
    @printf("  %-16s  loss = %.6f   ‖∇p‖ = %.6f\n", name, y, sqrt(sum(abs2, dp)))
    dp
end

match = isapprox(grads[1], grads[2]; rtol=1e-3)
println("\n  Mooncake agrees with Zygote (rtol=1e-3): $(match ? "✓" : "✗")")

# --- training loop with AutoMooncake ---
println("\nTraining (AutoMooncake, η=0.01, 200 steps):")
mc = AutoMooncake(config=nothing)
p  = copy(ps_flat)
η  = 0.01f0

for step in 1:200
    y, dp = value_and_pullback!!(loss, one(Float32), mc, p)
    p .-= η .* dp
    step in (1, 50, 100, 200) && @printf("  step %3d   loss = %.6f\n", step, y)
end
