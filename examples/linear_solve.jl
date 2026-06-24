# Gradient through a parameterised linear system via ValueAndGradient.jl
#
# System:   A(θ)·x = b   where A(θ) = θ₁·I + θ₂·tridiag(1)
# Loss:     L(θ) = ‖x(θ)‖²
# Gradient: ∂L/∂θ via matrix-transpose adjoint (Giles 2008)
#
# Backends: AutoMooncake, AutoZygote, AutoFiniteDifferences
# Validation: finite differences
#
# LinearSolveMooncakeExt provides a custom rrule!! for solve!/solve that uses
# the cached LU factorization directly in the adjoint, bypassing defaultalg_adjoint_eval
# (which Zygote hits and which is missing an adjoint(::Tuple{LU}) method).
#
# Setup:
#   julia> using Pkg; Pkg.add(["LinearSolve","Mooncake","Zygote","FiniteDifferences"])

using LinearSolve
using LinearAlgebra
using ValueAndGradient
using ADTypes: AutoMooncake, AutoZygote, AutoFiniteDifferences
using Mooncake
using Zygote
using FiniteDifferences: central_fdm
using Printf

const N = 5
const b = ones(N)

function make_A(θ)
    diag_vals    = fill(θ[1], N)
    offdiag_vals = fill(θ[2], N - 1)
    Tridiagonal(offdiag_vals, diag_vals, offdiag_vals)
end

function loss(θ)
    A   = make_A(θ)
    prob = LinearProblem(Matrix(A), b; sensealg=LinearSolveAdjoint())
    sol  = solve(prob)
    sum(abs2, sol.u)
end

θ₀ = [3.0, 0.5]

backends = [
    "AutoMooncake"          => AutoMooncake(config=nothing),
    "AutoZygote"            => AutoZygote(),
    "AutoFiniteDifferences" => AutoFiniteDifferences(fdm=central_fdm(5, 1)),
]

println("LinearSolve gradient — ∂L/∂θ at θ=$θ₀\n")
results = map(backends) do (name, backend)
    try
        y, dθ = value_and_pullback!!(loss, one(Float64), backend, θ₀)
        @printf("  %-24s  L = %.6f   ∇θ = [%.6f, %.6f]\n", name, y, dθ[1], dθ[2])
        dθ
    catch e
        @printf("  %-24s  ERROR: %s\n", name, sprint(showerror, e) |> x -> first(x, 80))
        nothing
    end
end

println()
refs = filter(!isnothing, results)
if isempty(refs)
    println("  (all backends failed — no comparison possible)")
else
    ref = last(refs)
    for ((name, _), dθ) in zip(backends, results)
        dθ === nothing && continue
        match = isapprox(dθ, ref; rtol=1e-4)
        @printf("  %-24s  agrees with FD: %s\n", name, match ? "✓" : "✗ (rtol=1e-4)")
    end
end

println("\n  Note: AutoZygote fails due to missing adjoint(::Tuple{LU}) in defaultalg_adjoint_eval (upstream LinearSolve bug).")
