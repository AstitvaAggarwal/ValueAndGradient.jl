# Benchmark: value_and_jacobian!! — input/output type coverage + path comparison
#
# One-time setup (from repo root):
#   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#
# Run:
#   julia --project=benchmark benchmark/bench_jacobian.jl

using ValueAndGradient
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake
using LinearAlgebra
using Logging: global_logger, NullLogger
using BenchmarkTools

global_logger(NullLogger())   # suppress derived-path @warn

const rev = AutoMooncake(config = nothing)
const fwd = AutoMooncakeForward(config = nothing)

println("=" ^ 70)
println("  value_and_jacobian!! benchmarks  (BenchmarkTools @btime)")
println("=" ^ 70)

# ─── 1. Input type sweep — Float32 ───────────────────────────────────────────
#
# Float32 scalar, vector, matrix inputs.  Output type follows the input eltype.

println("\n[1] Float32 inputs")

const f32_scalar_to_vec  = x -> [x^i for i in 1:4]       # R → R⁴
const f32_vec_to_vec     = x -> x .^ 2                    # R⁴ → R⁴
const f32_mat_to_vec     = x -> vec(x .^ 2)               # R^{2×2} → R⁴
const x_f32s = 2f0
const x_f32v = randn(Float32, 4)
const x_f32m = randn(Float32, 2, 2)

println("  scalar → Vector{Float32}:")
print("    reverse: "); @btime value_and_jacobian!!($f32_scalar_to_vec, $rev, $x_f32s)
print("    forward: "); @btime value_and_jacobian!!($f32_scalar_to_vec, $fwd, $x_f32s)

println("  Vector{Float32} → Vector{Float32}:")
print("    reverse: "); @btime value_and_jacobian!!($f32_vec_to_vec, $rev, $x_f32v)
print("    forward: "); @btime value_and_jacobian!!($f32_vec_to_vec, $fwd, $x_f32v)

println("  Matrix{Float32} → Vector{Float32}:")
print("    reverse: "); @btime value_and_jacobian!!($f32_mat_to_vec, $rev, $x_f32m)
print("    forward: "); @btime value_and_jacobian!!($f32_mat_to_vec, $fwd, $x_f32m)

# ─── 2. Input type sweep — Complex{Float64} ───────────────────────────────────

println("\n[2] Complex{Float64} inputs")

const f_cvec_to_cvec = x -> x .^ 2                        # C⁴ → C⁴
const f_cmat_to_cvec = x -> vec(x .^ 2)                   # C^{2×2} → C⁴
const x_cv = randn(ComplexF64, 4)
const x_cm = randn(ComplexF64, 2, 2)

println("  Vector{ComplexF64} → Vector{ComplexF64}:")
print("    reverse: "); @btime value_and_jacobian!!($f_cvec_to_cvec, $rev, $x_cv)
print("    forward: "); @btime value_and_jacobian!!($f_cvec_to_cvec, $fwd, $x_cv)

println("  Matrix{ComplexF64} → Vector{ComplexF64}:")
print("    reverse: "); @btime value_and_jacobian!!($f_cmat_to_cvec, $rev, $x_cm)
print("    forward: "); @btime value_and_jacobian!!($f_cmat_to_cvec, $fwd, $x_cm)

# ─── 3. Output type sweep — Matrix{Float64} input, varying output shapes ──────
#
# Matrix input forces the derived path (Layer 1 only handles AbstractVector input),
# so all output types are exercised through _jacobian_via_pullback/pushforward.
#   scalar output  → hits the `y isa DiffScalar` branch in pullback
#   vector output  → standard path
#   matrix output  → _flatten(::Array) = vec(...)

println("\n[3] Output type sweep — Matrix{Float64} input (n=4), derived path only")

const x_out = randn(2, 2)
const f_to_scalar = x -> sum(x .^ 2)                      # R^{2×2} → R   (scalar output)
const f_to_vec    = x -> vec(x .^ 2)                      # R^{2×2} → R⁴  (vector output)
const f_to_mat    = x -> reshape(x .^ 2, 1, 4)            # R^{2×2} → R^{1×4} (matrix output)

println("  → Float64 scalar (y isa DiffScalar branch):")
print("    reverse: "); @btime value_and_jacobian!!($f_to_scalar, $rev, $x_out)
print("    forward: "); @btime value_and_jacobian!!($f_to_scalar, $fwd, $x_out)

println("  → Vector{Float64}:")
print("    reverse: "); @btime value_and_jacobian!!($f_to_vec, $rev, $x_out)
print("    forward: "); @btime value_and_jacobian!!($f_to_vec, $fwd, $x_out)

println("  → Matrix{Float64} (flattened by _flatten):")
print("    reverse: "); @btime value_and_jacobian!!($f_to_mat, $rev, $x_out)
print("    forward: "); @btime value_and_jacobian!!($f_to_mat, $fwd, $x_out)

# ─── 4. Forward vs Reverse — shape analysis, Float64 ─────────────────────────
#
# Wide (m < n) → reverse wins.  Tall (m > n) → forward wins.

println("\n[4] Forward vs Reverse shape analysis — Float64")

const f_sq   = x -> vec(x .^ 2)
const x_sq   = randn(3, 3)          # n=9, m=9
const f_wide = x -> [sum(x[i, :]) for i in 1:4]
const x_wide = randn(4, 4)          # n=16, m=4
const f_tall = x -> repeat(x, 4)
const x_tall = randn(4)             # n=4,  m=16

println("  Square n=m=9:")
print("    reverse (9 pullbacks):    "); @btime value_and_jacobian!!($f_sq, $rev, $x_sq)
print("    forward (9 pushforwards): "); @btime value_and_jacobian!!($f_sq, $fwd, $x_sq)

println("  Wide n=16, m=4 (reverse should win ~4×):")
print("    reverse (4 pullbacks):    "); @btime value_and_jacobian!!($f_wide, $rev, $x_wide)
print("    forward (16 pushforwards):"); @btime value_and_jacobian!!($f_wide, $fwd, $x_wide)

println("  Tall n=4, m=16 (forward should win ~4×):")
print("    reverse (16 pullbacks):   "); @btime value_and_jacobian!!($f_tall, $rev, $x_tall)
print("    forward (4 pushforwards): "); @btime value_and_jacobian!!($f_tall, $fwd, $x_tall)

# ─── 5. Native (Layer 1) vs Derived (Layer 2) ─────────────────────────────────

println("\n[5] Native (Layer 1) vs Derived (Layer 2) — Vector{Float64}, n=m=10")

const f_native = x -> x .^ 2
const x_native = randn(10)

print("  reverse native  (Layer 1): "); @btime value_and_jacobian!!($f_native, $rev, $x_native)
print("  forward native  (Layer 1): "); @btime value_and_jacobian!!($f_native, $fwd, $x_native)
print("  reverse derived (Layer 2): "); @btime ValueAndGradient._jacobian_via_pullback($f_native, $rev, $x_native)
print("  forward derived (Layer 2): "); @btime ValueAndGradient._jacobian_via_pushforward($f_native, $fwd, $x_native)

# ─── 6. Cached vs uncached ────────────────────────────────────────────────────

println("\n[6] Cached vs uncached — Matrix{Float64} derived path (n=m=16)")

const f_cache = x -> vec(x .^ 2)
const x_cache = randn(4, 4)
const cache_rev = ValueAndGradient._prepare_jac_cache_reverse(f_cache, rev, (x_cache,))
const cache_fwd = ValueAndGradient._prepare_jac_cache_forward(f_cache, fwd, (x_cache,))

print("  reverse uncached: "); @btime value_and_jacobian!!($f_cache, $rev, $x_cache)
print("  reverse cached:   "); @btime value_and_jacobian!!($f_cache, $rev, $x_cache; ad_cache = $cache_rev)
print("  forward uncached: "); @btime value_and_jacobian!!($f_cache, $fwd, $x_cache)
print("  forward cached:   "); @btime value_and_jacobian!!($f_cache, $fwd, $x_cache; ad_cache = $cache_fwd)

# ─── 7. Multi-arg inputs ──────────────────────────────────────────────────────

println("\n[7] Multi-arg inputs")

const f_ma_same  = (x, y) -> x .* y
const x_ma, y_ma = randn(4), randn(4)            # (R⁴, R⁴) → R⁴

const f_ma_mixed = (x, y) -> vec(x) .* y
const x_ma2, y_ma2 = randn(2, 2), randn(4)       # (R^{2×2}, R⁴) → R⁴

println("  (Vector{Float64}, Vector{Float64}) → Vector{Float64}:")
print("    reverse: "); @btime value_and_jacobian!!($f_ma_same, $rev, $x_ma, $y_ma)
print("    forward: "); @btime value_and_jacobian!!($f_ma_same, $fwd, $x_ma, $y_ma)

println("  (Matrix{Float64}, Vector{Float64}) → Vector{Float64}:")
print("    reverse: "); @btime value_and_jacobian!!($f_ma_mixed, $rev, $x_ma2, $y_ma2)
print("    forward: "); @btime value_and_jacobian!!($f_ma_mixed, $fwd, $x_ma2, $y_ma2)

println()
