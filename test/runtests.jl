using ValueAndGradient
using FiniteDifferences, Test
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake
using LinearAlgebra

struct VGOutput{T}
    a::T; b::T
end

@testset "ValueAndGradient" begin

    for T in (Float32, Float64)
        CT = Complex{T}

        @testset "pullback AutoMooncake T=$T" begin
            backend = AutoMooncake(config = nothing)

            test_pullback(x -> x^2, one(T), backend, T(3))
            test_pullback(x -> sum(x .^ 2), one(T), backend, T[1, 2, 3])
            test_pullback(x -> x .^ 2, T[2, -1, 3], backend, T[1, 2, 3])
            test_pullback(x -> x[1]^2 + x[2]^2, one(T), backend, (T(1), T(2)))
            test_pullback((x, y) -> sum(x .* y), one(T), backend, T[1, 2], T[3, 4])
            test_pullback((x, y) -> x .* y, T[1, -1], backend, T[1, 2], T[3, 4])
            test_pullback(x -> (x[1]^2, x[2]^2), (one(T), one(T)), backend, T[1, 2])
            test_pullback(x -> (sum(x), x .^ 2), (one(T), T[1, 1, 1]), backend, T[1, 2, 3])
            test_pullback(x -> real(x * conj(x)), one(T), backend, CT(1, 2))
            test_pullback(x -> real(sum(x .* conj.(x))), one(T), backend, CT[CT(1, 2), CT(3, 4)])
            test_pullback(x -> (a = sum(x .^ 2), b = x[1] + x[2]), (a = one(T), b = one(T)), backend, T[1, 2, 3])
            test_pullback((x, y) -> (a = sum(x .* y), b = x[1] + y[1]), (a = one(T), b = one(T)), backend, T[1, 2], T[3, 4])
            # struct output: ȳ must be Mooncake.Tangent — VG.jl passes through, no output constraint
            @testset "value_and_pullback!!: struct output (array → VGOutput)" begin
                f = x -> VGOutput(sum(x .^ 2), x[1] + x[2])
                x = T[1, 2, 3]
                ȳ = Mooncake.Tangent{NamedTuple{(:a, :b), Tuple{T, T}}}((a = one(T), b = one(T)))
                y, x̄ = value_and_pullback!!(f, ȳ, backend, x)
                @test y isa VGOutput{T}
                @test x̄ ≈ 2 .* x + T[1, 1, 0]   # ∂sum(x²)/∂x + ∂(x₁+x₂)/∂x = [3,5,6]
            end
            # test_pullback not usable here: its grad() helper requires a real-valued function
            @testset "value_and_pullback!!: complex output (real → complex)" begin
                f = x -> Complex(x^2, x)
                x = T(3)
                ȳ = one(Complex{T})
                y, x̄ = value_and_pullback!!(f, ȳ, backend, x)
                @test y ≈ f(x)
                @test x̄ ≈ 2 * x    # Re(conj(ȳ) * (2x + im)) = 2x at x=3, ȳ=1+0im
            end

            @testset "ȳ scaling" begin
                f = x -> x .^ 2
                x = T[1, 2, 3]
                _, x̄1 = value_and_pullback!!(f, ones(T, 3), backend, x)
                _, x̄2 = value_and_pullback!!(f, T(2) .* ones(T, 3), backend, x)
                @test x̄2 ≈ 2 .* x̄1
            end
        end

        @testset "pushforward AutoMooncakeForward T=$T" begin
            backend = AutoMooncakeForward(config = nothing)

            test_pushforward(x -> x^2, one(T), backend, T(3))
            test_pushforward(x -> sum(x .^ 2), T[1, 0, 0], backend, T[1, 2, 3])
            test_pushforward(x -> x .^ 2, T[0, 1, -1], backend, T[1, 2, 3])
            test_pushforward(x -> x[1]^2 + x[2]^2, (one(T), one(T)), backend, (T(1), T(2)))
            test_pushforward((x, y) -> sum(x .* y), (T[1, 0], T[0, 1]), backend, T[1, 2], T[3, 4])
            test_pushforward(x -> (x[1]^2, x[2]^2), T[1, 1], backend, T[1, 2])
            test_pushforward(x -> (sum(x), x .^ 2), ones(T, 3), backend, T[1, 2, 3])
            test_pushforward(x -> real(x * conj(x)), one(CT), backend, CT(1, 2))
            test_pushforward(x -> real(sum(x .* conj.(x))), CT[one(CT), one(CT)], backend, CT[CT(1, 2), CT(3, 4)])
            test_pushforward(x -> Complex(x^2, x), one(T), backend, T(3))
            test_pushforward(x -> (a = sum(x .^ 2), b = x[1] + x[2]), ones(T, 3), backend, T[1, 2, 3])
            test_pushforward((x, y) -> (a = sum(x .* y), b = x[1] + y[1]), (T[1, 0], T[0, 1]), backend, T[1, 2], T[3, 4])
            @testset "array → struct (VGOutput)" begin
                f = x -> VGOutput(sum(x .^ 2), x[1] + x[2])
                x = T[1, 2, 3]
                ẋ = ones(T, 3)
                y, ẏ = value_and_pushforward!!(f, ẋ, backend, x)
                @test y isa VGOutput{T}
                @test ẏ.fields.a ≈ sum(2 .* x .* ẋ)   # d(sum(x²))/dx · ẋ = 2x·ẋ
                @test ẏ.fields.b ≈ T(2)                  # d(x₁+x₂)/dx · [1,1,1] = 2
            end

            @testset "ẋ scaling" begin
                f = x -> x .^ 2
                x = T[1, 2, 3]
                ẋ = ones(T, 3)
                _, ẏ1 = value_and_pushforward!!(f, ẋ, backend, x)
                _, ẏ2 = value_and_pushforward!!(f, T(2) .* ẋ, backend, x)
                @test ẏ2 ≈ 2 .* ẏ1
            end
        end

        @testset "cached gradient T=$T" begin
            f = x -> sum(x .^ 2)
            x = T[1, 2, 3]
            backend = AutoMooncake(config = nothing)
            cache = Mooncake.prepare_gradient_cache(f, x)

            y, ∇f = value_and_gradient!!(f, backend, x; ad_cache = cache)
            @test y ≈ f(x)
            @test ∇f ≈ 2 .* x

            y2, ∇f2 = value_and_gradient!!(f, backend, x)
            @test ∇f ≈ ∇f2

            y3, ∇f3 = value_and_gradient!!(f, backend, x; ad_cache = cache)
            @test ∇f3 ≈ ∇f
        end

        @testset "cached pullback T=$T" begin
            f = x -> sum(x .^ 2)
            x = T[1, 2, 3]
            backend = AutoMooncake(config = nothing)
            cache = Mooncake.prepare_pullback_cache(f, x)

            y, x̄ = value_and_pullback!!(f, one(T), backend, x; ad_cache = cache)
            @test y ≈ f(x)
            @test x̄ ≈ 2 .* x

            y2, x̄2 = value_and_pullback!!(f, one(T), backend, x)
            @test y ≈ y2
            @test x̄ ≈ x̄2

            y3, x̄3 = value_and_pullback!!(f, one(T), backend, x; ad_cache = cache)
            @test y3 ≈ y
            @test x̄3 ≈ x̄
        end

        @testset "gradient AutoMooncake T=$T" begin
            backend = AutoMooncake(config = nothing)

            test_gradient(x -> x^2, backend, T(3))
            test_gradient(x -> sum(x .^ 2), backend, T[1, 2, 3])
            test_gradient(x -> x[1]^2 + x[2]^2, backend, (T(1), T(2)))
            test_gradient((x, y) -> sum(x .* y), backend, T[1, 2], T[3, 4])
            test_gradient(x -> real(x * conj(x)), backend, CT(1, 2))
            test_gradient(x -> real(sum(x .* conj.(x))), backend, CT[CT(1, 2), CT(3, 4)])
        end

        @testset "jacobian AutoMooncake T=$T" begin
            backend = AutoMooncake(config = nothing)

            test_jacobian(x -> x .^ 2, backend, T[1, 2, 3])
            test_jacobian(x -> [x[1]^2 + x[2], x[2]^2 - x[1]], backend, T[2, 3])
            test_jacobian(x -> [x[1]^2, x[2]^2, x[1] * x[2]], backend, T[2, 3])
        end

        @testset "jacobian AutoMooncakeForward T=$T" begin
            backend = AutoMooncakeForward(config = nothing)

            test_jacobian(x -> x .^ 2, backend, T[1, 2, 3])
            test_jacobian(x -> [x[1]^2 + x[2], x[2]^2 - x[1]], backend, T[2, 3])
            test_jacobian(x -> [x[1]^2, x[2]^2, x[1] * x[2]], backend, T[2, 3])
        end

        @testset "cached jacobian T=$T" begin
            f = x -> x .^ 2
            x = T[1, 2, 3]

            for (backend, cache) in [
                    (AutoMooncake(config = nothing), Mooncake.prepare_pullback_cache(f, x)),
                    (AutoMooncakeForward(config = nothing), Mooncake.prepare_derivative_cache(f, x)),
                ]
                y, (J,) = value_and_jacobian!!(f, backend, x; ad_cache = cache)
                @test y ≈ f(x)
                @test J ≈ diagm(2 .* x)         # ∂(xᵢ²)/∂xⱼ = 2xᵢ δᵢⱼ

                y2, (J2,) = value_and_jacobian!!(f, backend, x)
                @test J ≈ J2

                y3, (J3,) = value_and_jacobian!!(f, backend, x; ad_cache = cache)
                @test J3 ≈ J
            end
        end

        @testset "structured array inputs T=$T" begin
            f = x -> sum(x .^ 2)
            backend = AutoMooncake(config = nothing)
            backend_ft = AutoMooncake(config = Mooncake.Config(friendly_tangents = true))

            # Symmetric and SymTridiagonal: Mooncake returns Matrix{T} with friendly_tangents=true
            # (PR #1103). Value and tangent type both correct.
            @testset "Symmetric friendly" begin
                x = Symmetric(T[1 2; 2 3])
                y, x̄ = value_and_pullback!!(f, one(T), backend_ft, x)
                @test y ≈ f(x)
                @test x̄ isa Matrix{T}
                # Mooncake differentiates w.r.t. stored upper triangle:
                # diagonal gets 2x, off-diagonal gets 4x (appears in both x[i,j] and x[j,i]), lower is 0
                @test x̄ ≈ T[2 8; 0 6]
            end
            @testset "SymTridiagonal friendly" begin
                x = SymTridiagonal(T[1, 2, 3], T[4, 5])
                y, x̄ = value_and_pullback!!(f, one(T), backend_ft, x)
                @test y ≈ f(x)
                @test x̄ isa Matrix{T}
                # SymTridiagonal friendly tangent is a full symmetric matrix:
                # diagonal 2*dv, off-diagonal 4*ev placed symmetrically (both sides)
                @test x̄ ≈ T[2 16 0; 16 4 20; 0 20 6]
            end

            # Representative broken cases: post-norm gap (raw Mooncake.Tangent returned).
            # Diagonal stands in for all structured arrays without friendly_tangents support.
            @testset "Diagonal (post-norm gap)" begin
                x = Diagonal(T[1, 2, 3])
                y, x̄ = value_and_pullback!!(f, one(T), backend, x)
                @test y ≈ f(x)
                @test_broken x̄ isa typeof(x)
            end
            # Hermitian: friendly_tangents=true errors upstream; default returns raw Mooncake.Tangent.
            @testset "Hermitian (upstream bug)" begin
                x = Hermitian(Complex{T}[1 2 + im; 2 - im 3])
                f_real = x -> real(sum(x .^ 2))
                y, x̄ = value_and_pullback!!(f_real, one(T), backend, x)
                @test y ≈ f_real(x)
                @test_broken x̄ isa typeof(x)
            end
        end

        @testset "derivative AutoMooncakeForward T=$T" begin
            backend = AutoMooncakeForward(config = nothing)
            test_derivative(x -> x^2, backend, T(3))
            test_derivative(x -> [x^2, x^3, x], backend, T(3))
            test_derivative(x -> Complex(x^2, x), backend, T(3))
            test_derivative(x -> real(x * conj(x)), backend, CT(1, 2))
            test_derivative(x -> x^2, backend, CT(1, 2))

            @testset "R→Tuple" begin
                f = x -> (x^2, x^3)
                x = T(3)
                y, ẏ = value_and_derivative!!(f, backend, x)
                @test y == f(x)
                @test ẏ[1] ≈ 2 * x
                @test ẏ[2] ≈ 3 * x^2
            end

            @testset "R→NamedTuple" begin
                f = x -> (a = x^2, b = x^3)
                x = T(3)
                y, ẏ = value_and_derivative!!(f, backend, x)
                @test y == f(x)
                @test ẏ.a ≈ 2 * x
                @test ẏ.b ≈ 3 * x^2
            end

            @testset "R→struct (VGOutput)" begin
                f = x -> VGOutput(x^2, x^3)
                x = T(3)
                y, ẏ = value_and_derivative!!(f, backend, x)
                @test y isa VGOutput{T}
                @test ẏ.fields.a ≈ 2 * x
                @test ẏ.fields.b ≈ 3 * x^2
            end

            test_derivative(x -> [real(x * conj(x)), imag(x)^2], backend, CT(1, 2))
            @testset "C→Tuple" begin
                f = x -> (real(x * conj(x)), x^2)
                x = CT(1, 2)
                y, ẏ = value_and_derivative!!(f, backend, x)
                @test y == f(x)
                @test ẏ[1] ≈ 2 * real(x)   # ∂|x|²/∂x_Re in dir 1+0im = 2*Re(x)
                @test ẏ[2] ≈ 2 * x          # d(x²)/dx in dir 1+0im = 2x
            end
        end

        @testset "cached derivative T=$T" begin
            f = x -> x^2
            x = T(3)
            backend = AutoMooncakeForward(config = nothing)
            cache = Mooncake.prepare_derivative_cache(f, x)

            y, ẏ = value_and_derivative!!(f, backend, x; ad_cache = cache)
            @test y ≈ f(x)
            @test ẏ ≈ 2 * x

            y2, ẏ2 = value_and_derivative!!(f, backend, x)
            @test ẏ ≈ ẏ2

            y3, ẏ3 = value_and_derivative!!(f, backend, x; ad_cache = cache)
            @test ẏ3 ≈ ẏ
        end

        @testset "cached pushforward T=$T" begin
            f = x -> x .^ 2
            x = T[1, 2, 3]
            ẋ = ones(T, 3)
            backend = AutoMooncakeForward(config = nothing)
            cache = Mooncake.prepare_derivative_cache(f, x)

            y, ẏ = value_and_pushforward!!(f, ẋ, backend, x; ad_cache = cache)
            @test y ≈ f(x)
            @test ẏ ≈ 2 .* x

            y2, ẏ2 = value_and_pushforward!!(f, ẋ, backend, x)
            @test y ≈ y2
            @test ẏ ≈ ẏ2

            y3, ẏ3 = value_and_pushforward!!(f, ẋ, backend, x; ad_cache = cache)
            @test y3 ≈ y
            @test ẏ3 ≈ ẏ
        end
    end

    @testset "jacobian derived path" begin

        @testset "native path still works" begin
            test_jacobian(x -> x .^ 2, AutoMooncakeForward(config = nothing), Float64[1, 2, 3])
            test_jacobian(x -> x .^ 2, AutoMooncake(config = nothing), Float64[1, 2, 3])
        end

        @testset "matrix input AutoMooncake (reverse derived)" begin
            f = x -> vec(x .^ 2)
            backend = AutoMooncake(config = nothing)
            x = Float64[1 2; 3 4]
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (J,) = value_and_jacobian!!(f, backend, x)
                xv = vec(x)           # [1,3,2,4] (column-major)
                @test y ≈ xv .^ 2
                @test size(J) == (4, 4)
                @test J ≈ diagm(2 .* xv)
            end
        end

        @testset "matrix input AutoMooncakeForward (forward derived)" begin
            f = x -> vec(x .^ 2)
            backend = AutoMooncakeForward(config = nothing)
            x = Float64[1 2; 3 4]
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (J,) = value_and_jacobian!!(f, backend, x)
                xv = vec(x)
                @test y ≈ xv .^ 2
                @test size(J) == (4, 4)
                @test J ≈ diagm(2 .* xv)
            end
        end

        @testset "complex vector input AutoMooncakeForward (forward derived)" begin
            # f: ℂ² → ℝ², columns of J are JVPs in standard complex basis directions
            f = x -> [real(sum(x .* conj.(x))), imag(x[1])^2]
            backend = AutoMooncakeForward(config = nothing)
            x = ComplexF64[1 + 2im, 3 + 4im]
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (J,) = value_and_jacobian!!(f, backend, x)
                @test y ≈ [real(sum(x .* conj.(x))), imag(x[1])^2]
                @test size(J) == (2, 2)
                # Column 1: tangent ẋ = [1+0im, 0+0im]
                # ∂(|x₁|²+|x₂|²)/∂x₁ in dir 1+0im = 2*Re(x₁) = 2*1 = 2
                # ∂(Im(x₁)²)/∂x₁ in dir 1+0im = 2*Im(x₁)*0 = 0 (real part of dir is 0 for Im)
                # Column 2: tangent ẋ = [0+0im, 1+0im]
                # ∂(|x₁|²+|x₂|²)/∂x₂ in dir 1+0im = 2*Re(x₂) = 2*3 = 6
                # ∂(Im(x₁)²)/∂x₂ in dir 1+0im = 0
                @test real(J[1, 1]) ≈ 2 * real(x[1])  # 2.0
                @test real(J[1, 2]) ≈ 2 * real(x[2])  # 6.0
            end
        end

        @testset "multi-arg AutoMooncake (reverse derived)" begin
            f = (x, y) -> x .* y
            backend = AutoMooncake(config = nothing)
            x1 = Float64[1.0, 2.0]
            x2 = Float64[3.0, 4.0]
            @test_logs (:warn, r"derived path") match_mode = :any begin
                result_y, Js = value_and_jacobian!!(f, backend, x1, x2)
                J1, J2 = Js
                @test result_y ≈ x1 .* x2
                @test size(J1) == (2, 2)
                @test size(J2) == (2, 2)
                # J1[i,j] = ∂(xᵢyᵢ)/∂xⱼ = yᵢ δᵢⱼ
                @test J1 ≈ diagm(x2)
                # J2[i,j] = ∂(xᵢyᵢ)/∂yⱼ = xᵢ δᵢⱼ
                @test J2 ≈ diagm(x1)
            end
        end

        @testset "multi-arg AutoMooncakeForward (forward derived)" begin
            f = (x, y) -> x .* y
            backend = AutoMooncakeForward(config = nothing)
            x1 = Float64[1.0, 2.0]
            x2 = Float64[3.0, 4.0]
            @test_logs (:warn, r"derived path") match_mode = :any begin
                result_y, Js = value_and_jacobian!!(f, backend, x1, x2)
                J1, J2 = Js
                @test result_y ≈ x1 .* x2
                @test size(J1) == (2, 2)
                @test size(J2) == (2, 2)
                @test J1 ≈ diagm(x2)
                @test J2 ≈ diagm(x1)
            end
        end

        @testset "tuple input AutoMooncake (reverse derived)" begin
            # f: (ℝ², ℝ²) as a single Tuple → ℝ²; J = [I | I] (2×4)
            f = t -> t[1] .+ t[2]
            v1 = Float64[2.0, 3.0]
            v2 = Float64[4.0, 5.0]
            backend = AutoMooncake(config = nothing)
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (J,) = value_and_jacobian!!(f, backend, (v1, v2))
                @test y ≈ v1 .+ v2
                @test size(J) == (2, 4)
                @test J ≈ [1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0]
            end
        end

        @testset "tuple input AutoMooncakeForward (forward derived)" begin
            f = t -> t[1] .+ t[2]
            v1 = Float64[2.0, 3.0]
            v2 = Float64[4.0, 5.0]
            backend = AutoMooncakeForward(config = nothing)
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (J,) = value_and_jacobian!!(f, backend, (v1, v2))
                @test y ≈ v1 .+ v2
                @test size(J) == (2, 4)
                @test J ≈ [1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0]
            end
        end

        @testset "tuple with scalar+array AutoMooncake (reverse derived)" begin
            # f((s, v)) = s * v; s::Float64, v::ℝ²
            # J (2×3): [∂f/∂s | ∂f/∂v] = [v | s*I]
            f = t -> t[1] .* t[2]
            s = 2.0
            v = Float64[3.0, 4.0]
            backend = AutoMooncake(config = nothing)
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (J,) = value_and_jacobian!!(f, backend, (s, v))
                @test y ≈ s .* v
                @test size(J) == (2, 3)
                # col 1 (∂/∂s): v; cols 2-3 (∂/∂v): s*I
                @test J ≈ [v[1] s 0.0; v[2] 0.0 s]
            end
        end

        @testset "tuple with scalar+array AutoMooncakeForward (forward derived)" begin
            f = t -> t[1] .* t[2]
            s = 2.0
            v = Float64[3.0, 4.0]
            backend = AutoMooncakeForward(config = nothing)
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (J,) = value_and_jacobian!!(f, backend, (s, v))
                @test y ≈ s .* v
                @test size(J) == (2, 3)
                @test J ≈ [v[1] s 0.0; v[2] 0.0 s]
            end
        end

        @testset "multi-arg scalar+array AutoMooncake (reverse derived)" begin
            # f(s, v) = s .* v; Js is (m×1), Jv is (m×m)
            f = (s, v) -> s .* v
            s = 2.0
            v = Float64[3.0, 4.0]
            backend = AutoMooncake(config = nothing)
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (Js, Jv) = value_and_jacobian!!(f, backend, s, v)
                @test y ≈ s .* v
                @test size(Js) == (2, 1)
                @test size(Jv) == (2, 2)
                @test Js ≈ reshape(v, 2, 1)   # ∂(s*v)/∂s = v
                @test Jv ≈ s .* I(2)          # ∂(s*v)/∂v = s*I
            end
        end

        @testset "multi-arg scalar+array AutoMooncakeForward (forward derived)" begin
            f = (s, v) -> s .* v
            s = 2.0
            v = Float64[3.0, 4.0]
            backend = AutoMooncakeForward(config = nothing)
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (Js, Jv) = value_and_jacobian!!(f, backend, s, v)
                @test y ≈ s .* v
                @test size(Js) == (2, 1)
                @test size(Jv) == (2, 2)
                @test Js ≈ reshape(v, 2, 1)
                @test Jv ≈ s .* I(2)
            end
        end

        @testset "multi-tuple-arg AutoMooncake (reverse derived)" begin
            # f((x,a), (y,b)) = x .* y .+ a*b  where x,y::Vector, a,b::Float64
            # J1 = ∂f/∂(x,a) flattened: [diag(y) | b·1] shape (2,3)
            # J2 = ∂f/∂(y,b) flattened: [diag(x) | a·1] shape (2,3)
            f = ((x, a), (y, b)) -> x .* y .+ a * b
            t1 = (Float64[1.0, 2.0], 3.0)
            t2 = (Float64[4.0, 5.0], 6.0)
            backend = AutoMooncake(config = nothing)
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (J1, J2) = value_and_jacobian!!(f, backend, t1, t2)
                @test y ≈ t1[1] .* t2[1] .+ t1[2] * t2[2]
                @test size(J1) == (2, 3)
                @test size(J2) == (2, 3)
                @test J1 ≈ [4.0 0.0 6.0; 0.0 5.0 6.0]
                @test J2 ≈ [1.0 0.0 3.0; 0.0 2.0 3.0]
            end
        end

        @testset "multi-tuple-arg AutoMooncakeForward (forward derived)" begin
            f = ((x, a), (y, b)) -> x .* y .+ a * b
            t1 = (Float64[1.0, 2.0], 3.0)
            t2 = (Float64[4.0, 5.0], 6.0)
            backend = AutoMooncakeForward(config = nothing)
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y, (J1, J2) = value_and_jacobian!!(f, backend, t1, t2)
                @test y ≈ t1[1] .* t2[1] .+ t1[2] * t2[2]
                @test size(J1) == (2, 3)
                @test size(J2) == (2, 3)
                @test J1 ≈ [4.0 0.0 6.0; 0.0 5.0 6.0]
                @test J2 ≈ [1.0 0.0 3.0; 0.0 2.0 3.0]
            end
        end

        @testset "scalar DiffScalar input (derived)" begin
            for T in (Float32, Float64)
                f_v = x -> [x^2, x^3]
                f_s = x -> x^2
                for backend in (AutoMooncake(config = nothing), AutoMooncakeForward(config = nothing))
                    @test_logs (:warn, r"derived path") match_mode = :any begin
                        y, (J,) = value_and_jacobian!!(f_v, backend, T(2))
                        @test y ≈ T[4, 8]
                        @test size(J) == (2, 1)
                        @test J ≈ T[4; 12;;]        # [2x; 3x²] at x=2
                    end
                    @test_logs (:warn, r"derived path") match_mode = :any begin
                        y, (J,) = value_and_jacobian!!(f_s, backend, T(3))
                        @test y ≈ T(9)
                        @test size(J) == (1, 1)
                        @test J[1, 1] ≈ T(6)        # 2x at x=3
                    end
                end
            end
        end

        @testset "matrix→scalar output (scalar-y branch, derived)" begin
            f = x -> sum(x .^ 2)
            for T in (Float32, Float64)
                x = T[1 2; 3 4]
                expected = reshape(T(2) .* vec(x), 1, 4)   # 1×4: [2,6,4,8] col-major
                for backend in (AutoMooncake(config = nothing), AutoMooncakeForward(config = nothing))
                    @test_logs (:warn, r"derived path") match_mode = :any begin
                        y, (J,) = value_and_jacobian!!(f, backend, x)
                        @test y ≈ f(x)
                        @test size(J) == (1, 4)
                        @test J ≈ expected
                    end
                end
            end
        end

        @testset "Float32 matrix derived path" begin
            f = x -> vec(x .^ 2)
            x = Float32[1 2; 3 4]
            for backend in (AutoMooncake(config = nothing), AutoMooncakeForward(config = nothing))
                @test_logs (:warn, r"derived path") match_mode = :any begin
                    y, (J,) = value_and_jacobian!!(f, backend, x)
                    @test y ≈ vec(x .^ 2)
                    @test J ≈ diagm(2f0 .* vec(x))
                end
            end
        end

        @testset "complex vector input AutoMooncake (reverse derived)" begin
            f = x -> [real(sum(x .* conj.(x))), imag(x[1])^2]
            x = ComplexF64[1 + 2im, 3 + 4im]
            @test_logs (:warn, r"derived path") match_mode = :any begin
                y_rev, (J_rev,) = value_and_jacobian!!(f, AutoMooncake(config = nothing), x)
                _, (J_fwd,) = value_and_jacobian!!(f, AutoMooncakeForward(config = nothing), x)
                @test y_rev ≈ f(x)
                @test size(J_rev) == (2, 2)
                @test real.(J_rev) ≈ real.(J_fwd)   # both directions agree on real part
            end
        end

        @testset "complex matrix input (derived)" begin
            f = x -> [real(sum(x .* conj.(x))), imag(x[1, 1])^2]
            x = ComplexF64[1+2im 3+4im; 5+6im 7+8im]
            for backend in (AutoMooncake(config = nothing), AutoMooncakeForward(config = nothing))
                @test_logs (:warn, r"derived path") match_mode = :any begin
                    y, (J,) = value_and_jacobian!!(f, backend, x)
                    @test y ≈ f(x)
                    @test size(J) == (2, 4)
                end
            end
        end

        @testset "cached jacobian derived path (matrix input)" begin
            f = x -> vec(x .^ 2)
            x = Float64[1.0 2.0; 3.0 4.0]
            for (backend, cache) in [
                    (AutoMooncake(config = nothing), Mooncake.prepare_pullback_cache(f, x)),
                    (AutoMooncakeForward(config = nothing), Mooncake.prepare_derivative_cache(f, x)),
                ]
                @test_logs (:warn, r"derived path") match_mode = :any begin
                    y, (J,) = value_and_jacobian!!(f, backend, x; ad_cache = cache)
                    @test y ≈ f(x)
                    @test J ≈ diagm(2 .* vec(x))
                end
            end
        end
    end

    @testset "error messages" begin
        @test_throws ArgumentError value_and_derivative!!(x -> x^2, AutoMooncake(config = nothing), 2.0)
        # complex arrays now supported via derived path (Layer 2) — no longer an error
        @test_throws ArgumentError value_and_pullback!!(x -> x^2, 1.0, AutoMooncakeForward(config = nothing), 2.0)
        @test_throws ArgumentError value_and_pushforward!!(x -> x^2, 1.0, AutoMooncake(config = nothing), 2.0)
        @test_throws ArgumentError value_and_gradient!!(x -> x^2, AutoMooncakeForward(config = nothing), 2.0)
        @test_logs (:warn, r"derived path") match_mode = :any begin
            y, (J,) = value_and_jacobian!!(x -> x .^ 2, AutoMooncake(config = nothing), 2.0)
            @test y ≈ 4.0 && size(J) == (1, 1) && J[1, 1] ≈ 4.0
        end
    end

end
