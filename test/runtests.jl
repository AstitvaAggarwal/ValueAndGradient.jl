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
                @test x̄ ≈ 2 .* x + T[1, 1, 0]
            end
            @testset "value_and_pullback!!: complex output (real → complex)" begin
                f = x -> Complex(x^2, x)
                x = T(3)
                ȳ = one(Complex{T})
                y, x̄ = value_and_pullback!!(f, ȳ, backend, x)
                @test y ≈ f(x)
                @test x̄ ≈ 2 * x
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
                @test ẏ.fields.a ≈ sum(2 .* x .* ẋ)
                @test ẏ.fields.b ≈ T(2)
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

        @testset "structured array inputs T=$T" begin
            f = x -> sum(x .^ 2)
            backend = AutoMooncake(config = nothing)
            backend_ft = AutoMooncake(config = Mooncake.Config(friendly_tangents = true))

            @testset "Symmetric friendly" begin
                x = Symmetric(T[1 2; 2 3])
                y, x̄ = value_and_pullback!!(f, one(T), backend_ft, x)
                @test y ≈ f(x)
                @test x̄ isa Matrix{T}
                @test x̄ ≈ T[2 8; 0 6]
            end
            @testset "SymTridiagonal friendly" begin
                x = SymTridiagonal(T[1, 2, 3], T[4, 5])
                y, x̄ = value_and_pullback!!(f, one(T), backend_ft, x)
                @test y ≈ f(x)
                @test x̄ isa Matrix{T}
                @test x̄ ≈ T[2 16 0; 16 4 20; 0 20 6]
            end

            @testset "Diagonal (post-norm gap)" begin
                x = Diagonal(T[1, 2, 3])
                y, x̄ = value_and_pullback!!(f, one(T), backend, x)
                @test y ≈ f(x)
                @test_broken x̄ isa typeof(x)
            end
            @testset "Hermitian (upstream bug)" begin
                x = Hermitian(Complex{T}[1 2 + im; 2 - im 3])
                f_real = x -> real(sum(x .^ 2))
                y, x̄ = value_and_pullback!!(f_real, one(T), backend, x)
                @test y ≈ f_real(x)
                @test_broken x̄ isa typeof(x)
            end
        end
    end

    @testset "error messages" begin
        @test_throws ArgumentError value_and_pullback!!(x -> x^2, 1.0, AutoMooncakeForward(config = nothing), 2.0)
        @test_throws ArgumentError value_and_pushforward!!(x -> x^2, 1.0, AutoMooncake(config = nothing), 2.0)
    end

end
