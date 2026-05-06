using ValueAndGradient
using FiniteDifferences, Test  # triggers ValueAndGradientFiniteDifferencesExt
import ADTypes
using ADTypes: AbstractADType, AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake
using Test

@testset "ValueAndGradient" begin

    for T in (Float32, Float64)
        CT = Complex{T}

        @testset "pullback AutoMooncake T=$T" begin
            backend = AutoMooncake(config=nothing)

            test_pullback(x -> x^2, one(T), backend, T(3))                                        # scalar -> scalar
            test_pullback(x -> sum(x .^ 2), one(T), backend, T[1, 2, 3])                          # array -> scalar
            test_pullback(x -> x .^ 2, T[2, -1, 3], backend, T[1, 2, 3])                         # array -> array
            test_pullback(x -> x[1]^2 + x[2]^2, one(T), backend, (T(1), T(2)))                   # tuple -> scalar
            test_pullback((x, y) -> sum(x .* y), one(T), backend, T[1, 2], T[3, 4])              # multi-arg -> scalar
            test_pullback((x, y) -> x .* y, T[1, -1], backend, T[1, 2], T[3, 4])                 # multi-arg -> array
            test_pullback(x -> (x[1]^2, x[2]^2), (one(T), one(T)), backend, T[1, 2])             # array -> tuple
            test_pullback(x -> (sum(x), x .^ 2), (one(T), T[1, 1, 1]), backend, T[1, 2, 3])      # array -> mixed tuple
            test_pullback(x -> real(x * conj(x)), one(T), backend, CT(1, 2))                      # complex scalar -> real scalar
            test_pullback(x -> real(sum(x .* conj.(x))), one(T), backend, CT[CT(1,2), CT(3,4)])   # complex array -> real scalar

            @testset "ȳ scaling" begin
                f = x -> x .^ 2
                x = T[1, 2, 3]
                _, x̄1 = value_and_pullback!!(f, ones(T, 3), backend, x)
                _, x̄2 = value_and_pullback!!(f, T(2) .* ones(T, 3), backend, x)
                @test x̄2 ≈ 2 .* x̄1
            end
        end

        @testset "pushforward AutoMooncakeForward T=$T" begin
            backend = AutoMooncakeForward(config=nothing)

            test_pushforward(x -> x^2, one(T), backend, T(3))                                              # scalar -> scalar
            test_pushforward(x -> sum(x .^ 2), T[1, 0, 0], backend, T[1, 2, 3])                          # array -> scalar
            test_pushforward(x -> x .^ 2, ones(T, 3), backend, T[1, 2, 3])                               # array -> array (uniform tangent)
            test_pushforward(x -> x .^ 2, T[0, 1, -1], backend, T[1, 2, 3])                              # array -> array (non-uniform tangent)
            test_pushforward(x -> x[1]^2 + x[2]^2, (one(T), one(T)), backend, (T(1), T(2)))              # tuple -> scalar
            test_pushforward((x, y) -> sum(x .* y), (T[1, 0], T[0, 1]), backend, T[1, 2], T[3, 4])      # multi-arg -> scalar
            test_pushforward(x -> (x[1]^2, x[2]^2), T[1, 1], backend, T[1, 2])                           # array -> tuple
            test_pushforward(x -> (sum(x), x .^ 2), ones(T, 3), backend, T[1, 2, 3])                     # array -> mixed tuple
            test_pushforward(x -> real(x * conj(x)), one(CT), backend, CT(1, 2))                          # complex scalar -> real scalar
            test_pushforward(x -> real(sum(x .* conj.(x))), CT[one(CT), one(CT)], backend, CT[CT(1,2), CT(3,4)])  # complex array -> real scalar

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
            backend = AutoMooncake(config=nothing)
            cache = Mooncake.prepare_pullback_cache(f, x)

            y, x̄ = value_and_pullback!!(f, one(T), backend, x; cache)
            @test y ≈ f(x)
            @test x̄ ≈ 2 .* x

            # same result as without cache
            y2, x̄2 = value_and_pullback!!(f, one(T), backend, x)
            @test y ≈ y2
            @test x̄ ≈ x̄2

            # repeated calls consistent
            y3, x̄3 = value_and_pullback!!(f, one(T), backend, x; cache)
            @test y3 ≈ y
            @test x̄3 ≈ x̄
        end

        @testset "cached pushforward T=$T" begin
            f = x -> x .^ 2
            x = T[1, 2, 3]
            ẋ = ones(T, 3)
            backend = AutoMooncakeForward(config=nothing)
            cache = Mooncake.prepare_derivative_cache(f, x)

            y, ẏ = value_and_pushforward!!(f, ẋ, backend, x; cache)
            @test y ≈ f(x)
            @test ẏ ≈ 2 .* x

            # same result as without cache
            y2, ẏ2 = value_and_pushforward!!(f, ẋ, backend, x)
            @test y ≈ y2
            @test ẏ ≈ ẏ2

            # repeated calls consistent
            y3, ẏ3 = value_and_pushforward!!(f, ẋ, backend, x; cache)
            @test y3 ≈ y
            @test ẏ3 ≈ ẏ
        end
    end

end
