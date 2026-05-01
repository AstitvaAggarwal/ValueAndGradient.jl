using ValueAndGradient
using ValueAndGradient.TestUtils
import ADTypes
using ADTypes: AbstractADType, AutoMooncake, AutoMooncakeForward
using Mooncake
using Test

@testset "ValueAndGradient" begin

    for T in (Float32, Float64)
        CT = Complex{T}

        @testset "pullback AutoMooncake T=$T" begin
            backend = AutoMooncake(config=nothing)

            # scalar
            test_pullback(x -> x^2, one(T), backend, T(3))
            # array -> scalar
            test_pullback(x -> sum(x .^ 2), one(T), backend, T[1, 2, 3])
            # array -> array
            test_pullback(x -> x .^ 2, T[2, -1, 3], backend, T[1, 2, 3])
            # tuple input
            test_pullback(x -> x[1]^2 + x[2]^2, one(T), backend, (T(1), T(2)))
            # multiple args
            test_pullback((x, y) -> sum(x .* y), one(T), backend, T[1, 2], T[3, 4])
            # multiple args, array output
            test_pullback((x, y) -> x .* y, T[1, -1], backend, T[1, 2], T[3, 4])
            # tuple output
            test_pullback(x -> (x[1]^2, x[2]^2), (one(T), one(T)), backend, T[1, 2])
            test_pullback(x -> (sum(x), x .^ 2), (one(T), T[1, 1, 1]), backend, T[1, 2, 3])
            # complex scalar
            test_pullback(x -> real(x * conj(x)), one(T), backend, CT(1, 2))
            # complex array
            test_pullback(x -> real(sum(x .* conj.(x))), one(T), backend, CT[CT(1,2), CT(3,4)])

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

            # scalar
            test_pushforward(x -> x^2, one(T), backend, T(3))
            # array -> scalar
            test_pushforward(x -> sum(x .^ 2), T[1, 0, 0], backend, T[1, 2, 3])
            # array -> array
            test_pushforward(x -> x .^ 2, ones(T, 3), backend, T[1, 2, 3])
            test_pushforward(x -> x .^ 2, T[0, 1, -1], backend, T[1, 2, 3])
            # tuple input
            test_pushforward(x -> x[1]^2 + x[2]^2, (one(T), one(T)), backend, (T(1), T(2)))
            # multiple args
            test_pushforward((x, y) -> sum(x .* y), (T[1, 0], T[0, 1]), backend, T[1, 2], T[3, 4])
            # tuple output
            test_pushforward(x -> (x[1]^2, x[2]^2), T[1, 1], backend, T[1, 2])
            test_pushforward(x -> (sum(x), x .^ 2), ones(T, 3), backend, T[1, 2, 3])
            # complex scalar
            test_pushforward(x -> real(x * conj(x)), one(CT), backend, CT(1, 2))
            # complex array
            test_pushforward(x -> real(sum(x .* conj.(x))), CT[one(CT), one(CT)], backend, CT[CT(1,2), CT(3,4)])

            @testset "ẋ scaling" begin
                f = x -> x .^ 2
                x = T[1, 2, 3]
                ẋ = ones(T, 3)
                _, ẏ1 = value_and_pushforward!!(f, ẋ, backend, x)
                _, ẏ2 = value_and_pushforward!!(f, T(2) .* ẋ, backend, x)
                @test ẏ2 ≈ 2 .* ẏ1
            end
        end
    end

end
