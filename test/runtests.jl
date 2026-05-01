using ValueAndGradient
using ValueAndGradient.TestUtils
import ADTypes
using ADTypes: AbstractADType, AutoMooncake, AutoMooncakeForward
using Mooncake
using Test

@testset "ValueAndGradient" begin

    @testset "Mooncake reverse-mode (pullback)" begin
        backend = AutoMooncake(config=nothing)

        test_pullback(x -> x^2, 1.0, backend, 3.0)
        test_pullback(x -> sum(x .^ 2), 1.0, backend, [1.0, 2.0, 3.0])
        test_pullback(x -> x .^ 2, [2.0, -1.0, 3.0], backend, [1.0, 2.0, 3.0])
        test_pullback(x -> x[1]^2 + x[2]^2, 1.0, backend, (1.0, 2.0))
        test_pullback((x, y) -> sum(x .* y), 1.0, backend, [1.0, 2.0], [3.0, 4.0])
        test_pullback((x, y) -> x .* y, [1.0, -1.0], backend, [1.0, 2.0], [3.0, 4.0])
        test_pullback(x -> real(x * conj(x)), 1.0, backend, 1.0 + 2.0im)
        test_pullback(x -> real(sum(x .* conj.(x))), 1.0, backend, [1.0+2.0im, 3.0+4.0im])

        @testset "ȳ scaling" begin
            f = x -> x .^ 2
            x = [1.0, 2.0, 3.0]
            _, x̄1 = value_and_pullback!!(f, ones(3), backend, x)
            _, x̄2 = value_and_pullback!!(f, 2 .* ones(3), backend, x)
            @test x̄2 ≈ 2 .* x̄1
        end
    end

    @testset "Mooncake forward-mode (pushforward)" begin
        backend = AutoMooncakeForward(config=nothing)

        test_pushforward(x -> x^2, 1.0, backend, 3.0)
        test_pushforward(x -> sum(x .^ 2), [1.0, 0.0, 0.0], backend, [1.0, 2.0, 3.0])
        test_pushforward(x -> x .^ 2, [1.0, 1.0, 1.0], backend, [1.0, 2.0, 3.0])
        test_pushforward(x -> x .^ 2, [0.0, 1.0, -1.0], backend, [1.0, 2.0, 3.0])
        test_pushforward(
            (x, y) -> sum(x .* y), ([1.0, 0.0], [0.0, 1.0]), backend, [1.0, 2.0], [3.0, 4.0]
        )
        test_pushforward(x -> real(x * conj(x)), 1.0 + 0.0im, backend, 1.0 + 2.0im)

        @testset "ẋ scaling" begin
            f = x -> x .^ 2
            x = [1.0, 2.0, 3.0]
            ẋ = [1.0, 1.0, 1.0]
            _, ẏ1 = value_and_pushforward!!(f, ẋ, backend, x)
            _, ẏ2 = value_and_pushforward!!(f, 2 .* ẋ, backend, x)
            @test ẏ2 ≈ 2 .* ẏ1
        end
    end

end
