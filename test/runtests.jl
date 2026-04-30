using ValueAndGradient
using ValueAndGradient.TestUtils
import ADTypes
using ADTypes: AbstractADType, AutoMooncake, AutoMooncakeForward
using Mooncake
using Test

@testset "ValueAndGradient" begin

    @testset "GradientOrder" begin
        @test GradientOrder{1}() isa GradientOrder
        @test GradientOrder(1) == GradientOrder{1}()
        @test GradientOrder{1}() < GradientOrder{2}()
        @test !(GradientOrder{2}() < GradientOrder{1}())
        @test_throws ArgumentError GradientOrder{-1}()
    end

    @testset "error fallbacks" begin
        struct _FakeBackend <: AbstractADType end
        ADTypes.mode(::_FakeBackend) = ADTypes.ForwardMode()
        f = x -> sum(x .^ 2)
        x = [1.0, 2.0]
        @test_throws ArgumentError ValueAndGradient.prepare_pullback_cache(f, _FakeBackend(), x)
        @test_throws ArgumentError ValueAndGradient.prepare_pushforward_cache(f, _FakeBackend(), x)
        @test ValueAndGradient.gradient_order(_FakeBackend()) === nothing
    end

    @testset "Mooncake reverse-mode (pullback)" begin
        backend = AutoMooncake(config=nothing)
        @test gradient_order(backend) == GradientOrder{1}()

        # scalar → scalar
        test_pullback(x -> x^2, 1.0, backend, 3.0)

        # array → scalar, unit seed
        test_pullback(x -> sum(x .^ 2), 1.0, backend, [1.0, 2.0, 3.0])

        # array → array, non-trivial seed (verifies ȳ is actually used)
        test_pullback(x -> x .^ 2, [2.0, -1.0, 3.0], backend, [1.0, 2.0, 3.0])

        # tuple input
        test_pullback(x -> x[1]^2 + x[2]^2, 1.0, backend, (1.0, 2.0))

        # multiple array args
        test_pullback((x, y) -> sum(x .* y), 1.0, backend, [1.0, 2.0], [3.0, 4.0])

        # multiple args, non-unit seed
        test_pullback((x, y) -> x .* y, [1.0, -1.0], backend, [1.0, 2.0], [3.0, 4.0])

        # complex scalar
        test_pullback(x -> real(x * conj(x)), 1.0, backend, 1.0 + 2.0im)

        # complex array
        test_pullback(x -> real(sum(x .* conj.(x))), 1.0, backend, [1.0+2.0im, 3.0+4.0im])

        # manually verify non-unit ȳ is used: ȳ=2 should double the gradient
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
        @test gradient_order(backend) == GradientOrder{1}()

        # scalar → scalar
        test_pushforward(x -> x^2, 1.0, backend, 3.0)

        # array → scalar
        test_pushforward(x -> sum(x .^ 2), [1.0, 0.0, 0.0], backend, [1.0, 2.0, 3.0])

        # array → array
        test_pushforward(x -> x .^ 2, [1.0, 1.0, 1.0], backend, [1.0, 2.0, 3.0])

        # array → array, non-trivial tangent
        test_pushforward(x -> x .^ 2, [0.0, 1.0, -1.0], backend, [1.0, 2.0, 3.0])

        # multiple array args
        test_pushforward(
            (x, y) -> sum(x .* y), ([1.0, 0.0], [0.0, 1.0]), backend, [1.0, 2.0], [3.0, 4.0]
        )

        # complex scalar
        test_pushforward(x -> real(x * conj(x)), 1.0 + 0.0im, backend, 1.0 + 2.0im)

        # manually verify ẋ scaling: doubling tangent should double ẏ
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
