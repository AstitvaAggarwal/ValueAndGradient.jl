using ADKernel
using ADKernel.TestUtils
using ADTypes
using Mooncake
using Test

@testset "ADKernel" begin

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
        @test_throws ArgumentError ADKernel.prepare_gradient_cache(f, _FakeBackend(), x)
        @test_throws ArgumentError ADKernel.prepare_jacobian_cache(f, _FakeBackend(), x)
        @test ADKernel.gradient_order(_FakeBackend()) === nothing
    end

    @testset "Mooncake reverse-mode backend" begin
        backend = AutoMooncake(config=nothing)

        @test gradient_order(backend) == GradientOrder{1}()

        @testset "gradient:scalar input" begin
            test_value_and_gradient(x -> x^2, backend, 3.0)
        end

        @testset "gradient:array input" begin
            test_value_and_gradient(x -> sum(x .^ 2), backend, [1.0, 2.0, 3.0])
        end

        @testset "gradient:tuple input" begin
            test_value_and_gradient(x -> x[1]^2 + x[2]^2, backend, (1.0, 2.0))
        end

        @testset "gradient:multiple array args" begin
            test_value_and_gradient((x, y) -> sum(x .* y), backend, [1.0, 2.0], [3.0, 4.0])
        end

        @testset "jacobian:scalar-valued" begin
            test_value_and_jacobian(x -> sum(x .^ 2), backend, [1.0, 2.0, 3.0])
        end

        @testset "jacobian:vector-valued" begin
            test_value_and_jacobian(x -> x .^ 2, backend, [1.0, 2.0, 3.0])
        end

        @testset "jacobian:multiple args scalar output" begin
            f = (x, y) -> sum(x .* y)
            x, y = [1.0, 2.0], [3.0, 4.0]
            val, (Jx, Jy) = ADKernel.value_and_jacobian!!(f, backend, x, y)
            @test val ≈ f(x, y)
            @test Jx ≈ y
            @test Jy ≈ x
        end

        @testset "gradient:complex scalar" begin
            test_value_and_gradient(x -> real(x * conj(x)), backend, 1.0 + 2.0im)
        end

        @testset "gradient:complex array" begin
            test_value_and_gradient(x -> real(sum(x .* conj.(x))), backend, [1.0+2.0im, 3.0+4.0im])
        end
    end

    @testset "Mooncake forward-mode backend" begin
        backend = AutoMooncakeForward(config=nothing)

        @test gradient_order(backend) == GradientOrder{1}()

        @testset "gradient:scalar input" begin
            test_value_and_gradient(x -> x^2, backend, 3.0)
        end

        @testset "gradient:array input" begin
            test_value_and_gradient(x -> sum(x .^ 2), backend, [1.0, 2.0, 3.0])
        end

        @testset "gradient:tuple input" begin
            test_value_and_gradient(x -> x[1]^2 + x[2]^2, backend, (1.0, 2.0))
        end

        @testset "gradient:multiple array args" begin
            test_value_and_gradient((x, y) -> sum(x .* y), backend, [1.0, 2.0], [3.0, 4.0])
        end

        @testset "jacobian:scalar-valued" begin
            test_value_and_jacobian(x -> sum(x .^ 2), backend, [1.0, 2.0, 3.0])
        end

        @testset "jacobian:vector-valued" begin
            test_value_and_jacobian(x -> x .^ 2, backend, [1.0, 2.0, 3.0])
        end

        @testset "jacobian:multiple args scalar output" begin
            f = (x, y) -> sum(x .* y)
            x, y = [1.0, 2.0], [3.0, 4.0]
            val, (Jx, Jy) = ADKernel.value_and_jacobian!!(f, backend, x, y)
            @test val ≈ f(x, y)
            @test Jx ≈ y
            @test Jy ≈ x
        end

        @testset "gradient:complex scalar" begin
            test_value_and_gradient(x -> real(x * conj(x)), backend, 1.0 + 2.0im)
        end

        @testset "gradient:complex array" begin
            test_value_and_gradient(x -> real(sum(x .* conj.(x))), backend, [1.0+2.0im, 3.0+4.0im])
        end
    end

end
