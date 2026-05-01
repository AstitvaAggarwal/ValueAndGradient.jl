module TestUtils

using Test: @test, @testset
using FiniteDifferences: central_fdm, grad, jvp
using ADTypes: AbstractADType
using ..ValueAndGradient: value_and_pullback!!, value_and_pushforward!!

const _fdm = central_fdm(5, 1)

_vdot(ȳ::Number, y::Number) = ȳ * y
_vdot(ȳ::AbstractArray, y::AbstractArray) = sum(conj.(ȳ) .* y)

"""
    test_pullback(f, ȳ, backend, xs...; rtol=1e-5, atol=1e-5)

Check `value_and_pullback!!` against finite differences. `ȳ` is the cotangent
seed, a scalar or array matching the output type of `f`.
"""
function test_pullback(f, ȳ, backend::AbstractADType, xs...; rtol=1e-5, atol=1e-5)
    N = length(xs)
    @testset "value_and_pullback!!: $(typeof(backend)), $(map(typeof, xs))" begin
        y, x̄s = value_and_pullback!!(f, ȳ, backend, xs...)

        @testset "value correct" begin
            @test y ≈ f(xs...)
        end

        @testset "pullback matches finite differences" begin
            if N == 1
                fd_x̄ = grad(_fdm, t -> _vdot(ȳ, f(t)), only(xs))[1]
                @test isapprox(_collect(x̄s), _collect(fd_x̄); rtol, atol)
            else
                for k in 1:N
                    fk = xk -> f(ntuple(i -> i == k ? xk : xs[i], Val(N))...)
                    fd_x̄k = grad(_fdm, t -> _vdot(ȳ, fk(t)), xs[k])[1]
                    @test isapprox(_collect(x̄s[k]), _collect(fd_x̄k); rtol, atol)
                end
            end
        end
    end
    return nothing
end

"""
    test_pushforward(f, ẋ, backend, xs...; rtol=1e-5, atol=1e-5)

Check `value_and_pushforward!!` against finite differences. `ẋ` is the tangent
seed, same structure as `x` for single-arg or a tuple of tangents for multi-arg.
"""
function test_pushforward(f, ẋ, backend::AbstractADType, xs...; rtol=1e-5, atol=1e-5)
    N = length(xs)
    @testset "value_and_pushforward!!: $(typeof(backend)), $(map(typeof, xs))" begin
        y, ẏ = value_and_pushforward!!(f, ẋ, backend, xs...)

        @testset "value correct" begin
            @test y ≈ f(xs...)
        end

        @testset "pushforward matches finite differences" begin
            fd_ẏ = if N == 1
                jvp(_fdm, f, (only(xs), ẋ))
            else
                jvp(_fdm, (args...) -> f(args...), ntuple(k -> (xs[k], ẋ[k]), Val(N))...)
            end
            @test isapprox(_collect(ẏ), _collect(fd_ẏ); rtol, atol)
        end
    end
    return nothing
end

_collect(x::AbstractArray) = collect(x)
_collect(x::Number) = x
_collect(x::Tuple) = collect(map(_collect, x))

export test_pullback, test_pushforward

end
