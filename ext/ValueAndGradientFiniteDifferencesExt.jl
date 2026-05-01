module ValueAndGradientFiniteDifferencesExt

using ValueAndGradient: ValueAndGradient
using FiniteDifferences: central_fdm, grad, jvp
using Test: @test, @testset
using ADTypes: AbstractADType

const _fdm = central_fdm(5, 1)

_vdot(ȳ::Number, y::Number) = ȳ * y
_vdot(ȳ::AbstractArray, y::AbstractArray) = sum(conj.(ȳ) .* y)
_vdot(ȳ::Tuple, y::Tuple) = sum(_vdot(ȳi, yi) for (ȳi, yi) in zip(ȳ, y))

_collect(x::AbstractArray) = collect(x)
_collect(x::Number) = x
_collect(x::Tuple) = map(_collect, x)

_isapprox(a, b; kw...) = isapprox(a, b; kw...)
_isapprox(a::Tuple, b::Tuple; kw...) = all(_isapprox(ai, bi; kw...) for (ai, bi) in zip(a, b))

function ValueAndGradient.test_pullback(f, ȳ, backend::AbstractADType, xs...; rtol=1e-5, atol=1e-5)
    N = length(xs)
    @testset "value_and_pullback!!: $(typeof(backend)), $(map(typeof, xs))" begin
        y, x̄s = ValueAndGradient.value_and_pullback!!(f, ȳ, backend, xs...)

        @testset "value correct" begin
            @test y == f(xs...)
        end

        @testset "pullback matches finite differences" begin
            if N == 1
                fd_x̄ = grad(_fdm, t -> _vdot(ȳ, f(t)), only(xs))[1]
                @test _isapprox(_collect(x̄s), _collect(fd_x̄); rtol, atol)
            else
                for k in 1:N
                    fk = xk -> f(ntuple(i -> i == k ? xk : xs[i], Val(N))...)
                    fd_x̄k = grad(_fdm, t -> _vdot(ȳ, fk(t)), xs[k])[1]
                    @test _isapprox(_collect(x̄s[k]), _collect(fd_x̄k); rtol, atol)
                end
            end
        end
    end
    return nothing
end

function ValueAndGradient.test_pushforward(f, ẋ, backend::AbstractADType, xs...; rtol=1e-5, atol=1e-5)
    N = length(xs)
    @testset "value_and_pushforward!!: $(typeof(backend)), $(map(typeof, xs))" begin
        y, ẏ = ValueAndGradient.value_and_pushforward!!(f, ẋ, backend, xs...)

        @testset "value correct" begin
            @test y == f(xs...)
        end

        @testset "pushforward matches finite differences" begin
            if ẏ isa Tuple
                for k in eachindex(ẏ)
                    fk = N == 1 ? (t -> f(t)[k]) : ((args...) -> f(args...)[k])
                    fd_ẏk = N == 1 ? jvp(_fdm, fk, (only(xs), ẋ)) :
                                     jvp(_fdm, fk, ntuple(i -> (xs[i], ẋ[i]), Val(N))...)
                    @test _isapprox(_collect(ẏ[k]), _collect(fd_ẏk); rtol, atol)
                end
            else
                fd_ẏ = N == 1 ? jvp(_fdm, f, (only(xs), ẋ)) :
                                jvp(_fdm, (args...) -> f(args...), ntuple(k -> (xs[k], ẋ[k]), Val(N))...)
                @test _isapprox(_collect(ẏ), _collect(fd_ẏ); rtol, atol)
            end
        end
    end
    return nothing
end

end
