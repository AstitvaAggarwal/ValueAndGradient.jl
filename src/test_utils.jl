module TestUtils

using Test: @test, @testset
using FiniteDifferences: central_fdm, grad, jacobian
using ADTypes: AbstractADType
using ..ADKernel:
    value_and_gradient!!,
    value_and_jacobian!!,
    prepare_gradient_cache,
    prepare_jacobian_cache,
    AbstractGradientCache

# Finite-difference helpers

const _fdm = central_fdm(5, 1)

# Single array or scalar arg: wrap in tuple for uniform return shape
_fd_gradient(f, x::AbstractArray) = (grad(_fdm, f, x)[1],)
_fd_gradient(f, x::Number) = (grad(_fdm, f, x)[1],)

# Tuple arg: perturb each element independently, return tuple of gradients
function _fd_gradient(f, x::Tuple)
    N = length(x)
    gs = ntuple(Val(N)) do i
        fi = v -> f(ntuple(j -> j == i ? v : x[j], Val(N)))
        grad(_fdm, fi, x[i])[1]
    end
    return (gs,)  # wrapped in outer tuple: single argument, gradient is a tuple
end

# Multiple args: grad returns one gradient per argument
function _fd_gradient(f, xs::Vararg{Any, N}) where {N}
    return grad(_fdm, (args...) -> f(args...), xs...)
end

function _fd_jacobian(f, x::AbstractArray)
    y = f(x)
    return y isa Number ? grad(_fdm, f, x)[1] : jacobian(_fdm, f, x)[1]
end
_fd_jacobian(f, x::Number) = grad(_fdm, f, x)[1]

# Core test functions

"""
    test_value_and_gradient(f, backend, xs...; rtol=1e-5, atol=1e-5)

Check that `value_and_gradient!!` gives correct results for `backend` on `xs...`,
using finite differences as the reference. Tests both the one-shot and cached forms,
and checks that repeated cached calls are consistent.
"""
function test_value_and_gradient(f, backend::AbstractADType, xs...; rtol=1e-5, atol=1e-5)
    N = length(xs)
    # For N==1 the backend returns a single gradient (possibly a tuple for tuple inputs);
    # for N>1 it returns a tuple (g1, g2, ...). Wrap in a tuple so the loop below
    # works the same way for both cases.
    _wrap(g) = N == 1 ? (g,) : Tuple(g)

    @testset "value_and_gradient!!: $(typeof(backend)), $(map(typeof, xs))" begin

        # one-shot form
        y, gs_raw = value_and_gradient!!(f, backend, xs...)
        gs = _wrap(gs_raw)

        @testset "value correct" begin
            @test y ≈ f(xs...)
        end

        @testset "gradients match finite differences" begin
            fd_gs = _fd_gradient(f, xs...)
            for i in eachindex(fd_gs)
                @test isapprox(_collect(gs[i]), _collect(fd_gs[i]); rtol, atol)
            end
        end

        # cached form
        cache = prepare_gradient_cache(f, backend, xs...)

        @testset "prepare_gradient_cache returns AbstractGradientCache" begin
            @test cache isa AbstractGradientCache
        end

        @testset "cached form agrees with one-shot" begin
            y2, gs2_raw = value_and_gradient!!(cache, f, xs...)
            gs2 = _wrap(gs2_raw)
            @test y2 ≈ y
            for i in eachindex(gs)
                @test isapprox(_collect(gs2[i]), _collect(gs[i]); rtol, atol)
            end
        end

        @testset "repeated cached calls are consistent" begin
            y3, gs3_raw = value_and_gradient!!(cache, f, xs...)
            gs3 = _collect.(_wrap(deepcopy(gs3_raw)))
            y4, gs4_raw = value_and_gradient!!(cache, f, xs...)
            gs4 = _collect.(_wrap(gs4_raw))
            @test y3 ≈ y4
            for i in eachindex(gs3)
                @test isapprox(gs3[i], gs4[i]; rtol, atol)
            end
        end
    end
    return nothing
end

"""
    test_value_and_jacobian(f, backend, x; rtol=1e-5, atol=1e-5)

Check that `value_and_jacobian!!` gives correct results for `backend` on `x`,
using finite differences as the reference.
"""
function test_value_and_jacobian(f, backend::AbstractADType, x; rtol=1e-5, atol=1e-5)
    @testset "value_and_jacobian!!: $(typeof(backend)), x::$(typeof(x))" begin

        y, J = value_and_jacobian!!(f, backend, x)
        fd_J = _fd_jacobian(f, x)

        @testset "value correct" begin
            @test y ≈ f(x)
        end

        @testset "Jacobian matches finite differences" begin
            @test isapprox(_collect(J), _collect(fd_J); rtol, atol)
        end

        cache = prepare_jacobian_cache(f, backend, x)

        @testset "prepare_jacobian_cache returns AbstractGradientCache" begin
            @test cache isa AbstractGradientCache
        end

        @testset "cached form agrees with one-shot" begin
            y2, J2 = value_and_jacobian!!(cache, f, x)
            @test y2 ≈ y
            @test isapprox(_collect(J2), _collect(fd_J); rtol, atol)
        end
    end
    return nothing
end

# Internal helpers

_collect(x::AbstractArray) = collect(x)
_collect(x::Number) = x
_collect(x::Tuple) = collect(map(_collect, x))

export test_value_and_gradient, test_value_and_jacobian

end
