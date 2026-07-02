using ValueAndGradient
using FiniteDifferences, Test
using ADTypes:
    AutoMooncake,
    AutoMooncakeForward,
    AutoFiniteDifferences,
    AutoFiniteDiff,
    AutoForwardDiff,
    AutoReverseDiff,
    AutoTracker,
    AutoZygote,
    AutoEnzyme
using Mooncake: Mooncake
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Tracker: Tracker
using Zygote: Zygote
using Enzyme: Enzyme
using LinearAlgebra

struct VGOutput{T}
    a::T;
    b::T
end

struct NoCanonStruct
    a::Float64;
    b::Float64
    NoCanonStruct() = new(0.0, 0.0)  # inner constructor blocks default (a,b) positional form
end

@testset "ValueAndGradient" begin

    # ---- AutoMooncake (pullback) + AutoMooncakeForward (pushforward) ----

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
            test_pullback(
                x -> real(sum(x .* conj.(x))),
                one(T),
                backend,
                CT[CT(1, 2), CT(3, 4)],
            )
            test_pullback(
                x -> (a = sum(x .^ 2), b = x[1] + x[2]),
                (a = one(T), b = one(T)),
                backend,
                T[1, 2, 3],
            )
            test_pullback(
                (x, y) -> (a = sum(x .* y), b = x[1] + y[1]),
                (a = one(T), b = one(T)),
                backend,
                T[1, 2],
                T[3, 4],
            )
            @testset "struct output (array → VGOutput)" begin
                f = x -> VGOutput(sum(x .^ 2), x[1] + x[2])
                x = T[1, 2, 3]
                ȳ = Mooncake.Tangent{NamedTuple{(:a, :b),Tuple{T,T}}}((
                    a = one(T),
                    b = one(T),
                ))
                y, x̄ = value_and_pullback!!(f, ȳ, backend, x)
                @test y isa VGOutput{T}
                @test x̄ ≈ 2 .* x + T[1, 1, 0]
            end
            @testset "complex output (real → complex)" begin
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
            test_pushforward(
                (x, y) -> sum(x .* y),
                (T[1, 0], T[0, 1]),
                backend,
                T[1, 2],
                T[3, 4],
            )
            test_pushforward(x -> (x[1]^2, x[2]^2), T[1, 1], backend, T[1, 2])
            test_pushforward(x -> (sum(x), x .^ 2), ones(T, 3), backend, T[1, 2, 3])
            test_pushforward(x -> real(x * conj(x)), one(CT), backend, CT(1, 2))
            test_pushforward(
                x -> real(sum(x .* conj.(x))),
                CT[one(CT), one(CT)],
                backend,
                CT[CT(1, 2), CT(3, 4)],
            )
            test_pushforward(x -> Complex(x^2, x), one(T), backend, T(3))
            test_pushforward(
                x -> (a = sum(x .^ 2), b = x[1] + x[2]),
                ones(T, 3),
                backend,
                T[1, 2, 3],
            )
            test_pushforward(
                (x, y) -> (a = sum(x .* y), b = x[1] + y[1]),
                (T[1, 0], T[0, 1]),
                backend,
                T[1, 2],
                T[3, 4],
            )
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

        @testset "cached pullback multi-arg T=$T" begin
            f2 = (x, y) -> sum(x .* y)
            x2, y2 = T[1, 2], T[3, 4]
            backend2 = AutoMooncake(config = nothing)
            cache2 = Mooncake.prepare_pullback_cache(f2, x2, y2)
            val, x̄s = value_and_pullback!!(f2, one(T), backend2, x2, y2; ad_cache = cache2)
            @test val ≈ f2(x2, y2)
            @test x̄s[1] ≈ y2
            @test x̄s[2] ≈ x2
            val2, x̄s2 = value_and_pullback!!(f2, one(T), backend2, x2, y2)
            @test val ≈ val2
            @test x̄s[1] ≈ x̄s2[1]
            @test x̄s[2] ≈ x̄s2[2]
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

        @testset "cached pushforward multi-arg T=$T" begin
            f2 = (x, y) -> x .* y
            x2, y2 = T[1, 2], T[3, 4]
            ẋ2 = (ones(T, 2), ones(T, 2))
            backend2 = AutoMooncakeForward(config = nothing)
            cache2 = Mooncake.prepare_derivative_cache(f2, x2, y2)
            val, ẏ2 = value_and_pushforward!!(f2, ẋ2, backend2, x2, y2; ad_cache = cache2)
            @test val ≈ f2(x2, y2)
            @test ẏ2 ≈ x2 .+ y2
            val2, ẏ2b = value_and_pushforward!!(f2, ẋ2, backend2, x2, y2)
            @test val ≈ val2
            @test ẏ2 ≈ ẏ2b
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

    @testset "fallback warnings" begin
        # Forward-mode backend used for pullback: warns and falls back via pushforward calls
        @test_warn "not natively supported" value_and_pullback!!(
            x -> x^2,
            1.0,
            AutoMooncakeForward(config = nothing),
            2.0,
        )
        # Reverse-mode backend used for pushforward: warns and falls back via pullback calls
        @test_warn "not natively supported" value_and_pushforward!!(
            x -> x^2,
            1.0,
            AutoMooncake(config = nothing),
            2.0,
        )
    end

    # ---- AutoFiniteDifferences (both ops) ----

    for T in (Float32, Float64)
        fdm_backend = AutoFiniteDifferences(fdm = central_fdm(5, 1))

        @testset "pullback AutoFiniteDifferences T=$T" begin
            test_pullback(x -> x^2, one(T), fdm_backend, T(3))
            test_pullback(x -> sum(x .^ 2), one(T), fdm_backend, T[1, 2, 3])
            test_pullback(x -> x .^ 2, T[2, -1, 3], fdm_backend, T[1, 2, 3])
            test_pullback((x, y) -> sum(x .* y), one(T), fdm_backend, T[1, 2], T[3, 4])
            test_pullback((x, y) -> x .* y, T[1, -1], fdm_backend, T[1, 2], T[3, 4])
        end

        @testset "pushforward AutoFiniteDifferences T=$T" begin
            test_pushforward(x -> x^2, one(T), fdm_backend, T(3))
            test_pushforward(x -> sum(x .^ 2), T[1, 0, 0], fdm_backend, T[1, 2, 3])
            test_pushforward(x -> x .^ 2, T[0, 1, -1], fdm_backend, T[1, 2, 3])
            test_pushforward(
                (x, y) -> sum(x .* y),
                (T[1, 0], T[0, 1]),
                fdm_backend,
                T[1, 2],
                T[3, 4],
            )
        end

        CT = Complex{T}
        @testset "pullback AutoFiniteDifferences complex T=$T" begin
            test_pullback(x -> real(x * conj(x)), one(T), fdm_backend, CT(1, 2))
            test_pullback(
                x -> real(sum(x .* conj.(x))),
                one(T),
                fdm_backend,
                CT[CT(1, 2), CT(3, 4)],
            )
            test_pullback(
                (x, y) -> real(sum(x .* conj.(y))),
                one(T),
                fdm_backend,
                CT[CT(1, 2)],
                CT[CT(3, 4)],
            )
        end

        @testset "pushforward AutoFiniteDifferences complex T=$T" begin
            test_pushforward(x -> real(x * conj(x)), one(CT), fdm_backend, CT(1, 2))
            test_pushforward(
                x -> real(sum(x .* conj.(x))),
                CT[one(CT), one(CT)],
                fdm_backend,
                CT[CT(1, 2), CT(3, 4)],
            )
        end
    end

    # ---- AutoFiniteDiff (both ops, AbstractArray only) ----

    for T in (Float32, Float64)
        fd_backend = AutoFiniteDiff()

        @testset "pullback AutoFiniteDiff T=$T" begin
            test_pullback(x -> sum(x .^ 2), one(T), fd_backend, T[1, 2, 3])
            test_pullback(x -> x .^ 2, T[2, -1, 3], fd_backend, T[1, 2, 3])
            test_pullback((x, y) -> sum(x .* y), one(T), fd_backend, T[1, 2], T[3, 4])
            test_pullback((x, y) -> x .* y, T[1, -1], fd_backend, T[1, 2], T[3, 4])
        end

        @testset "pushforward AutoFiniteDiff T=$T" begin
            # Float32 central differences have ~1e-4 cancellation error; Float64 uses default 1e-5.
            tol = T === Float32 ? 1e-2 : 1e-5
            test_pushforward(
                x -> sum(x .^ 2),
                T[1, 0, 0],
                fd_backend,
                T[1, 2, 3];
                rtol = tol,
                atol = tol,
            )
            test_pushforward(
                x -> x .^ 2,
                T[0, 1, -1],
                fd_backend,
                T[1, 2, 3];
                rtol = tol,
                atol = tol,
            )
            test_pushforward(
                (x, y) -> sum(x .* y),
                (T[1, 0], T[0, 1]),
                fd_backend,
                T[1, 2],
                T[3, 4];
                rtol = tol,
                atol = tol,
            )
        end

        @testset "cached pullback AutoFiniteDiff T=$T" begin
            f = x -> sum(x .^ 2)
            x = T[1, 2, 3]
            ∂x = similar(x)
            cache = FiniteDiff.GradientCache(∂x, x)
            y, x̄ = value_and_pullback!!(f, one(T), fd_backend, x; ad_cache = cache)
            @test y ≈ f(x)
            @test x̄ ≈ 2 .* x
            y2, x̄2 = value_and_pullback!!(f, one(T), fd_backend, x)
            @test y ≈ y2
            @test x̄ ≈ x̄2
        end
    end

    # ---- AutoForwardDiff (pushforward only) ----

    for T in (Float32, Float64)
        fwd_backend = AutoForwardDiff()

        @testset "pushforward AutoForwardDiff T=$T" begin
            test_pushforward(x -> x^2, one(T), fwd_backend, T(3))
            test_pushforward(x -> sum(x .^ 2), T[1, 0, 0], fwd_backend, T[1, 2, 3])
            test_pushforward(x -> x .^ 2, T[0, 1, -1], fwd_backend, T[1, 2, 3])
            test_pushforward(
                (x, y) -> sum(x .* y),
                (T[1, 0], T[0, 1]),
                fwd_backend,
                T[1, 2],
                T[3, 4],
            )
        end

        @testset "pullback AutoForwardDiff (derived via pushforward) T=$T" begin
            # Falls back to n pushforward calls — warns but produces correct result
            test_pullback(x -> x^2, one(T), fwd_backend, T(3))
            test_pullback(x -> sum(x .^ 2), one(T), fwd_backend, T[1, 2, 3])
            test_pullback(x -> x .^ 2, T[2, -1, 3], fwd_backend, T[1, 2, 3])
            test_pullback((x, y) -> sum(x .* y), one(T), fwd_backend, T[1, 2], T[3, 4])
            test_pullback(x -> (x[1]^2, x[2]^2), (one(T), one(T)), fwd_backend, T[1, 2])
            test_pullback(
                x -> (a = x[1]^2, b = x[2]^2),
                (a = one(T), b = one(T)),
                fwd_backend,
                T[1, 2],
            )
        end
    end

    # ---- AutoReverseDiff (pullback native; pushforward derived) ----

    for T in (Float32, Float64)
        rd_backend = AutoReverseDiff()

        @testset "pullback AutoReverseDiff T=$T" begin
            test_pullback(x -> sum(x .^ 2), one(T), rd_backend, T[1, 2, 3])
            test_pullback(x -> x .^ 2, T[2, -1, 3], rd_backend, T[1, 2, 3])
            test_pullback((x, y) -> sum(x .* y), one(T), rd_backend, T[1, 2], T[3, 4])
            test_pullback((x, y) -> x .* y, T[1, -1], rd_backend, T[1, 2], T[3, 4])
        end

        @testset "cached pullback AutoReverseDiff T=$T" begin
            f = x -> sum(x .^ 2)
            x = T[1, 2, 3]
            cache = ReverseDiff.compile(ReverseDiff.GradientTape(f, x))
            y, x̄ = value_and_pullback!!(f, one(T), rd_backend, x; ad_cache = cache)
            @test y ≈ f(x)
            @test x̄ ≈ 2 .* x
            y2, x̄2 = value_and_pullback!!(f, one(T), rd_backend, x)
            @test y ≈ y2
            @test x̄ ≈ x̄2
        end

        @testset "cached pullback AutoReverseDiff multi-arg T=$T" begin
            f2 = (x, y) -> sum(x .* y)
            x2, y2 = T[1, 2], T[3, 4]
            cache2 = ReverseDiff.compile(ReverseDiff.GradientTape(f2, (x2, y2)))
            val, x̄s =
                value_and_pullback!!(f2, one(T), rd_backend, x2, y2; ad_cache = cache2)
            @test val ≈ f2(x2, y2)
            @test x̄s[1] ≈ y2
            @test x̄s[2] ≈ x2
            val2, x̄s2 = value_and_pullback!!(f2, one(T), rd_backend, x2, y2)
            @test val ≈ val2
            @test x̄s[1] ≈ x̄s2[1]
            @test x̄s[2] ≈ x̄s2[2]
        end

        @testset "pushforward AutoReverseDiff (derived via pullback) T=$T" begin
            # Falls back to m pullback calls — warns but produces correct result
            test_pushforward(x -> sum(x .^ 2), T[1, 0, 0], rd_backend, T[1, 2, 3])
            test_pushforward(x -> x .^ 2, T[0, 1, -1], rd_backend, T[1, 2, 3])
            test_pushforward(
                (x, y) -> sum(x .* y),
                (T[1, 0], T[0, 1]),
                rd_backend,
                T[1, 2],
                T[3, 4],
            )
            test_pushforward(x -> (x[1]^2, x[2]^2), T[1, 1], rd_backend, T[1, 2])
            test_pushforward(x -> (a = x[1]^2, b = x[2]^2), T[1, 1], rd_backend, T[1, 2])
        end
    end

    # ---- AutoTracker (pullback native; pushforward derived) ----

    for T in (Float32, Float64)
        tk_backend = AutoTracker()

        @testset "pullback AutoTracker T=$T" begin
            test_pullback(x -> sum(x .^ 2), one(T), tk_backend, T[1, 2, 3])
            test_pullback(x -> x .^ 2, T[2, -1, 3], tk_backend, T[1, 2, 3])
            test_pullback((x, y) -> sum(x .* y), one(T), tk_backend, T[1, 2], T[3, 4])
            test_pullback((x, y) -> x .* y, T[1, -1], tk_backend, T[1, 2], T[3, 4])
        end

        @testset "pushforward AutoTracker (derived via pullback) T=$T" begin
            test_pushforward(x -> sum(x .^ 2), T[1, 0, 0], tk_backend, T[1, 2, 3])
            test_pushforward(x -> x .^ 2, T[0, 1, -1], tk_backend, T[1, 2, 3])
            test_pushforward(
                (x, y) -> sum(x .* y),
                (T[1, 0], T[0, 1]),
                tk_backend,
                T[1, 2],
                T[3, 4],
            )
            test_pushforward(x -> (x[1]^2, x[2]^2), T[1, 1], tk_backend, T[1, 2])
            test_pushforward(x -> (a = x[1]^2, b = x[2]^2), T[1, 1], tk_backend, T[1, 2])
        end
    end

    # ---- AutoZygote (pullback native; pushforward derived) ----

    for T in (Float32, Float64)
        zy_backend = AutoZygote()

        @testset "pullback AutoZygote T=$T" begin
            test_pullback(x -> x^2, one(T), zy_backend, T(3))
            test_pullback(x -> sum(x .^ 2), one(T), zy_backend, T[1, 2, 3])
            test_pullback(x -> x .^ 2, T[2, -1, 3], zy_backend, T[1, 2, 3])
            test_pullback((x, y) -> sum(x .* y), one(T), zy_backend, T[1, 2], T[3, 4])
            test_pullback((x, y) -> x .* y, T[1, -1], zy_backend, T[1, 2], T[3, 4])
        end

        @testset "pushforward AutoZygote (derived via pullback) T=$T" begin
            test_pushforward(x -> x^2, one(T), zy_backend, T(3))
            test_pushforward(x -> sum(x .^ 2), T[1, 0, 0], zy_backend, T[1, 2, 3])
            test_pushforward(x -> x .^ 2, T[0, 1, -1], zy_backend, T[1, 2, 3])
            test_pushforward(
                (x, y) -> sum(x .* y),
                (T[1, 0], T[0, 1]),
                zy_backend,
                T[1, 2],
                T[3, 4],
            )
            test_pushforward(x -> (x[1]^2, x[2]^2), T[1, 1], zy_backend, T[1, 2])
            test_pushforward(x -> (a = x[1]^2, b = x[2]^2), T[1, 1], zy_backend, T[1, 2])
        end

        CT = Complex{T}
        @testset "pullback AutoZygote complex T=$T" begin
            test_pullback(x -> real(x * conj(x)), one(T), zy_backend, CT(1, 2))
            test_pullback(
                x -> real(sum(x .* conj.(x))),
                one(T),
                zy_backend,
                CT[CT(1, 2), CT(3, 4)],
            )
        end

        @testset "pushforward AutoZygote complex (derived) T=$T" begin
            test_pushforward(x -> real(x * conj(x)), one(CT), zy_backend, CT(1, 2))
            test_pushforward(
                x -> real(sum(x .* conj.(x))),
                CT[one(CT), one(CT)],
                zy_backend,
                CT[CT(1, 2), CT(3, 4)],
            )
        end
    end

    # ---- normalise_tangents ----

    @testset "normalise_tangents: nothing → zero (Zygote multi-arg)" begin
        # f ignores y → Zygote returns nothing for ȳ
        f = (x, y) -> sum(x .^ 2)
        x = Float64[1.0, 2.0]
        y = Float64[3.0, 4.0]
        _, x̄s = value_and_pullback!!(f, 1.0, AutoZygote(), x, y; normalise_tangents = true)
        @test x̄s[1] ≈ 2 .* x
        @test x̄s[2] ≈ zero(y)
    end

    @testset "normalise_tangents: Mooncake.Tangent → reconstructed struct (Mooncake pushforward)" begin
        f = x -> VGOutput(sum(x .^ 2), x[1] + x[2])
        x = Float64[1.0, 2.0, 3.0]
        ẋ = ones(Float64, 3)
        backend = AutoMooncakeForward(config = nothing)
        y_raw, ẏ_raw = value_and_pushforward!!(f, ẋ, backend, x)
        @test ẏ_raw isa Mooncake.Tangent  # default: raw Mooncake.Tangent
        y2, ẏ2 = value_and_pushforward!!(f, ẋ, backend, x; normalise_tangents = true)
        @test ẏ2 isa VGOutput{Float64}    # normalised: reconstructed struct
        @test ẏ2.a ≈ sum(2 .* x .* ẋ)
        @test ẏ2.b ≈ 2.0
    end

    @testset "normalise_tangents: DiffLeaf tangent unchanged (AutoFiniteDifferences)" begin
        f = x -> sum(x .^ 2)
        x = Float64[1.0, 2.0, 3.0]
        backend = AutoFiniteDifferences(fdm = central_fdm(5, 1))
        _, x̄_false = value_and_pullback!!(f, 1.0, backend, x; normalise_tangents = false)
        _, x̄_true = value_and_pullback!!(f, 1.0, backend, x; normalise_tangents = true)
        @test x̄_false ≈ x̄_true
    end

    @testset "normalise_tangents: no positional constructor → warns, returns NamedTuple" begin
        x = NoCanonStruct()
        nt = (a = 1.0, b = 2.0)
        result =
            @test_warn "cannot auto-reconstruct" ValueAndGradient._normalise(x, nt, nothing)
        @test result === nt
    end

    # ---- normalise_pullback / normalise_pushforward ----

    @testset "normalise_pushforward: overrides normalise_tangents" begin
        f = x -> VGOutput(sum(x .^ 2), x[1] + x[2])
        x = Float64[1.0, 2.0, 3.0]
        ẋ = ones(Float64, 3)
        backend = AutoMooncakeForward(config = nothing)
        sentinel = Ref(false)
        nf = t -> (sentinel[] = true; VGOutput(Float64(42), Float64(99)))
        _, ẏ = value_and_pushforward!!(
            f,
            ẋ,
            backend,
            x;
            normalise_tangents = true,
            normalise_pushforward = nf,
        )
        @test sentinel[]
        @test ẏ == VGOutput(Float64(42), Float64(99))
    end

    @testset "normalise_pushforward: handles type with no positional constructor" begin
        f = x -> NoCanonStruct()
        x = Float64[1.0]
        ẋ = Float64[1.0]
        backend = AutoMooncakeForward(config = nothing)
        _, ẏ = value_and_pushforward!!(
            f,
            ẋ,
            backend,
            x;
            normalise_pushforward = t -> NoCanonStruct(),
        )
        @test ẏ isa NoCanonStruct
    end

    @testset "normalise_pullback: custom conversion on Zygote nothing cotangent" begin
        f = (x, y) -> sum(x .^ 2)
        x = Float64[1.0, 2.0]
        y = Float64[3.0, 4.0]
        # Zygote returns nothing for unused y; replace with a sentinel value instead of zero
        nf = cotangent -> map(ti -> ti === nothing ? fill(-1.0, 2) : ti, cotangent)
        _, x̄s = value_and_pullback!!(f, 1.0, AutoZygote(), x, y; normalise_pullback = nf)
        @test x̄s[1] ≈ 2 .* x
        @test x̄s[2] ≈ fill(-1.0, 2)
    end

    # ---- AutoEnzyme (both ops, AbstractArray only) ----

    for T in (Float32, Float64)
        enz_rev = AutoEnzyme(mode = Enzyme.Reverse)
        enz_fwd = AutoEnzyme(mode = Enzyme.Forward)

        @testset "pullback AutoEnzyme T=$T" begin
            test_pullback(x -> sum(x .^ 2), one(T), enz_rev, T[1, 2, 3])
            test_pullback(x -> x .^ 2, T[2, -1, 3], enz_rev, T[1, 2, 3])
            test_pullback((x, y) -> sum(x .* y), one(T), enz_rev, T[1, 2], T[3, 4])
            test_pullback((x, y) -> x .* y, T[1, -1], enz_rev, T[1, 2], T[3, 4])
        end

        @testset "pushforward AutoEnzyme T=$T" begin
            test_pushforward(x -> sum(x .^ 2), T[1, 0, 0], enz_fwd, T[1, 2, 3])
            test_pushforward(x -> x .^ 2, T[0, 1, -1], enz_fwd, T[1, 2, 3])
            test_pushforward(
                (x, y) -> sum(x .* y),
                (T[1, 0], T[0, 1]),
                enz_fwd,
                T[1, 2],
                T[3, 4],
            )
        end
    end

end
