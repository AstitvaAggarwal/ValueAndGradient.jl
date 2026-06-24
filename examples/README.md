# ValueAndGradient.jl — Evaluation Scripts

Five standalone scripts demonstrating `value_and_pullback!!` and `value_and_pushforward!!`
across the SciML ecosystem. Each script runs multiple backends on the same loss function
to show that swapping the backend is a one-argument change.

| Script | Problem | Validates against |
|---|---|---|
| `ode_param_estimation.jl` | ODE parameter estimation | finite differences + convergence |
| `integrals_gradient.jl` | Numerical integration gradient | analytical formula |
| `linear_solve.jl` | Parameterised linear system | finite differences |
| `lux_training.jl` | Lux MLP training loop | gradient check + loss decrease |
| `neural_ode.jl` | Neural ODE (Lux inside ODE RHS) | finite differences + convergence |

## Setup

From the repo root, install dependencies into a temporary environment:

```julia
julia --project=examples/ -e '
    using Pkg
    Pkg.develop(path=".")   # add local ValueAndGradient
    Pkg.add([
        "OrdinaryDiffEq", "SciMLSensitivity",
        "Integrals",
        "LinearSolve",
        "Lux", "Optimisers",
        "Mooncake", "Zygote", "ForwardDiff", "FiniteDifferences",
        "ADTypes", "Random",
    ])
'
```

Then run any script with:

```
julia --project=examples/ examples/ode_param_estimation.jl
```

## Backends used per script

| Script | Backends |
|---|---|
| `ode_param_estimation.jl` | AutoMooncake, AutoZygote, AutoFiniteDifferences |
| `integrals_gradient.jl` | AutoForwardDiff, AutoFiniteDifferences |
| `linear_solve.jl` | AutoMooncake, AutoZygote, AutoFiniteDifferences |
| `lux_training.jl` | AutoMooncake, AutoZygote |
| `neural_ode.jl` | AutoMooncake, AutoZygote, AutoFiniteDifferences |

`integrals_gradient.jl` uses AutoForwardDiff instead of Mooncake/Zygote because `IntegralsMooncakeExt` requires Integrals v5+ which conflicts with the Mooncake 0.5 SciMLBase compat bounds in this environment, and AutoZygote segfaults on Julia 1.12 inside QuadGK's error-handling path.
