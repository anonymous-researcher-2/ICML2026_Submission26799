"""
    BoolSemiring <: Semiring{Bool}

Boolean semiring: ``(\\{\\text{false}, \\text{true} \\}, \\lor, \\land, \\text{false}, \\text{true} )`` where
``\\lor`` and ``\\land`` are the logical "or" and logical "and operators.
"""
struct BoolSemiring <: Semiring{Bool}
    val::Bool
end

BoolSemiring(x::Semiring) = BoolSemiring(! iszero(x))

Base.:+(x::BoolSemiring, y::BoolSemiring) = BoolSemiring(x.val || y.val)
Base.:*(x::BoolSemiring, y::BoolSemiring) = BoolSemiring(x.val && y.val)
Base.zero(::Type{<:BoolSemiring}) = BoolSemiring(false)
Base.one(::Type{<:BoolSemiring}) = BoolSemiring(true)
Base.inv(x::BoolSemiring) = BoolSemiring(true)


"""
    struct RealSemiring{T} <: Semiring{T}
        val::T
    end
    const ProbSemiring{T} where T = RealSemiring{T}

Probability semiring ``( (\\mathbb{R}_+``, +, \\cdot, 0, 1 )``.
"""
struct RealSemiring{T} <: Semiring{T}
    val::T
end

const ProbSemiring{T} = RealSemiring{T}

Base.:+(x::RealSemiring, y::RealSemiring) = RealSemiring(val(x) + val(y))
Base.:*(x::RealSemiring, y::RealSemiring) = RealSemiring(val(x) * val(y))
Base.zero(S::Type{<:RealSemiring{T}}) where T = S(zero(T))
Base.one(S::Type{<:RealSemiring{T}}) where T = S(one(T))
Base.inv(x::S) where S<:RealSemiring = S(inv(val(x)))

∂sum(z::RealSemiring, x::RealSemiring{T}, Δu::SemiringTangent) where T = SemiringTangent{T}(val(Δu))
∂rmul(x::RealSemiring{T}, a::RealSemiring, Δu::SemiringTangent) where T = SemiringTangent{T}(val(Δu) * val(a))
∂lmul(a::RealSemiring, x::RealSemiring{T}, Δu::SemiringTangent) where T = SemiringTangent{T}(val(Δu) * val(a))


"""
    ScaledLogSemiring{τ,T} <: Semiring{T}
    const LogSemiring{T} = ScaledLogSemiring{1,T}
    const NegativeLogSemiring{T} = ScaledLogSemiring{-1,T}

Scaled Logarithmic semiring: ``(\\mathbb{R} \\cup \\{\\text{sgn}(\\tau) \\cdot +\\infty \\}, \\oplus_{\\log}, +, -\\infty, 0)``
where

```math
x \\oplus y = \\frac{1}{\\tau} \\log ( e^{\\tau x} + e^{\\tau y} ).
```

!!! info
    The logarithmic semiring defined in [OpenFst](https://www.openfst.org/)
    corresponds to `NegativeLogSemiring{T} = ScaledLogSemiring{-1,T}`.
"""
struct ScaledLogSemiring{τ,T} <: Semiring{T}
    val::T
end

const LogSemiring{T} = ScaledLogSemiring{1,T}
const NegativeLogSemiring{T} = ScaledLogSemiring{-1,T}

ScaledLogSemiring{τ}(x::T) where {τ,T} = ScaledLogSemiring{τ,T}(x)

function Base.:+(x::ScaledLogSemiring{τ}, y::ScaledLogSemiring{τ}) where τ
    T = promote_type(valtype(x), valtype(y))
    ScaledLogSemiring{τ,T}(logaddexp(T(τ)*val(x), T(τ)*val(y))/T(τ))
end

Base.:*(x::ScaledLogSemiring{τ}, y::ScaledLogSemiring{τ}) where τ = ScaledLogSemiring{τ}(val(x) + val(y))
Base.inv(x::S) where S<:ScaledLogSemiring= iszero(x) ? S(NaN) : S(-val(x))

Base.:*(i::Integer, x::ScaledLogSemiring{τ}) where τ = ScaledLogSemiring{τ}(val(x) + log(i)/τ)

Base.zero(S::Type{ScaledLogSemiring{τ,T}}) where {τ,T} = S(ifelse(τ > 0, T(-Inf), T(Inf)))
Base.one(S::Type{ScaledLogSemiring{τ,T}}) where {τ,T} = S(T(0))

function ∂sum(z::ScaledLogSemiring{τ}, x::ScaledLogSemiring{τ,T}, Δu::SemiringTangent) where {τ,T}
    if iszero(z)
        SemiringTangent{T}(false)
    else
        SemiringTangent{T}(val(Δu) * exp(τ*(val(x) - val(z))))
    end
end

∂rmul(x::ScaledLogSemiring{τ,T}, a::ScaledLogSemiring{τ}, Δu::SemiringTangent) where {τ,T} = SemiringTangent{T}(Δu.val)
∂lmul(a::ScaledLogSemiring{τ}, x::ScaledLogSemiring{τ,T}, Δu::SemiringTangent) where {τ,T} = SemiringTangent{T}(Δu.val)


"""
    ArcticSemiring{T} <: Semiring{T}

Arctic semiring: ``( \\mathbb{R} \\cup \\{\\infty \\}, \\max, +,
-\\infty, 0 )``. The arctic semiring is not differentiable
(see [`ExtendedArcticSemiring`](@ref) for differentiating).

See also [`ScaledLogSemiring`](@ref) and [`TropicalSemiring`](@ref).
"""
struct ArcticSemiring{T} <: Semiring{T}
    val::T
end

ScaledLogSemiring{T,Inf}(v) where T = ArcticSemiring{T}(v)

Base.:+(x::S, y::S) where S<:ArcticSemiring = S(max(val(x), val(y)))
Base.:*(x::S, y::S) where S<:ArcticSemiring = S(val(x) + val(y))
Base.inv(x::S) where S<:ArcticSemiring = iszero(x) ? S(NaN) : S(-val(x))
Base.zero(S::Type{<:ArcticSemiring}) = ArcticSemiring(-Inf)
Base.zero(S::Type{<:ArcticSemiring{T}}) where T = S(T(-Inf))
Base.one(S::Type{<:ArcticSemiring}) = ArcticSemiring(0)
Base.one(S::Type{<:ArcticSemiring{T}}) where T = S(T(0))


"""
    TropicalSemiring{T} <: Semiring{T}

Tropical semiring: ``(\\mathbb{R} \\cup \\{- \\infty \\}, \\min, +, \\infty, 0)``.
The tropical semiring is not differentiable (see
[`ExtendedTropicalSemiring`](@ref) for differentiating).

See also [`ScaledLogSemiring`](@ref) and [`ArcticSemiring`](@ref).
"""
struct TropicalSemiring{T} <: Semiring{T}
    val::T
end

ScaledLogSemiring{T,-Inf}(v) where T = TropicalSemiring{T}(v)

Base.:+(x::S, y::S) where S<:TropicalSemiring = S(min(val(x), val(y)))

Base.:*(x::S, y::S) where S<:TropicalSemiring = S(val(x) + val(y))
Base.inv(x::S) where S<:TropicalSemiring = iszero(x) ? S(NaN) : S(-val(x))
Base.zero(S::Type{<:TropicalSemiring{T}}) where T = S(T(Inf))
Base.one(S::Type{<:TropicalSemiring}) = S(0)


"""
    ExtendedArcticSemiring <: Semiring

Semiring equivalent to the [`ArcticSemiring`](@ref) which, when differentiated
through (see [`∂sum`](@ref), [`∂lmul`](@ref) and [`∂rmul`](@ref)) will return
a *subgradient* of the computation.
"""
struct ExtendedArcticSemiring{T} <: Semiring{T}
    val::T
    count::T
end

function ExtendedArcticSemiring(v, c)
    T = promote_type(eltype(v), eltype(c))
    ExtendedArcticSemiring{T}(v, c)
end

ExtendedArcticSemiring(v::T) where T = ExtendedArcticSemiring(v, T(1))
ExtendedArcticSemiring{T}(v) where T = ExtendedArcticSemiring{T}(v, 1)

Base.promote_rule(::Type{ArcticSemiring{T1}}, ::Type{ExtendedArcticSemiring{T2}}) where {T1,T2} = ExtendedArcticSemiring{promote_type(T1, T2)}

function Base.:+(x::ExtendedArcticSemiring, y::ExtendedArcticSemiring)
    m = max(val(x), val(y))
    t = x.count + y.count
    ExtendedArcticSemiring(m, (val(x) == m) * x.count + (val(y) == m) * y.count)
end
Base.:+(x::ArcticSemiring, y::ExtendedArcticSemiring) = ExtendedArcticSemiring(val(x), 1) + y
Base.:+(x::ExtendedArcticSemiring, y::ArcticSemiring) = x * ExtendedArcticSemiring(val(y), 1)

Base.:*(x::ExtendedArcticSemiring, y::ExtendedArcticSemiring) = ExtendedArcticSemiring(val(x) + val(y), x.count * y.count)
Base.:*(x::ArcticSemiring, y::ExtendedArcticSemiring) = ExtendedArcticSemiring(val(x), 1) * y
Base.:*(x::ExtendedArcticSemiring, y::ArcticSemiring) = x * ExtendedArcticSemiring(val(y), 1)

Base.inv(x::ExtendedArcticSemiring) = iszero(x) ? ExtendedArcticSemiring(NaN, NaN) : ExtendedArcticSemiring(-val(x), inv(x.count))

Base.zero(S::Type{ExtendedArcticSemiring{T}}) where T = S(T(-Inf), T(0))
Base.one(S::Type{ExtendedArcticSemiring{T}}) where T = S(T(0), T(1))

∂sum(z::ExtendedArcticSemiring, x::ExtendedArcticSemiring{T}, Δu::SemiringTangent) where T = SemiringTangent{T}(Δu.val * (val(x) == val(z)) * x.count / z.count)
∂sum(z::ExtendedArcticSemiring, x::ArcticSemiring{T}, Δu::SemiringTangent) where T = SemiringTangent{T}(Δu.val * (val(x) == val(z)) / z.count)

∂rmul(x::ExtendedArcticSemiring{T}, a::ExtendedArcticSemiring, Δu::SemiringTangent) where T= SemiringTangent{T}(Δu.val)
∂rmul(x::ArcticSemiring{T}, a::ExtendedArcticSemiring, Δu::SemiringTangent) where T = SemiringTangent{T}(Δu.val)
∂rmul(x::ExtendedArcticSemiring{T}, a::ArcticSemiring, Δu::SemiringTangent) where T = SemiringTangent{T}(Δu.val)

∂lmul(a::ExtendedArcticSemiring, x::ExtendedArcticSemiring{T}, Δu) where T = SemiringTangent{T}(Δu.val)
∂lmul(a::ExtendedArcticSemiring, x::ArcticSemiring{T}, Δu) where T = SemiringTangent{T}(Δu.val)
∂lmul(a::ArcticSemiring, x::ExtendedArcticSemiring{T}, Δu) where T = SemiringTangent{T}(Δu.val)


"""
    ExtendedTropicalSemiring{T} <: Semiring{T}

Semiring equivalent to the [`TropicalSemiring`](@ref) which, when differentiated
through will return a *subgradient* of the computation.
"""
struct ExtendedTropicalSemiring{T} <: Semiring{T}
    val::T
    count::T
end

Base.promote_rule(::Type{TropicalSemiring{T1}}, ::Type{ExtendedTropicalSemiring{T2}}) where {T1,T2} = ExtendedTropicalSemiring{promote_type(T1, T2)}

function ExtendedTropicalSemiring(v, c)
    T = promote_type(eltype(v), eltype(c))
    ExtendedTropicalSemiring{T}(v, c)
end

ExtendedTropicalSemiring(v::T) where T = ExtendedTropicalSemiring(v, T(1))
ExtendedTropicalSemiring{T}(v) where T = ExtendedTropicalSemiring{T}(v, 1)

function Base.:+(x::ExtendedTropicalSemiring, y::ExtendedTropicalSemiring)
    m = min(val(x), val(y))
    t = x.count + y.count
    ExtendedTropicalSemiring(m, (val(x) == m) * x.count + (val(y) == m) * y.count)
end
Base.:+(x::TropicalSemiring, y::ExtendedTropicalSemiring) = ExtendedTropicalSemiring(val(x), 1) + y
Base.:+(x::ExtendedTropicalSemiring, y::TropicalSemiring) = x * ExtendedTropicalSemiring(val(y), 1)

Base.:*(x::ExtendedTropicalSemiring, y::ExtendedTropicalSemiring) = ExtendedTropicalSemiring(val(x) + val(y), x.count * y.count)
Base.:*(x::TropicalSemiring, y::ExtendedTropicalSemiring) = ExtendedTropicalSemiring(val(x), 1) * y
Base.:*(x::ExtendedTropicalSemiring, y::TropicalSemiring) = x * ExtendedTropicalSemiring(val(y), 1)

Base.inv(x::ExtendedTropicalSemiring) = iszero(x) ? ExtendedTropicalSemiring(NaN, NaN) : ExtendedTropicalSemiring(-val(x), inv(x.count))

Base.zero(S::Type{ExtendedTropicalSemiring{T}}) where T = S(T(Inf), T(0))
Base.one(S::Type{ExtendedTropicalSemiring{T}}) where T = S(T(0), T(1))

∂sum(z::ExtendedTropicalSemiring, x::ExtendedTropicalSemiring{T}, Δu) where T = SemiringTangent{T}(Δu.val * (val(x) == val(z)) * x.count / z.count)
∂sum(z::ExtendedTropicalSemiring, x::TropicalSemiring{T}, Δu) where T = SemiringTangent{T}(Δu.val * (val(x) == val(z)) / z.count)

∂rmul(x::ExtendedTropicalSemiring{T}, a::ExtendedTropicalSemiring, Δu) where T = SemiringTangent{T}(Δu.val)
∂rmul(x::TropicalSemiring{T}, a::ExtendedTropicalSemiring, Δu) where T = SemiringTangent{T}(Δu.val)
∂rmul(x::ExtendedTropicalSemiring{T}, a::TropicalSemiring, Δu) where T = SemiringTangent{T}(Δu.val)

∂lmul(a::ExtendedTropicalSemiring, x::ExtendedTropicalSemiring{T}, Δu) where T = SemiringTangent{T}(Δu.val)
∂lmul(a::ExtendedTropicalSemiring, x::TropicalSemiring{T}, Δu) where T = SemiringTangent{T}(Δu.val)
∂lmul(a::TropicalSemiring, x::ExtendedTropicalSemiring{T}, Δu) where T = SemiringTangent{T}(Δu.val)


"""
    DualNumberSemiring{T<:AbstractFloat,F} <: Semiring{T}

[Dual number](https://en.wikipedia.org/wiki/Dual_number) semiring:
``( \\mathbb{R} \\times \\mathbb{R}, \\oplus_{\\text{dual}}, \\otimes_{\\text{dual}}, 0 + 0\\epsilon, 1 + 0\\epsilon )``
where
```math
\\begin{align}
    a + b\\epsilon \\oplus_{\\text{dual}} c + d\\epsilon &= a + c  + (b + d)\\epsilon \\\\
    a + b\\epsilon \\otimes_{\\text{dual}} c + d\\epsilon &= ac  + (ad + bc)\\epsilon \\\\
\\end{align}
```
"""
struct DualNumberSemiring{T} <: Semiring{Tuple{T,T}}
    a::T
    b::T
end

val(x::DualNumberSemiring) = (x.a, x.b)
Base.:+(x::DualNumberSemiring, y::DualNumberSemiring) = DualNumberSemiring(x.a + y.a, x.b + y.b)
Base.:*(x::DualNumberSemiring, y::DualNumberSemiring) = DualNumberSemiring(x.a * y.a, x.a * y.b + x.b * y.a)
Base.inv(x::DualNumberSemiring) = DualNumberSemiring(inv(x.a), -x.b/x.a^2)

Base.zero(::Type{DualNumberSemiring{T}}) where T = DualNumberSemiring(zero(T), zero(T))
Base.one(::Type{DualNumberSemiring{T}}) where T = DualNumberSemiring(one(T), zero(T))

Base.isapprox(x::DualNumberSemiring, y::DualNumberSemiring; kwargs...) = Base.isapprox(x.a, y.a; kwargs...) && Base.isapprox(x.b, y.b; kwargs...)


"""
    EntropySemiring{T<:AbstractFloat,F} <: Semiring{T}

Entropy semiring: ``( (\\mathbb{R} \\cup \\{ -\\infty \\}) \\times \\mathbb{R}, \\oplus, \\otimes, (-\\infty, 0), (0, 0) )``
where
```math
\\begin{align}
    (x, a) \\oplus (y, b) = (\\log(e^x + e^y), a + b) \\\\
    (x, a) \\otimes (y, b) = (x + y, b e^x + a e^y) \\\\
\\end{align}
```
"""
struct EntropySemiring{T} <: Semiring{Tuple{T,T}}
    logp::T
    H::T
end

val(x::EntropySemiring) = (x.logp, x.H)
Base.:+(x::EntropySemiring, y::EntropySemiring) = EntropySemiring(logaddexp(x.logp, y.logp), x.H + y.H)
Base.:*(x::EntropySemiring, y::EntropySemiring) = EntropySemiring(x.logp + y.logp, xexpy(y.H, x.logp) + xexpy(x.H, y.logp))
Base.zero(::Type{EntropySemiring{T}}) where T = EntropySemiring(T(-Inf), zero(T))
Base.one(::Type{EntropySemiring{T}}) where T = EntropySemiring(T(0), zero(T))

Base.isapprox(x::EntropySemiring, y::EntropySemiring; kwargs...) = Base.isapprox(x.logp, y.logp; kwargs...) && Base.isapprox(x.H, y.H; kwargs...)


struct EntropySemiringTangent{T} <: ChainRulesCore.AbstractTangent
    logp::T
    H::T
end

val(Δx::EntropySemiringTangent) = Δx.logp, Δx.H

Base.zero(::Type{EntropySemiringTangent{T}}) where T = EntropySemiringTangent(zero(T), zero(T))
Base.zero(x::EntropySemiringTangent) = zero(typeof(x))

Base.:+(Δx::EntropySemiringTangent, Δy::EntropySemiringTangent) =
    EntropySemiringTangent(Δx.logp + Δy.logp, Δx.H + Δy.H)

Base.:*(Δx::EntropySemiringTangent, a) =
    EntropySemiringTangent(Δx.logp * a, Δx.H * a)
Base.:*(a, Δx::EntropySemiringTangent) =
    EntropySemiringTangent(a * Δx.logp, a * Δx.H)

Base.:+(x::S, Δx) where S<:EntropySemiring = S(x.logp + Δx.logp, x.H + Δx.H)
Base.:+(Δx, x::S) where S<:EntropySemiring = S(Δx.logp + x.logp, Δx.H + x.H)

function ChainRulesCore.rrule(S::Type{EntropySemiring}, x, y)
    semiring_pullback(Δs) = NoTangent(), Δs.logp, Δs.H
    S(x, y), semiring_pullback
end

function ChainRulesCore.rrule(::typeof(val), x::EntropySemiring{T}) where T
    val_pullback(Δx) = NoTangent(), EntropySemiringTangent{T}(Δx...)
    val(x), val_pullback
end


function ∂sum(z::EntropySemiring, x::EntropySemiring{T}, Δu::EntropySemiringTangent) where T
    EntropySemiringTangent{T}(
        z.logp == -Inf ? zero(T) : xexpy(Δu.logp, x.logp - z.logp),
        Δu.H
    )
end

function ∂rmul(x::EntropySemiring{T}, a::EntropySemiring, Δu::EntropySemiringTangent) where T
    EntropySemiringTangent{T}(
        Δu.logp + Δu.H * xexpy(a.H, x.logp),
        Δu.H * exp(a.logp)
    )
end

function ∂lmul(a::EntropySemiring, x::EntropySemiring{T}, Δu::EntropySemiringTangent) where T
    EntropySemiringTangent{T}(
        Δu.logp + Δu.H * xexpy(a.H, x.logp),
        Δu.H * exp(a.logp)
    )
end

tangent_type(::Type{EntropySemiring{T}}) where T = EntropySemiringTangent{T}
tangent_type(x::Semiring) = tangent_type(typeof(x))

