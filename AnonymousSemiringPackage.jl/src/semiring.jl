"""
    Semiring{T}

Abstract Semiring type. `T` is the type of the value wrapped by the semiring
structure (e.g. Float32, Bool, ...).
"""
abstract type Semiring{T} end


"""
    val(x::Semiring)

Return the value wrapped by the semiring type.
If `x` is a subtype of `Number` behaves like the identity function.
"""
val(x::Semiring) = x.val

Base.zero(x::Semiring) = zero(typeof(x))
Base.one(x::Semiring) = one(typeof(x))


#= Integer multiplication =#

function Base.:*(i::Integer, s::Semiring)
    i < 0 && throw(ArgumentError("integer must be positive"))
    iszero(i) && return zero(s)

    power = 1
    sum_power = 1
    total = s
    x = s
    while sum_power < i
        total += x
        sum_power += power
        power *= 2
        if power < (i - sum_power)
            x = x+x
        else
            power = 1
            x = s
        end
    end
    total
end
Base.:*(s::Semiring, i::Integer) = i * s


#= Integer exponentiation =#

function Base.:^(s::Semiring, i::Integer)
    i < 0 && throw(ArgumentError("integer must be non-negative"))
    iszero(i) && return one(s)

    power = 1
    sum_power = 1
    total = s
    x = s
    while sum_power < i
        total *= x
        sum_power += power
        power *= 2
        if power < (i - sum_power)
            x = x*x
        else
            power = 1
            x = s
        end
    end
    total
end


#= Iteration interface =#
Base.valtype(::Type{<:Semiring{T}}) where T = T
Base.valtype(x::Semiring) = valtype(typeof(x))
Base.eltype(S::Type{<:Semiring}) = S
Base.length(s::Semiring) = 1
Base.size(s::Semiring) = ()
Base.iterate(x::Semiring) = iterate(val(x))
Base.iterate(x::Semiring, state) = iterate(val(x), state)


#= Type conversion =#
Base.convert(T::Type{<:Semiring}, x::Number) = T(x)
Base.convert(T::Type{<:Number}, x::Semiring) = T(val(x))
Base.convert(T::Type{<:Semiring}, x::Semiring) = T(valtype(T)(val(x)))
#(T::Type{<:Semiring})(x::Semiring) = T(val(x))


#= Comparison =#
Base.isless(x::Semiring, y::Semiring) = isless(val(x), val(y))
Base.:(==)(x::Semiring, y::Semiring) = val(x) == val(y)
Base.isequal(x::Semiring, y::Semiring) = isequal(val(x), val(y))
Base.isapprox(x::Semiring, y::Semiring; kwargs...) = Base.isapprox(val(x), val(y); kwargs...)


#= Rounding =#
Base.round(x::S; kwargs...) where S<:Semiring{<:Number} = S(round(val(x); kwargs...))


#= Automatic differentiation interface =#

# This is the default tangent type for semiring. Compound semiring, e.g.
# EntropySemiring needs to create their own types to match their specific
# fields.

struct SemiringTangent{T} <: ChainRulesCore.AbstractTangent
    val::T
end

val(x::SemiringTangent) = x.val

Base.zero(::Type{SemiringTangent{T}}) where T = SemiringTangent(zero(T))
Base.zero(x::SemiringTangent) = zero(typeof(x))

Base.:+(Δx::SemiringTangent, Δy::SemiringTangent) = SemiringTangent(Δx.val + Δy.val)

Base.:*(Δx::SemiringTangent, a) = SemiringTangent(Δx.val * a)
Base.:*(a, Δx::SemiringTangent) = SemiringTangent(a * Δx.val)

Base.:+(x::S, Δx) where S<:Semiring = S(x.val + Δx.val)
Base.:+(Δx, x::S) where S<:Semiring = S(Δx.val + x.val)

function ChainRulesCore.rrule(S::Type{<:Semiring}, x)
    semiring_pullback(Δx) = NoTangent(), Δx.val
    S(x), semiring_pullback
end

function ChainRulesCore.rrule(::typeof(val), x::Semiring)
    v = val(x)
    val_pullback(Δx) = NoTangent(), SemiringTangent(Δx)
    v, val_pullback
end

function ChainRulesCore.rrule(::typeof(Base.:+), x::Semiring, y::Semiring)
    z = x + y
    add_pullback(Δz) = NoTangent(), ∂sum(z, x, Δz), ∂sum(z, y, Δz)
    z, add_pullback
end

function ChainRulesCore.rrule(::typeof(Base.:*), x::Semiring, y::Semiring)
    z = x * y
    mul_pullback(Δz) = NoTangent(), ∂rmul(x, y, Δz), ∂lmul(x, y, Δz)
    z, mul_pullback
end

"""
    ∂sum(u::S, x::S, Δu) where S<:Semiring

Compute ``\\Delta u \\cdot \\frac{\\partial u}{\\partial x}`` where
``u = x \\oplus y \\oplus z \\oplus ...``.

!!! warning
    This method is implemented only for semirings for which there is a
    invertible morphism such that ``\\mu(u) = \\mu(x) + \\mu(y) + \\mu(z) + ...`` where ``+``
    is the natural addition.
"""
∂sum

"""
    ∂rmul(x::S, a::S, Δu) where S<:Semiring

Compute ``\\Delta u \\cdot \\frac{\\partial u}{\\partial x}`` where ``u = x \\otimes a``.
"""
∂rmul

"""
    ∂lmul(a::S, x::S, Δu) where S<:Semiring

Compute ``\\Delta u \\cdot \\frac{\\partial u}{\\partial x}`` where ``u = a \\otimes x``.
"""
∂lmul


"""
    tangent_type(::Type{S}) where S<:Semiring

Return the tangent type for `S`.
"""
tangent_type(::Type{<:Semiring{T}}) where T = SemiringTangent{T}

