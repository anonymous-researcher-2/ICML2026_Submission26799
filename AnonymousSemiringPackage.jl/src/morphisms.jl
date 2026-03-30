"""
    exp(x::LogSemiring{T})::RealSemiring{T}

Exponential morphism of the log-semiring.

See also [`Base.log`](@ref).
"""
Base.exp(x::LogSemiring{T}) where T = RealSemiring{T}(exp(val(x)))


"""
    log(x::RealSemiring{T})::LogSemiring{T}

Logarithm morphism of the probaility semiring.

See also [`Base.exp`](@ref).
"""
Base.log(x::RealSemiring{T}) where T = LogSemiring{T}(log(val(x)))

