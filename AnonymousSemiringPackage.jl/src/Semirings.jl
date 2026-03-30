# SPDX-License-Identifier: CECILL-B

module Semirings

using ChainRulesCore
import LogExpFunctions: logaddexp, xexpy

#= Generic API =#
export Semiring, val, ∂sum, ∂rmul, ∂lmul, tangent_type

#= Concrete semiring type =#
export ArcticSemiring,
       BoolSemiring,
       DualNumberSemiring,
       EntropySemiring,
       ExtendedArcticSemiring,
       ExtendedTropicalSemiring,
       LogSemiring,
       NegativeLogSemiring,
       ProbSemiring,
       RealSemiring,
       ScaledLogSemiring,
       TropicalSemiring

#= Tangent types =#
export SemiringTangent,
       EntropySemiringTangent

include("semiring.jl")
include("types.jl")
include("morphisms.jl")

end

