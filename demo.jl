### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ ae08ba14-2c1a-11f1-ab5a-a1d86e88e20b
begin
	# Use the local virtual environment.
	using Pkg
	Pkg.activate("AnonymousSemiringPackage.jl")

	# Anonymized package for differentiable semring API.
	using AnonymousSemiringPackage

	# Extra packages from the Julia's ecosystem.
	using Zygote
	using LogExpFunctions
	using Plots
	using ChainRulesCore
	using LinearAlgebra
end

# ╔═╡ f8addabb-735e-4c42-9aab-24ae9bd214a9
md"""
# Demonstration of Automatic Differentiation of Semiring-based Computation

This notebook illustrate the use of the _morphism-trick_ introduced in "_Fast and General Automatic Differentiation for Finite-State Methods_", __submission 26799__ for __ICML 2026__.

!!! note
	This demonstration is limited to the semiring part of our open-source software (via the package `AnonymousSemiringPackage.jl`). It shows how the _morphism trick_ can help the implementation of general vector-Jacobian product to differentiate via "arbitrary" semirings.
"""

# ╔═╡ 68ca476e-b29d-49ed-85b2-eb5ebcac5745
md"""
The core idea of the _morphism-trick_ is to enable designing software libraries using arbitrary semirings and yet supporting fast differentiation without implementing dedicated routines for each possible semiring.

Our demonstration is structured as follows
1. Example of a toy library with abstract semiring computation and general vector-jacobian product implementation (with the _morphism-trick_);
2. Use of our toy library to perform tropical-curve fitting;
3. Demonstrate how to add a custom semiring that can be differentiated to.
"""

# ╔═╡ e7bc2fd1-52d4-41b6-a39b-1f70f8d38b46
md"""## 1. Semiring-Agnostic Library

Typical semiring-based libraries implement graph-based operations. The graphs can be of very different nature, _e.g._ knowledge graphs, finite state automata and neural networks. 

From high-level perspective, these libraries can be seen as sparse or dense linear algebra API  in various semiring.

In our example, our toy library will implement a unique function `muladd(S, W, b, X)` that computes
```math
	\mathbf{y}_i = \mathbf{W} \mathbf{x}_i + \mathbf{b}
```
for each column ``i`` of a matrix ``{X}``,  where ``S`` is an arbitrary semiring and ``\mathbf{W} \in S^{D\times K}``, ``\mathbf{b} \in S^K`` and ``\mathbf{X} \in S^{D \times N}``.

"""

# ╔═╡ c0962921-3e3c-49df-b8c0-09ebfc3101ce
md""" ## 2. Tropical-curve fitting


Here, we leverage our simple API (defined above) to fit a tropical curve. 

See [https://en.wikipedia.org/wiki/Tropical_geometry](https://en.wikipedia.org/wiki/Tropical_geometry) for information on tropical geometry.

### 2.1 Model 

Let ``\mathcal{T}`` be the tropical semiring, we consider data generated from a tropical-affine function:
```math
\begin{align}
	\mathbf{y} &= \mathbf{W} \mathbf{x} + \mathbf{b}  
\end{align}
```
where $\mathbf{x} \in \mathcal{T}^D$, $W \in \mathcal{T}^{K \times D}$, and $\mathbf{b} \in \mathcal{T}^K$.

"""

# ╔═╡ 318f57aa-545e-4fd3-80eb-e12d6411ccaa
md"""We generate some data by using random ``\mathbf{W}`` and ``\mathbf{b}``."""

# ╔═╡ 9884b875-badd-494e-8c78-2dea7e742b3f
md""" ### 2.2 Data
We sample data from a randomly generated ``\mathbf{W}`` and ``\mathbf{b}``.
"""

# ╔═╡ e117c120-c9da-467a-938a-11514151daf5
md""" ### 2.3 Curve fitting

For a set of input ``\mathbf{X}``, predicted output ``\hat{\mathbf{Y}}`` such that
```math
	\hat{\mathbf{y}}_i = \mathbf{W} \mathbf{x}_i + \mathbf{b}, 
```
and reference output ``\mathbf{Y}``, the loss function is (in the real semiring)
```math
	\mathcal{L}(\mathbf{W}, \mathbf{b}) = \sqrt{\sum_{i,j} |\hat{y}_{i,j} - y_{i,j}|^2}
```
"""

# ╔═╡ a230e4f5-3f4e-479e-8fb8-0ff0c6539c76
md"""
Training is achieved with with a gradient descent with learning rate ``\rho`` and a L2 norm regularization controlled by ``\beta``.
"""

# ╔═╡ f6d4349f-4a7e-42cf-a248-c5d5847b2517
md"""Parameter initialization."""

# ╔═╡ 70e43dcf-5d82-4132-9702-a264025d2406
md""" For the sake of the example, we train with 3 different semiring:
  * the real semiring (`RealSemiring{T}`)
  * the log semiring (`NegativeSemiring{T}`)
  * the tropical semiring (`ExtendedSemiring{T}`)

Not that we use the `extended tropical` to differentiate in the tropical semiring. __This does not require any change in the library defined above__, it is sufficient for the user to use the adequate semiring.
"""

# ╔═╡ ce457a16-9dee-4ab8-9da5-e91e5926463d
md"""
## 3. Implementing a new semiring

The semirings we have used are part of the `AnonymousSemiringPackage.jl` (code is present in the repository, see `AnonymousSemiringPackage.jl/src/types.jl`).

We show now, how a user can add another a differentiable semiring to the library. As an example, we implement a scaled version of the log semiring (as described in the paper in example 6.1)

### 3.1 User code
The first part is the implementation of the semiring itself. The user must implement the ``\oplus``, ``\otimes`` operators and the neutral elements ``\bar{0}`` and ``\bar{1}``.
"""

# ╔═╡ e5321ca3-3694-47c6-b74c-f828d8d34ed6
begin
	struct MySemiring{τ,T} <: Semiring{T}
	    val::T
	end

	MySemiring{τ}(x::T) where {τ,T} = MySemiring{τ,T}(x)

	function Base.:+(x::MySemiring{τ}, y::MySemiring{τ}) where τ
	    T = promote_type(valtype(x), valtype(y))
	    MySemiring{τ,T}(logaddexp(T(τ)*val(x), T(τ)*val(y))/T(τ))
	end

	Base.:*(x::MySemiring{τ}, y::MySemiring{τ}) where τ = MySemiring{τ}(val(x) + val(y))

	Base.zero(S::Type{MySemiring{τ,T}}) where {τ,T} = S(ifelse(τ > 0, T(-Inf), T(Inf)))
	Base.one(S::Type{MySemiring{τ,T}}) where {τ,T} = S(T(0))

	
end

# ╔═╡ 33f9a365-e98c-4967-a4f5-661b66fd954a
md"""
In addition, for `MySemiring` to be differentiable with our API, the user must define the 3 other functions corresponding to __the 3 terms in eq. 8 in the manuscript.__
"""

# ╔═╡ d2843dbe-1c89-4d9f-bece-95264a4035e2
begin
	function AnonymousSemiringPackage.∂sum(z::MySemiring{τ}, 
										   x::MySemiring{τ,T}, 
										   Δu::SemiringTangent) where {τ,T}
	    if iszero(z)
			# Special case if the sum is semiring-zero, then we set the 
			# derivative to 0.
	        SemiringTangent{T}(false)
	    else
	        SemiringTangent{T}(val(Δu) * exp(τ*(val(x) - val(z))))
	    end
	end

	AnonymousSemiringPackage.∂rmul(x::MySemiring{τ,T}, a::MySemiring{τ}, Δu::SemiringTangent) where {τ,T} = SemiringTangent{T}(Δu.val)
	AnonymousSemiringPackage.∂lmul(a::MySemiring{τ}, x::MySemiring{τ,T}, Δu::SemiringTangent) where {τ,T} = SemiringTangent{T}(Δu.val)
end

# ╔═╡ 2a30d904-da4b-4f35-9813-52c7b2655275
begin
	"""
		muladd(W, b, X)

	Compute `W * X .+ b` in an arbitrary semiring `S`.
	"""
	function muladd(W::Matrix{S}, b::Vector{S}, X::Matrix{S}) where S<:Semiring
		H = zeros(S, size(W, 1), size(X, 2))

		# This is obviously not optimized and simply serve as an example. 
		for n in 1:size(X, 2)
			for i in 1:size(W, 1)
				val = b[i]
				for j in 1:size(W, 2)
					
					val += W[i, j] * X[j, n] 
					
				end
				H[i, n] = val
			end
		end
	
		H
	end

	#=
	Vector-Jacobian product implementation of the `muladd` function, a.k.a.
	"reverse rule" in ChainRulesCore terminology.

	=> Note that the implementation is defined for an abstract type semiring `S`.
	=> The implementation relies on overloading `∂sum`, `∂lmul` and ∂rmul`
	=> An example is provided in Section 3 of the notebook.
	
	=#
	function ChainRulesCore.rrule(::typeof(muladd), 
								  W::Matrix{S}, 
								  b::Vector{S},
								  X::Matrix{S}) where S<:Semiring
		H = muladd(W, b, X)
		
		function muladd_pullback(ΔH)
			ΔW = zeros(tangent_type(S), size(W)...)
			Δb = zeros(tangent_type(S), size(b)...)
			
			for n in 1:size(X, 2)
				for i in 1:size(W, 1)
					# Computation stored in the forward path.
					z = H[i,n]

					# Differentiate the semiring-sum w.r.t. to `b` with the morphism trick.
					#====================================================#
					# Must be atomic in a parallel implementation 
					Δb[i] += ∂sum(z, b[i], ΔH[i,n]) 
					#====================================================#
					
					for j in 1:size(W, 2)
						# Reconstruct the product.
						W_X = W[i,j] * X[j,n] 

						# Differentiate the semiring-sum w.r.t. `W[i,j]` with the morphism trick.
						ΔW_X = ∂sum(z, W[i,j] * X[j,n], ΔH[i,n]) 

						# Backpropagate through the semiring-product
						#====================================================#
						# Must be atomic in a parallel implementation 
						ΔW[i,j] += ∂rmul(W[i,j], X[j,n], ΔW_X)
						#====================================================#
					end
				end
			end
			
			NoTangent(), ΔW, Δb, NoTangent()
		end
		H, muladd_pullback
	end

	#=
	The rest of the API is just to manage the conversion (and the differentiation)
	of array of FloatXX values to arrays of semiring values. 
	=#

	"""
		convert_array_type(T::Type, x) = convert(Array{T}, x)

	This alias is just to introduce hook in the gradient computation (with ChainRulesCore.jl) to have a fine control over the gradient type. 
	"""
	convert_array_type(T::Type, x::Array) = convert(Array{T}, x)

	function ChainRulesCore.rrule(::typeof(convert_array_type), T::Type{S}, x) where S<:Semiring
		convert_pullback(Δy) = NoTangent(), NoTangent(), val.(Δy)
		y = convert_array_type(T, x), convert_pullback
	end

	function ChainRulesCore.rrule(::typeof(convert_array_type), T::Type{N}, x::Array{S}) where {N<:Number,S<:Semiring}
		rconvert_pullback(Δy) = NoTangent(), NoTangent(), tangent_type(S).(Δy)
		convert_array_type(T, x), rconvert_pullback
	end
end

# ╔═╡ 43beccc0-ea3a-4e80-afd7-0dd71537050e
# Semiring-affine function using the API define above.
function semiring_affine(S::Type{<:Semiring}, W, b, X)	
	# Convert the arrays to the semiring `S` type
	W = convert_array_type(S, W)
	b = convert_array_type(S, b)
	X = convert_array_type(S, X)
	
	# Compute `W*X .+ b`, i.e. in the semiring. 
	Y = muladd(W, b, X)

	# Convert the matrix to FloatXX values.
	val.(Y)
end

# ╔═╡ 4dc521e7-e8f2-4074-90ac-3800fcffa299
begin
	# Input dimension
	D = 2  
	
	# Output dimension
	K = 2  

	# Input is the line `y = x`.
	X = stack([[i, i] for i in -4:0.1:4]) 

	# Generate data with random `W` and `b`.
	Y = semiring_affine(TropicalSemiring{Float32}, randn(K, D), randn(K), X) 

	# Number of training samples.
	N = size(X, 2) 

	p_x = plot(X[1,:], X[2,:], title="Input", label = "x", size = (400, 200))
	p_y = plot(Y[1,:], Y[2,:], title="Output", label = "y", size = (400, 200))

	plot(p_x, p_y)
end

# ╔═╡ fb875e9e-4ee7-471f-af0a-6a42a71a335a
begin
	W = randn(K, D)
	b = zeros(K) .-10
end

# ╔═╡ cabe010d-54b6-446a-99d7-2b6339fa7e2d
function loss(S::Type, W, b, X, Y)
	Y_pred = semiring_affine(S, W, b, X)

	# L2 norm
	norm(Y_pred - Y)
end

# ╔═╡ 58b71848-3ef6-4fb6-999b-ff285841b8eb
function train!(S, W, b, X, Y; epochs=1000, ρ=1e-2, β=1e-1)
	train_loss = []
	for e in 1:epochs
		(L, (∇W, ∇b)) = withgradient(W, b) do W, b
			# Reverse-model automatic differentiation happens here.
			# Zygote will callback our vector-Jacobian product implementations.
			loss(S, W, b, X, Y) + β*(norm(W) + norm(b))
		end
		push!(train_loss, L)
		
		W .-= ρ * ∇W
		b .-= ρ * ∇b
	end

	# Return the training losses and the final parameters `W` and `b`.
	train_loss, W, b
end

# ╔═╡ d2adce85-dfed-4cd0-8867-901f34fa5d65
begin
	loss_real, W_real, b_real = train!(RealSemiring{Float32}, copy(W), copy(b), X, Y)
	loss_log, W_log, b_log = train!(NegativeLogSemiring{Float32}, copy(W), copy(b), X, Y)
	loss_trop, W_trop, b_trop = train!(ExtendedTropicalSemiring{Float32}, copy(W), copy(b), X, Y)
end

# ╔═╡ bacab774-a840-44be-bd19-b33aeaa9dc90
begin
	plot(loss_real, lw=2, label="real", size = (400, 200), title="Training Loss")
	plot!(loss_log, lw=2, label="log")
	plot!(loss_trop, lw=2, label="tropical")
end

# ╔═╡ a6d22461-f327-4132-bff0-18ac4aa1d92f
begin
	local Y_pred = semiring_affine(RealSemiring{Float32}, W_real, b_real, X)
	p_real = scatter(Y[1,:], Y[2,:], title="real", label = "true", size=(600, 200))
	scatter!(Y_pred[1,:], Y_pred[2,:], label = "pred.", alpha=0.5)

	local Y_pred = semiring_affine(NegativeLogSemiring{Float32}, W_log, b_log, X)
	p_log = scatter(Y[1,:], Y[2,:], title="log", label = "true", size=(300, 200))
	scatter!(Y_pred[1,:], Y_pred[2,:], label = "pred.", alpha=0.5)

	local Y_pred = semiring_affine(TropicalSemiring{Float32}, W_trop, b_trop, X)
	p_trop = scatter(Y[1,:], Y[2,:], title="tropical", label = "true", size=(300, 200))
	scatter!(Y_pred[1,:], Y_pred[2,:], label = "pred.", alpha=0.5)

	plot(p_real, p_log, p_trop; layout=(1, 3))
end

# ╔═╡ 4cdd2230-a407-4b3c-bf52-9fc405d97149
md"""
### 3.2 Testing
We test with 3 different scales:
- ``\tau = -0.1``
- ``\tau = -1`` this is equivalent to the regular (negative) log-semiring
- ``\tau = -10`` this should be closer to the tropical-semiring (note that the tropical semiring is the limit of the log-semiring when ``\tau = -\infty``)
"""

# ╔═╡ 61f8d514-d10b-4fd5-9032-49b741359aa4
# τ = -0.1
loss_mysemiring_n01, W_mysemiring_n01, b_mysemiring_n01 = train!(MySemiring{-0.1,Float32}, copy(W), copy(b), X, Y)

# ╔═╡ 9e27b9b8-a085-4a91-ba0c-0dba5b0d2773
# τ = -1, this is equivalent to the (negative) log-semiring.
loss_mysemiring_n1, W_mysemiring_n1, b_mysemiring_n1 = train!(MySemiring{-1,Float32}, copy(W), copy(b), X, Y)

# ╔═╡ f8b59a85-a002-4fc6-8299-83491d569bdd
# τ = -10
loss_mysemiring_n10, W_mysemiring_n10, b_mysemiring_n10 = train!(MySemiring{-10,Float32}, copy(W), copy(b), X, Y)

# ╔═╡ 64a66be7-615e-45e6-ada0-b8bc407e53b2
begin
	local Y_pred = semiring_affine(MySemiring{-0.1,Float32}, W_mysemiring_n01, b_mysemiring_n01, X)
	p_mysemiring_n01 = scatter(Y[1,:], Y[2,:], title="log (tau = -0.1)", label = "true", size=(600, 200))
	scatter!(Y_pred[1,:], Y_pred[2,:], label = "pred.", alpha=0.5)

	local Y_pred = semiring_affine(MySemiring{-1,Float32}, W_mysemiring_n1, b_mysemiring_n1, X)
	p_mysemiring_n1 = scatter(Y[1,:], Y[2,:], title="log (tau = -1)", label = "true", size=(300, 200))
	scatter!(Y_pred[1,:], Y_pred[2,:], label = "pred.", alpha=0.5)

	local Y_pred = semiring_affine(MySemiring{-10,Float32}, W_mysemiring_n10, b_mysemiring_n10, X)
	p_mysemiring_n10 = scatter(Y[1,:], Y[2,:], title="log (tau = -10)", label = "true", size=(300, 200))
	scatter!(Y_pred[1,:], Y_pred[2,:], label = "pred.", alpha=0.5)

	plot(p_mysemiring_n01, p_mysemiring_n1, p_mysemiring_n10; layout=(1, 3))
end

# ╔═╡ Cell order:
# ╟─f8addabb-735e-4c42-9aab-24ae9bd214a9
# ╠═ae08ba14-2c1a-11f1-ab5a-a1d86e88e20b
# ╟─68ca476e-b29d-49ed-85b2-eb5ebcac5745
# ╟─e7bc2fd1-52d4-41b6-a39b-1f70f8d38b46
# ╠═2a30d904-da4b-4f35-9813-52c7b2655275
# ╟─c0962921-3e3c-49df-b8c0-09ebfc3101ce
# ╠═43beccc0-ea3a-4e80-afd7-0dd71537050e
# ╟─318f57aa-545e-4fd3-80eb-e12d6411ccaa
# ╟─9884b875-badd-494e-8c78-2dea7e742b3f
# ╠═4dc521e7-e8f2-4074-90ac-3800fcffa299
# ╟─e117c120-c9da-467a-938a-11514151daf5
# ╠═cabe010d-54b6-446a-99d7-2b6339fa7e2d
# ╟─a230e4f5-3f4e-479e-8fb8-0ff0c6539c76
# ╠═58b71848-3ef6-4fb6-999b-ff285841b8eb
# ╟─f6d4349f-4a7e-42cf-a248-c5d5847b2517
# ╠═fb875e9e-4ee7-471f-af0a-6a42a71a335a
# ╟─70e43dcf-5d82-4132-9702-a264025d2406
# ╠═d2adce85-dfed-4cd0-8867-901f34fa5d65
# ╠═bacab774-a840-44be-bd19-b33aeaa9dc90
# ╠═a6d22461-f327-4132-bff0-18ac4aa1d92f
# ╟─ce457a16-9dee-4ab8-9da5-e91e5926463d
# ╠═e5321ca3-3694-47c6-b74c-f828d8d34ed6
# ╟─33f9a365-e98c-4967-a4f5-661b66fd954a
# ╠═d2843dbe-1c89-4d9f-bece-95264a4035e2
# ╟─4cdd2230-a407-4b3c-bf52-9fc405d97149
# ╠═61f8d514-d10b-4fd5-9032-49b741359aa4
# ╠═9e27b9b8-a085-4a91-ba0c-0dba5b0d2773
# ╠═f8b59a85-a002-4fc6-8299-83491d569bdd
# ╠═64a66be7-615e-45e6-ada0-b8bc407e53b2
