# lau-banach-agents

> Banach space theory for agents — a functional analysis framework in Rust.

A crate that implements the pillars of **functional analysis** — Banach spaces, bounded operators, the big theorems (Hahn-Banach, Open Mapping, Closed Graph, Uniform Boundedness), dual spaces, spectral theory, and the Banach fixed-point theorem — then applies them to **iterative agent learning convergence**.

Every result is verified by 96 unit tests.

---

## What This Does

This crate gives you concrete, finite-dimensional implementations of abstract objects from functional analysis:

- **Banach spaces** with configurable norms (ℓ¹, ℓ², ℓᵖ, ℓ∞)
- **Bounded linear operators** between normed spaces, with operator norm computation
- **Contraction mappings** and the Banach fixed-point theorem with iterative solvers
- **The four pillar theorems**: Hahn-Banach extension & separation, Open Mapping, Closed Graph, Uniform Boundedness
- **Dual spaces**, annihilators, weak convergence, and reflexivity
- **Riesz representation** for Lᵖ duality, with Hölder's inequality
- **Spectral theory**: eigenvalues, spectral radius, normal/self-adjoint operators, resolvent
- **Agent learning**: models gradient descent as a contraction mapping, proving convergence via Banach's theorem

All structures serialize via `serde` and use `nalgebra` for linear algebra.

---

## Key Idea

The Banach fixed-point theorem says: *any contraction mapping on a complete metric space has exactly one fixed point, and iterating from any starting point converges to it.*

This crate treats **gradient-based learning** as exactly that: the update rule `θ_{n+1} = θ_n − ηGθ_n + η·target` is an affine contraction mapping. If the learning rate is small enough, the operator `(I − ηG)` has spectral radius < 1, so:

1. Learning converges to a **unique** optimum (no local minima ambiguity)
2. The **convergence rate** is the contraction constant k
3. After n steps, the error is bounded by `kⁿ / (1−k) · ‖x₁ − x₀‖`

The rest of the crate provides the mathematical infrastructure (norms, operators, duality, spectra) that makes this rigorous.

---

## Install

```toml
[dependencies]
lau-banach-agents = "0.1"
```

Or clone directly:

```bash
git clone https://github.com/SuperInstance/lau-banach-agents.git
cargo build
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` 0.33 | Linear algebra (vectors, matrices, SVD, eigenvalues) |
| `serde` 1 | Serialization of all mathematical structures |
| `num-complex` 0.4 | Complex eigenvalues in spectral theory |
| `approx` 0.5 (dev) | Floating-point assertions in tests |

---

## Quick Start

### Banach space elements

```rust
use lau_banach_agents::{BanachElement, NormType};

let v = BanachElement::new(vec![3.0, 4.0], NormType::L2);
assert!((v.norm() - 5.0).abs() < 1e-10);

let w = BanachElement::new(vec![1.0, 2.0], NormType::L2);
let sum = v.add(&w).unwrap();
assert!((sum.data[0] - 4.0).abs() < 1e-10);
```

### Contraction mapping → fixed point

```rust
use lau_banach_agents::{ContractionMapping, NormType};
use nalgebra::{DMatrix, DVector};

// T(x) = 0.5x + 1 → fixed point at x = 2
let m = DMatrix::from_row_slice(1, 1, &[0.5]);
let b = DVector::from_vec(vec![1.0]);
let c = ContractionMapping::new(m, b, NormType::L2);

assert!(c.is_contraction());

let result = c.find_fixed_point(&DVector::from_vec(vec![0.0]), 100, 1e-10);
assert!(result.converged);
assert!((result.fixed_point[0] - 2.0).abs() < 1e-8);
```

### Agent learning

```rust
use lau_banach_agents::LearningAgent;
use nalgebra::{DMatrix, DVector};

// Gradient descent on a 2D quadratic: G = diag(2,3), target = [1,1]
let g = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
let target = DVector::from_vec(vec![1.0, 1.0]);
let mut agent = LearningAgent::new(vec![0.0, 0.0], g, target, 0.1);

assert!(agent.is_contractive());
let result = agent.train(1000, 1e-10);
assert!(result.converged);
```

---

## API Reference

### `banach_space` — Complete Normed Vector Spaces

| Type / Function | Description |
|----------------|-------------|
| `NormType` | Enum: `L1`, `L2`, `LInf`, `Lp(f64)` |
| `BanachElement` | Vector + norm type. Methods: `new`, `norm`, `add`, `sub`, `scale`, `distance`, `zero`, `dimension` |
| `compute_norm(v, norm_type)` | Compute ‖v‖ under any `NormType` |
| `CauchySequence` | Sequence of `BanachElement`s. Methods: `is_cauchy(ε)`, `limit()` |
| `triangle_inequality(x, y)` | Verify ‖x+y‖ ≤ ‖x‖ + ‖y‖ |
| `absolute_homogeneity(x, α)` | Verify ‖αx‖ = |α|·‖x‖ |
| `positive_definiteness(x)` | Verify ‖x‖ = 0 ⟺ x = 0 |
| `verify_completeness(seq, ε)` | Check Cauchy sequence convergence |

### `bounded_operator` — Linear Maps Between Banach Spaces

| Type / Function | Description |
|----------------|-------------|
| `BoundedOperator` | Matrix + domain/codomain norms. Methods: `apply`, `apply_element`, `operator_norm(samples)`, `operator_norm_l2()`, `compose`, `add`, `scale`, `identity`, `zero`, `verify_linearity`, `verify_boundedness` |

`operator_norm_l2()` uses SVD (exact). `operator_norm(samples)` samples unit vectors for general norms.

### `fixed_point` — Banach Fixed-Point Theorem

| Type / Function | Description |
|----------------|-------------|
| `ContractionMapping` | Affine map T(x) = Ax + b. Methods: `apply`, `contraction_constant`, `is_contraction`, `find_fixed_point`, `exact_fixed_point` |
| `FixedPointResult` | Fixed point, iterations, final error, convergence flag, full error history |
| `verify_fixed_point_theorem(c, tol, max_iter)` | Verify contraction converges to a true fixed point |
| `verify_uniqueness(c, initial_points, tol, max_iter)` | Verify all starting points reach the same fixed point |

The exact fixed point solves `(I − A)⁻¹b` analytically. The iterative solver gives error history for convergence analysis.

### `hahn_banach` — Functional Extension & Separation

| Type / Function | Description |
|----------------|-------------|
| `LinearFunctional` | f(x) = cᵀx. Methods: `evaluate`, `dual_norm`, `extend(dim, bound)` |
| `ConvexSet` | Intersection of half-spaces {x : aᵢᵀx ≤ bᵢ}. Method: `contains` |
| `separate_convex_sets(a_center, b_center)` | Hahn-Banach separation: returns separating hyperplane (f, α) |
| `verify_extension(original, extended, dim)` | Verify extension preserves norm and agrees on subspace |

Dual norms: ‖f‖_{L1→dual} = ‖c‖_∞, ‖f‖_{L2→dual} = ‖c‖_2, ‖f‖_{L∞→dual} = ‖c‖_1.

### `open_mapping` — Surjective Bounded Operators Are Open Maps

| Type / Function | Description |
|----------------|-------------|
| `verify_open_mapping(op)` → `OpenMappingResult` | Checks surjectivity (full row rank), computes min singular value, determines if open map |
| `OpenMappingResult` | Fields: `is_surjective`, `rank`, `is_open_map`, `min_singular_value` |
| `bounded_inverse(op)` → `Option<f64>` | Corollary: if T is bijective, returns ‖T⁻¹‖ = 1/σ_min |

### `closed_graph` — Closed Graph ↔ Bounded

| Type / Function | Description |
|----------------|-------------|
| `GraphOperator` | Operator with graph G(T) = {(x, Tx)}. Methods: `apply`, `graph_matrix`, `is_graph_closed`, `is_bounded`, `verify_closed_graph_theorem` |
| `ClosedGraphResult` | Fields: `graph_closed`, `bounded`, `operator_norm`, `theorem_holds` |

For finite-dimensional operators, the graph is always closed (linear subspaces are closed), and all linear operators are bounded. The theorem holds trivially but the machinery is correct.

### `uniform_boundedness` — Pointwise Bounded ⟹ Uniformly Bounded

| Type / Function | Description |
|----------------|-------------|
| `OperatorFamily` | Vec of `BoundedOperator`s. Methods: `is_pointwise_bounded(test_vectors, bound)`, `uniform_bound(samples)`, `verify_uniform_boundedness(samples)` |
| `UniformBoundednessResult` | Fields: `num_operators`, `uniform_bound`, `is_uniformly_bounded` |
| `verify_finite_dim_uniform_boundedness()` | Demonstrates the principle in finite dimensions |

### `dual_space` — Bounded Linear Functionals

| Type / Function | Description |
|----------------|-------------|
| `DualSpace` | Represents X*. Methods: `functional(coeffs)`, `dual_norm_type`, `dual_dimension`, `canonical_embedding(x, f)`, `is_reflexive` |
| `annihilator(basis, dim, norm_type)` | Compute M⊥ = {f ∈ X* : f(m)=0 ∀m ∈ M} via null-space of basis matrix |

Duality rules: (L¹)* = L∞, (L²)* = L², (L∞)* = L¹, (Lᵖ)* = Lᵍ with 1/p + 1/q = 1.

### `weak_topology` — Weak vs. Strong Convergence

| Type / Function | Description |
|----------------|-------------|
| `WeakConvergence` | Check weak convergence against test functionals. Method: `check_weak_convergence(limit, functionals, ε)` |
| `check_strong_convergence(seq, limit, ε)` | ‖x_n − x‖ → 0 |
| `strong_implies_weak(seq, limit, funcs, ε)` | Verify: strong ⟹ weak |
| `weak_not_implies_strong_example()` | Orthonormal basis: converges weakly to 0 but ‖e_n‖ = 1 |

### `riesz_representation` — Lᵖ Duality

| Type / Function | Description |
|----------------|-------------|
| `RieszRepresentation` | Methods: `conjugate_exponent(p)`, `represent(f)`, `verify_norm_equality()` |
| `verify_lp_duality(p, coeffs)` | Full verification of (Lᵖ)* = Lᵍ |
| `holder_inequality(x, y, p)` | Verify |⟨x,y⟩| ≤ ‖x‖_p · ‖y‖_q |

### `spectral` — Spectrum, Spectral Radius, Normal Operators

| Type / Function | Description |
|----------------|-------------|
| `SpectralAnalysis` | Methods: `analyze(matrix)`, `verify_spectral_radius_formula(mat, max_power)`, `verify_spectral_bound()`, `is_normal(mat)`, `is_self_adjoint(mat)`, `spectrum()`, `is_in_resolvent(λ)`, `verify_normal_spectral_equality()` |
| `SpectralAnalysis` fields | `eigenvalues: Vec<Complex64>`, `spectral_radius: f64`, `operator_norm: f64` |

Key results verified: r(T) ≤ ‖T‖ always; for normal operators r(T) = ‖T‖.

### `agent_learning` — Convergence via Banach Fixed-Point Theorem

| Type / Function | Description |
|----------------|-------------|
| `LearningAgent` | Agent with parameters θ learning via T(θ) = (I−ηG)θ + η·target. Methods: `new`, `is_contractive`, `step`, `train`, `optimal_parameters` |
| `MultiAgentSystem` | Multiple agents. Methods: `all_converge`, `joint_contraction_rate`, `train_all` |
| `verify_convergence_theorem(agent, max_iter, tol)` → `ConvergenceVerification` | Verifies contraction, convergence, error bound, and monotone convergence |
| `ConvergenceVerification` | Fields: `is_contraction`, `converged`, `error_bound_holds`, `rate_monotone` |

---

## How It Works

### Architecture

```
banach_space ──→ bounded_operator ──→ fixed_point ──→ agent_learning
     │                  │
     ├── hahn_banach ───┤
     ├── dual_space ────┤
     ├── weak_topology ─┤
     ├── riesz_representation
     ├── open_mapping
     ├── closed_graph
     ├── uniform_boundedness
     └── spectral
```

The dependency graph flows from foundations (norms, vectors) up through operators and theorems to the application layer.

### Finite-Dimensional Realization

This crate works in **finite-dimensional** spaces (ℝⁿ), where:
- All norms are equivalent (so completeness is automatic)
- All linear operators are bounded (closed graph holds trivially)
- The dual space has the same dimension as the primal

This is deliberate: it lets us compute everything exactly (SVD, eigenvalue decomposition, matrix inversion) while still demonstrating the *structure* of the theorems. The results extend to infinite-dimensional Banach spaces conceptually.

### Operator Norm via SVD

For the ℓ² norm, the operator norm ‖T‖ is the **largest singular value** of the matrix T. This is computed via `nalgebra`'s SVD. For other norms, the crate samples unit basis vectors and all-ones vectors to get an estimate.

### Spectral Radius Formula

The spectral radius r(T) = max|λ| satisfies the formula:

```
r(T) = lim_{n→∞} ‖Tⁿ‖^{1/n}
```

The crate verifies this numerically by computing matrix powers and checking convergence.

---

## The Math

### Banach Spaces

A **Banach space** (X, ‖·‖) is a complete normed vector space. The norm must satisfy:

1. **Positive definiteness**: ‖x‖ ≥ 0, with equality iff x = 0
2. **Absolute homogeneity**: ‖αx‖ = |α|·‖x‖
3. **Triangle inequality**: ‖x + y‖ ≤ ‖x‖ + ‖y‖

**Completeness** means every Cauchy sequence converges within the space. In finite dimensions, this is automatic.

### Banach Fixed-Point Theorem

If T: X → X is a **contraction** (i.e., ‖T(x) − T(y)‖ ≤ k·‖x − y‖ for some k < 1), then:

1. T has a **unique** fixed point x* with T(x*) = x*
2. For any x₀, the iteration x_{n+1} = T(x_n) converges to x*
3. The error satisfies: ‖x_n − x*‖ ≤ kⁿ/(1−k) · ‖x₁ − x₀‖

For an affine map T(x) = Ax + b, the contraction condition is ‖A‖ < 1 (operator norm), and the fixed point is (I − A)⁻¹b.

### The Four Pillar Theorems

These are the "crown jewels" of functional analysis, all interconnected:

- **Hahn-Banach**: Extend bounded functionals from subspaces while preserving the norm. Consequence: separate disjoint convex sets with a hyperplane.
- **Open Mapping**: A surjective bounded operator between Banach spaces is an open map. Consequence: bijective bounded operators have bounded inverses (‖T⁻¹‖ = 1/σ_min).
- **Closed Graph**: A linear operator between Banach spaces is bounded iff its graph is closed. (In finite dimensions, always true.)
- **Uniform Boundedness**: If a family of operators is pointwise bounded, it's uniformly bounded: sup_i ‖T_i‖ < ∞.

### Duality and Weak Topology

The **dual space** X* consists of all bounded linear functionals f: X → ℝ. The **dual norm** is ‖f‖ = sup_{‖x‖=1} |f(x)|.

Key duality: (Lᵖ)* = Lᵍ where 1/p + 1/q = 1 (Hölder conjugates). For p = 2, this gives (L²)* = L² (Hilbert space self-duality).

**Weak convergence** x_n ⇀ x means f(x_n) → f(x) for all f ∈ X*. Strong convergence implies weak convergence, but not vice versa (e.g., orthonormal sequences converge weakly to zero but not strongly).

### Spectral Theory

For a bounded operator T, the **spectrum** σ(T) = {λ ∈ ℂ : T − λI is not invertible}. The **spectral radius** r(T) = max{|λ| : λ ∈ σ(T)}.

Key relations:
- r(T) ≤ ‖T‖ always
- r(T) = ‖T‖ when T is **normal** (T*T = TT*)
- r(T) = lim ‖Tⁿ‖^{1/n} (spectral radius formula)

For real symmetric matrices, the spectrum is real and the operator norm equals the spectral radius.

---

## License

MIT
