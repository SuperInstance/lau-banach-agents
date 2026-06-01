# lau-banach-agents

**Banach space theory for agents — a functional analysis framework in Rust.**

A crate that turns the pillars of functional analysis — Banach's fixed-point theorem, Hahn–Banach separation, the open mapping theorem, the closed graph theorem, the uniform boundedness principle, Riesz representation, and spectral theory — into composable, tested Rust abstractions. The final module ties everything together with a concrete application: proving convergence guarantees for iterative agent learning via contraction mappings.

96 tests. Zero dependencies beyond `nalgebra`, `serde`, and `num-complex`.

---

## What This Does

| Module | Mathematical Result | What You Get |
|---|---|---|
| `banach_space` | Complete normed vector spaces | `BanachElement` with L¹, L², L∞, Lᵖ norms, Cauchy sequences, completeness verification |
| `bounded_operator` | Bounded linear operators ‖T‖ = sup‖Tx‖/‖x‖ | Matrix-backed operators, operator norms (sampling & SVD), linearity & boundedness checks, composition |
| `fixed_point` | Banach Fixed-Point Theorem | Contraction mappings, iterative fixed-point solver, exact analytical solution, uniqueness verification |
| `hahn_banach` | Hahn–Banach Extension & Separation | Linear functionals, norm-preserving extension, convex-set separation via hyperplanes |
| `open_mapping` | Open Mapping Theorem | Surjectivity check, open-map verification, bounded-inverse corollary |
| `closed_graph` | Closed Graph Theorem | Graph matrix construction, closed-graph ↔ bounded equivalence |
| `uniform_boundedness` | Uniform Boundedness Principle | Operator families, pointwise vs. uniform boundedness, finite-dimensional guarantee |
| `dual_space` | Dual Spaces X*, Annihilators | Dual norm computation (Lᵖ↔Lᵍ duality), canonical embedding, reflexivity, annihilator computation |
| `weak_topology` | Weak vs. Strong Convergence | Weak convergence testing, strong→weak implication, orthonormal counterexample |
| `riesz_representation` | Riesz Representation for Lᵖ | Conjugate exponent computation, norm-equality verification, Hölder's inequality |
| `spectral` | Spectral Radius, Normal Operators | Eigenvalue computation, spectral radius formula r(T) = lim‖Tⁿ‖^{1/n}, normal/self-adjoint checks, resolvent set |
| `agent_learning` | Application: Convergence of Iterative Learning | `LearningAgent` (gradient-descent-as-contraction), `MultiAgentSystem`, convergence-theorem verification |

---

## Key Idea

The big insight: **Banach's fixed-point theorem gives you a convergence proof for free.**

If you can show your learning operator is a contraction (spectral radius < 1), the theorem guarantees:

1. **Convergence** — the iteration reaches a fixed point.
2. **Uniqueness** — it doesn't matter where you start.
3. **Rate bound** — error decreases geometrically as kⁿ.

This crate builds that bridge: pure math theorems → composable Rust types → agent learning guarantees.

```rust
use lau_banach_agents::*;

// A learning agent doing gradient descent on a strongly convex function
let gradient_matrix = nalgebra::DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
let target = nalgebra::DVector::from_vec(vec![1.0, 1.0]);
let mut agent = LearningAgent::new(
    vec![0.0, 0.0],  // initial parameters
    gradient_matrix,
    target,
    0.1,              // learning rate
);

// The theorem guarantees convergence
assert!(agent.is_contractive());

// Train and converge
let result = agent.train(1000, 1e-10);
assert!(result.converged);
```

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-banach-agents = "0.1"
```

Or use it directly:

```bash
git clone https://github.com/SuperInstance/lau-banach-agents.git
cd lau-banach-agents
cargo build
```

### Dependencies

- **nalgebra** 0.33 — linear algebra (matrices, vectors, SVD, eigenvalues)
- **serde** 1 — serialization of all types
- **num-complex** 0.4 — complex eigenvalues in spectral theory
- **approx** 0.5 (dev) — floating-point assertions in tests

---

## Quick Start

### Creating a Banach Space Element

```rust
use lau_banach_agents::{BanachElement, NormType};

let v = BanachElement::new(vec![3.0, 4.0], NormType::L2);
assert!((v.norm() - 5.0).abs() < 1e-10);  // Euclidean norm

let w = BanachElement::new(vec![3.0, -4.0], NormType::L1);
assert!((w.norm() - 7.0).abs() < 1e-10);  // Manhattan norm
```

### Bounded Linear Operators

```rust
use lau_banach_agents::{BoundedOperator, NormType};
use nalgebra::{DMatrix, DVector};

let matrix = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 4.0]);
let op = BoundedOperator::new(matrix, NormType::L2, NormType::L2);

// Exact operator norm via SVD
assert!((op.operator_norm_l2() - 4.0).abs() < 1e-10);

// Verify linearity: T(αx + βy) = αT(x) + βT(y)
let x = DVector::from_vec(vec![1.0, 0.0]);
let y = DVector::from_vec(vec![0.0, 1.0]);
assert!(op.verify_linearity(&x, &y, 2.0, 3.0));
```

### Fixed-Point Iteration

```rust
use lau_banach_agents::{ContractionMapping, NormType};
use nalgebra::{DMatrix, DVector};

// T(x) = 0.5x + 1 → fixed point at x = 2
let matrix = DMatrix::from_row_slice(1, 1, &[0.5]);
let bias = DVector::from_vec(vec![1.0]);
let contraction = ContractionMapping::new(matrix, bias, NormType::L2);

assert!(contraction.is_contraction());

let x0 = DVector::from_vec(vec![0.0]);
let result = contraction.find_fixed_point(&x0, 100, 1e-10);
assert!(result.converged);
assert!((result.fixed_point[0] - 2.0).abs() < 1e-8);

// Exact solution: (I - A)^{-1} b
let exact = contraction.exact_fixed_point().unwrap();
```

### Spectral Analysis

```rust
use lau_banach_agents::SpectralAnalysis;
use nalgebra::{DMatrix, DVector};

let matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 3.0]));
let analysis = SpectralAnalysis::analyze(&matrix);

assert!((analysis.spectral_radius - 3.0).abs() < 1e-10);  // max|λ|
assert!(analysis.verify_spectral_bound());                  // r(T) ≤ ‖T‖

// Verify spectral radius formula: r(T) = lim ‖Tⁿ‖^{1/n}
assert!(analysis.verify_spectral_radius_formula(&matrix, 20));
```

### Hahn–Banach Separation

```rust
use lau_banach_agents::separate_convex_sets;
use nalgebra::DVector;

let set_a_center = DVector::from_vec(vec![0.0, 0.0]);
let set_b_center = DVector::from_vec(vec![4.0, 0.0]);

let (functional, threshold) = separate_convex_sets(&set_a_center, &set_b_center);
// functional(x) ≤ threshold for x in A, functional(y) > threshold for y in B
```

---

## API Reference

### `banach_space`

| Type / Function | Description |
|---|---|
| `NormType` | Enum: `L1`, `L2`, `LInf`, `Lp(f64)` |
| `BanachElement` | Vector + norm. Methods: `norm()`, `add()`, `sub()`, `scale()`, `distance()`, `zero()` |
| `CauchySequence` | Sequence of `BanachElement`s. Methods: `is_cauchy()`, `limit()` |
| `compute_norm(v, norm_type)` | Compute norm of a `DVector` |
| `triangle_inequality(x, y)` | Verify ‖x+y‖ ≤ ‖x‖ + ‖y‖ |
| `absolute_homogeneity(x, α)` | Verify ‖αx‖ = |α|‖x‖ |
| `positive_definiteness(x)` | Verify ‖x‖ = 0 ⟺ x = 0 |
| `verify_completeness(seq, ε)` | Check Cauchy sequence converges |

### `bounded_operator`

| Type / Function | Description |
|---|---|
| `BoundedOperator` | Matrix + domain/codomain norms. Methods: `apply()`, `operator_norm()`, `operator_norm_l2()`, `compose()`, `add()`, `scale()`, `verify_linearity()`, `verify_boundedness()` |
| `BoundedOperator::identity(n, norm)` | Identity operator |
| `BoundedOperator::zero(m, n, ...)` | Zero operator |

### `fixed_point`

| Type / Function | Description |
|---|---|
| `ContractionMapping` | Affine map T(x) = Ax + b. Methods: `apply()`, `contraction_constant()`, `is_contraction()`, `find_fixed_point()`, `exact_fixed_point()` |
| `FixedPointResult` | Fixed point, iterations, error, convergence flag, error history |
| `verify_fixed_point_theorem(c, tol, max)` | End-to-end verification |
| `verify_uniqueness(c, starts, tol, max)` | Different starting points → same fixed point |

### `hahn_banach`

| Type / Function | Description |
|---|---|
| `LinearFunctional` | f(x) = cᵀx. Methods: `evaluate()`, `dual_norm()`, `extend(dim, bound)` |
| `ConvexSet` | Intersection of half-spaces. Method: `contains(x)` |
| `separate_convex_sets(a, b)` | Returns separating hyperplane (functional, threshold) |
| `verify_extension(orig, ext, k)` | Check extension agrees on subspace & preserves norm |

### `open_mapping`

| Type / Function | Description |
|---|---|
| `verify_open_mapping(op)` → `OpenMappingResult` | Checks surjectivity, rank, openness, min singular value |
| `bounded_inverse(op)` → `Option<f64>` | Computes ‖T⁻¹‖ = 1/σ_min if bijective |

### `closed_graph`

| Type / Function | Description |
|---|---|
| `GraphOperator` | Operator with graph G(T) = {(x, Tx)}. Methods: `graph_matrix()`, `is_graph_closed()`, `verify_closed_graph_theorem()` |
| `ClosedGraphResult` | Graph closed, bounded, operator norm, theorem holds |

### `uniform_boundedness`

| Type / Function | Description |
|---|---|
| `OperatorFamily` | Collection of `BoundedOperator`s. Methods: `is_pointwise_bounded()`, `uniform_bound()`, `verify_uniform_boundedness()` |
| `verify_finite_dim_uniform_boundedness()` | Demonstrates finite-dim guarantee |

### `dual_space`

| Type / Function | Description |
|---|---|
| `DualSpace` | Represents X*. Methods: `functional()`, `dual_norm_type()`, `dual_dimension()`, `canonical_embedding()`, `is_reflexive()` |
| `annihilator(basis, dim, norm)` | Compute M⊥ = {f ∈ X* : f|_M = 0} |

### `weak_topology`

| Type / Function | Description |
|---|---|
| `WeakConvergence` | Check f(xₙ) → f(x) for all test functionals |
| `check_strong_convergence(seq, limit, ε)` | ‖xₙ - x‖ → 0 |
| `strong_implies_weak(...)` | Verify implication |
| `weak_not_implies_strong_example()` | Orthonormal basis: weak→0, strong↛0 |

### `riesz_representation`

| Type / Function | Description |
|---|---|
| `RieszRepresentation` | Conjugate exponent + representing element. Method: `verify_norm_equality()` |
| `verify_lp_duality(p, coeffs)` | End-to-end (Lᵖ)* = Lᵍ check |
| `holder_inequality(x, y, p)` | Verify Hölder's inequality |

### `spectral`

| Type / Function | Description |
|---|---|
| `SpectralAnalysis` | Eigenvalues, spectral radius, operator norm. Methods: `verify_spectral_radius_formula()`, `verify_spectral_bound()`, `is_normal()`, `is_self_adjoint()`, `spectrum()`, `is_in_resolvent()`, `verify_normal_spectral_equality()` |

### `agent_learning`

| Type / Function | Description |
|---|---|
| `LearningAgent` | Parameters + contraction-based learning. Methods: `is_contractive()`, `step()`, `train()`, `optimal_parameters()` |
| `MultiAgentSystem` | Multiple `LearningAgent`s. Methods: `all_converge()`, `joint_contraction_rate()`, `train_all()` |
| `verify_convergence_theorem(agent, max, tol)` | Verify contraction → convergence + error bounds |

---

## How It Works

### Architecture

The crate is layered:

```
agent_learning        ← Application layer (convergence guarantees)
    ↑
fixed_point           ← Core theorem (Banach fixed-point)
spectral              ← Eigenvalue analysis (contraction check)
bounded_operator      ← Operator theory
banach_space          ← Foundation (norms, vectors, completeness)
    ↑
hahn_banach  dual_space  riesz_representation  weak_topology   ← Duality & separation
open_mapping  closed_graph  uniform_boundedness                ← The "big three" theorems
```

### Operator Norms

For L² norms, the operator norm is computed exactly via SVD (largest singular value). For other norms, a deterministic sampling strategy evaluates candidate unit vectors and takes the maximum.

### Fixed-Point Solver

The iterative solver computes x_{n+1} = T(x_n) until ‖x_{n+1} - x_n‖ < tolerance. For the affine case T(x) = Ax + b, the exact solution (I - A)⁻¹b is also available via matrix inversion.

### Agent Learning

The update rule θ_{n+1} = (I - ηG)θ_n + η·target is a contraction when the learning rate η is small enough relative to the gradient operator G. The contraction constant is the operator norm of (I - ηG), which must be < 1.

---

## The Math

### Banach Spaces

A **Banach space** is a complete normed vector space (X, ‖·‖). The norm satisfies:

- **Positive definiteness**: ‖x‖ = 0 ⟺ x = 0
- **Absolute homogeneity**: ‖αx‖ = |α|‖x‖
- **Triangle inequality**: ‖x + y‖ ≤ ‖x‖ + ‖y‖

**Completeness** means every Cauchy sequence converges within the space. In finite dimensions, all norms are equivalent and the space is automatically complete.

### Banach Fixed-Point Theorem

If T: X → X is a **contraction** (‖T(x) - T(y)‖ ≤ k‖x - y‖ for some k < 1), then:

1. T has a **unique** fixed point x* with T(x*) = x*
2. For any starting point x₀, the iteration x_{n+1} = T(x_n) converges to x*
3. Error bound: ‖x_n - x*‖ ≤ kⁿ/(1-k) · ‖x₁ - x₀‖

### The Big Three Theorems

These three results form the backbone of functional analysis:

- **Hahn–Banach**: Linear functionals defined on a subspace can be extended to the whole space without increasing the norm. Corollary: disjoint convex sets can be separated by a hyperplane.
- **Open Mapping**: Surjective bounded operators between Banach spaces are open maps. Corollary: bijective bounded operators have bounded inverses.
- **Closed Graph**: An operator between Banach spaces is bounded if and only if its graph is closed.

### Uniform Boundedness Principle

If a family {Tᵢ} of bounded operators is **pointwise bounded** (for each x, supᵢ ‖Tᵢx‖ < ∞), then it is **uniformly bounded** (supᵢ ‖Tᵢ‖ < ∞). This fails in infinite dimensions but always holds in finite dimensions.

### Duality and Riesz Representation

The **dual space** X* consists of all bounded linear functionals on X. For Lᵖ spaces, (Lᵖ)* = Lᵍ where 1/p + 1/q = 1 (Hölder conjugates). The Riesz representation theorem identifies each functional f(x) = ∫x(t)g(t)dt with a unique g ∈ Lᵍ satisfying ‖f‖ = ‖g‖ᵍ.

### Spectral Theory

The **spectrum** σ(T) is the set of λ ∈ ℂ where (T - λI) is not invertible. For matrices, these are eigenvalues. The **spectral radius** r(T) = max|λ| satisfies:

- r(T) = lim_{n→∞} ‖Tⁿ‖^{1/n} (spectral radius formula)
- r(T) ≤ ‖T‖ always
- r(T) = ‖T‖ for normal operators (T*T = TT*)

### Weak Topology

A sequence xₙ → x **weakly** if f(xₙ) → f(x) for every f ∈ X*. Strong convergence implies weak convergence, but not vice versa (orthonormal bases converge weakly to 0 but have constant norm 1).

---

## License

MIT
