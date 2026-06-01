# lau-banach-agents

> Complete normed spaces, bounded operators, and fixed-point theorems for convergent agent learning.

## What This Does

This crate provides a functional analysis framework built around Banach space theory. It implements complete normed vector spaces with Lᵖ norms, bounded linear operators with operator norms, the Banach fixed point theorem for contraction mappings, and the three pillars of linear operator theory: the Hahn-Banach theorem, open mapping theorem, and closed graph theorem. It also covers dual spaces, weak topology, Riesz representation, spectral theory, and an application to iterative agent learning convergence.

Use this when you need rigorous guarantees about convergence, boundedness, or existence of solutions in normed spaces — especially for agent learning systems where you need to prove that iterative updates converge.

## The Key Idea

A Banach space is a vector space where you can measure distances (a norm) and every Cauchy sequence converges (completeness). This combination is surprisingly powerful: it guarantees that fixed-point iterations converge (Banach fixed point theorem), linear operators are continuous if and only if they're bounded, and the three fundamental theorems (Hahn-Banach, open mapping, closed graph) hold. For agent learning, this means: if your update rule is a contraction mapping, convergence to a unique optimal policy is guaranteed.

## Install

```bash
cargo add lau-banach-agents
```

## Quick Start

```rust
use lau_banach_agents::*;
use nalgebra::{DVector, DMatrix};

fn main() {
    // Create elements in different norms
    let x = BanachElement::new(vec![1.0, 2.0, 3.0], NormType::L2);
    let y = BanachElement::new(vec![4.0, 5.0, 6.0], NormType::L2);
    println!("‖x‖₂ = {:.4}", x.norm());           // 3.7417
    println!("‖x+y‖₂ = {:.4}", x.add(&y).unwrap().norm());

    // Contraction mapping: T(x) = Ax + b, with ‖A‖ < 1
    let a = DMatrix::from_row_slice(2, 2, &[0.5, 0.1, 0.0, 0.3]);
    let b = DVector::from_vec(vec![1.0, 2.0]);
    let contraction = ContractionMapping::new(a, b, NormType::L2);
    println!("Is contraction: {}", contraction.is_contraction()); // true

    // Find the unique fixed point
    let initial = DVector::from_vec(vec![0.0, 0.0]);
    let result = contraction.find_fixed_point(&initial, 1000, 1e-10);
    println!("Converged in {} steps: {:?}", result.iterations, result.fixed_point);

    // Agent learning: gradient descent as contraction
    let g = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
    let target = DVector::from_vec(vec![1.0, 1.0]);
    let mut agent = LearningAgent::new(vec![0.0, 0.0], g, target, 0.1);
    let train_result = agent.train(1000, 1e-10);
    println!("Agent converged: {} in {} steps", train_result.converged, train_result.iterations);
}
```

## API Reference

### `BanachElement`
A vector in a Banach space, equipped with a norm.

```rust
let x = BanachElement::new(vec![1.0, 2.0, 3.0], NormType::L2);
let z = BanachElement::zero(3, NormType::L2);

x.norm();                    // ‖x‖
x.add(&y).unwrap();         // x + y
x.sub(&y).unwrap();         // x - y
x.scale(2.0);               // 2x
x.distance(&y).unwrap();    // ‖x - y‖
x.dimension();              // number of components
```

### `NormType`
Supported norms: `L1`, `L2`, `LInf`, `Lp(f64)`.

```rust
NormType::L1       // ‖x‖₁ = Σ|xᵢ|
NormType::L2       // ‖x‖₂ = √(Σxᵢ²)
NormType::LInf     // ‖x‖∞ = max|xᵢ|
NormType::Lp(3.0)  // ‖x‖₃ = (Σ|xᵢ|³)^{1/3}
```

### Norm Property Verification

```rust
triangle_inequality(&x, &y);       // ‖x+y‖ ≤ ‖x‖ + ‖y‖
absolute_homogeneity(&x, 3.0);     // ‖αx‖ = |α|‖x‖
positive_definiteness(&x);          // ‖x‖ = 0 ⟺ x = 0
```

### `CauchySequence`
A Cauchy sequence in a Banach space.

```rust
let seq = CauchySequence::new(elements);
seq.is_cauchy(1e-6);   // check if tail elements are within ε
seq.limit();            // last element as approximation
verify_completeness(&seq, 1e-6);  // completeness check
```

### `BoundedOperator`
A bounded linear operator T: X → Y with ‖T‖ < ∞.

```rust
let op = BoundedOperator::new(matrix, NormType::L2, NormType::L2);

op.apply(&x);                           // T(x)
op.operator_norm_l2();                   // ‖T‖ via max singular value
op.operator_norm(100);                   // ‖T‖ via sampling
op.compose(&other).unwrap();             // S ∘ T
op.verify_linearity(&x, &y, α, β);      // T(αx+βy) = αTx+βTy
op.verify_boundedness(&x);              // ‖Tx‖ ≤ ‖T‖‖x‖

BoundedOperator::identity(n, norm);      // I
BoundedOperator::zero(m, n, d, c);      // 0
```

### `ContractionMapping`
T(x) = Ax + b with Lipschitz constant k < 1.

```rust
let cm = ContractionMapping::new(a_matrix, b_vector, NormType::L2);
cm.apply(&x);                         // T(x)
cm.contraction_constant();             // ‖A‖ (operator norm)
cm.is_contraction();                   // k < 1?

let result = cm.find_fixed_point(&x0, 1000, 1e-10);
// result.fixed_point, result.iterations, result.final_error, result.converged

let exact = cm.exact_fixed_point();    // (I-A)⁻¹ b if invertible
```

### `FixedPointResult`

```rust
pub struct FixedPointResult {
    pub fixed_point: DVector<f64>,
    pub iterations: usize,
    pub final_error: f64,
    pub converged: bool,
    pub error_history: Vec<f64>,
}
```

### Hahn-Banach Theorem

#### `LinearFunctional`
A bounded linear functional f: X → ℝ.

```rust
let f = LinearFunctional::new(vec![1.0, 2.0, 3.0], NormType::L2);
f.evaluate(&x);          // f(x) = ⟨c, x⟩
f.dual_norm();           // ‖f‖ (depends on primal norm)
let extended = f.extend(higher_dim, bound);  // Hahn-Banach extension
```

#### `separate_convex_sets`
Find a separating hyperplane between two convex sets.

```rust
let (f, alpha) = separate_convex_sets(&center_a, &center_b);
// f(x) ≤ α for x ∈ A, f(y) > α for y ∈ B
```

#### `verify_extension`
Check that the extension agrees on the subspace and preserves the norm bound.

### Open Mapping Theorem

```rust
let result = verify_open_mapping(&operator);
// result.is_surjective, result.is_open_map, result.rank, result.min_singular_value

let inv_norm = bounded_inverse(&operator);  // ‖T⁻¹‖ if T is bijective
```

### Closed Graph Theorem

```rust
let gop = GraphOperator::new(matrix, domain_norm, codomain_norm);
gop.graph_matrix();           // [I; A]
gop.is_graph_closed();        // always true for linear operators
let result = gop.verify_closed_graph_theorem();
// result.graph_closed, result.bounded, result.theorem_holds
```

### Uniform Boundedness Principle

```rust
let family = OperatorFamily::new(operators);
family.is_pointwise_bounded(&test_vectors, bound);
family.uniform_bound(samples);
let result = family.verify_uniform_boundedness(samples);
```

### Dual Spaces

#### `DualSpace`
The space of bounded linear functionals on X.

```rust
let dual = DualSpace::new(3, NormType::L2);
dual.dual_norm_type();         // L2 (Hilbert space is self-dual)
dual.dual_dimension();         // same as primal for finite-dim
dual.is_reflexive();           // always true for finite-dim
dual.canonical_embedding(&x, &f);  // J: X → X**
```

#### `annihilator`
M⊥ = {f ∈ X* : f(m) = 0 for all m ∈ M}.

```rust
let annihilators = annihilator(&subspace_basis, dimension, norm_type);
```

### Weak Topology

```rust
let wc = WeakConvergence::new(sequence);
wc.check_weak_convergence(&limit, &functionals, epsilon);

check_strong_convergence(&sequence, &limit, epsilon);
strong_implies_weak(&sequence, &limit, &functionals, epsilon);

// Example: orthonormal basis converges weakly to 0 but not strongly
let (seq, limit) = weak_not_implies_strong_example();
```

### Riesz Representation

```rust
let rep = RieszRepresentation::represent(&functional);
rep.conjugate_exponent;           // q where 1/p + 1/q = 1
rep.verify_norm_equality();       // ‖f‖ = ‖g‖_q

verify_lp_duality(p, &coefficients);

// Hölder's inequality: |⟨x,y⟩| ≤ ‖x‖ₚ · ‖y‖_q
holder_inequality(&x, &y, p);
```

### Spectral Theory

```rust
let analysis = SpectralAnalysis::analyze(&matrix);
analysis.eigenvalues;              // complex eigenvalues
analysis.spectral_radius;          // max|λ|
analysis.operator_norm;            // ‖T‖

analysis.verify_spectral_radius_formula(&matrix, 20);  // r(T) = lim ‖Tⁿ‖^{1/n}
analysis.verify_spectral_bound();   // r(T) ≤ ‖T‖
SpectralAnalysis::is_normal(&matrix);    // T*T = TT*
SpectralAnalysis::is_self_adjoint(&matrix);  // T = T*
analysis.is_in_resolvent(lambda);   // λ ∉ σ(T)?
```

### Agent Learning

#### `LearningAgent`
An agent whose learning rule is a contraction mapping.

```rust
let mut agent = LearningAgent::new(initial_params, gradient_matrix, target, learning_rate);
agent.is_contractive();     // does the update rule converge?
agent.step();               // one iteration, returns error
let result = agent.train(1000, 1e-10);  // train to convergence
agent.optimal_parameters(); // exact solution (I-ηG)⁻¹ η·target
```

#### `MultiAgentSystem`

```rust
let mut system = MultiAgentSystem::new(agents);
system.all_converge();              // all agents contractive?
system.joint_contraction_rate();    // max contraction constant
let results = system.train_all(1000, 1e-10);
```

#### `verify_convergence_theorem`

```rust
let v = verify_convergence_theorem(&agent, 1000, 1e-10);
// v.is_contraction, v.converged, v.error_bound_holds, v.rate_monotone
```

## How It Works

**Banach elements** are `nalgebra::DVector<f64>` with an attached `NormType`. The four norm types (L1, L2, L∞, Lp) all satisfy the norm axioms (triangle inequality, absolute homogeneity, positive definiteness), verified explicitly.

**Bounded operators** are matrices with domain and codomain norms. The operator norm ‖T‖ = sup_{‖x‖=1} ‖Tx‖ is computed via SVD for the L2 case (exact: max singular value) or by sampling unit vectors for general norms.

**Contraction mappings** T(x) = Ax + b have ‖A‖ < 1. The fixed point is found by iterating x_{n+1} = T(x_n), which converges geometrically at rate ‖A‖. The exact fixed point (I−A)⁻¹b is also available when I−A is invertible.

**Hahn-Banach** extends a functional from a subspace to the whole space without increasing the dual norm. Separation finds a hyperplane between convex sets using the perpendicular to the line connecting their centers.

**Open mapping** checks surjectivity via rank, and computes the bounded inverse norm as 1/σ_min.

**Closed graph** constructs the graph matrix [I; A] and verifies that for linear operators, closed graph ⟺ bounded.

**Uniform boundedness** checks that pointwise bounded families of operators have a uniform norm bound.

**Dual spaces** implement the Lp duality (Lp)* = Lq, the canonical embedding X → X**, and annihilator computation via SVD null space.

**Spectral analysis** uses `nalgebra`'s `complex_eigenvalues()` for eigenvalue decomposition, verifies the spectral radius formula r(T) = lim ‖Tⁿ‖^{1/n}, and checks normal/self-adjoint properties.

**Agent learning** models gradient descent as θ_{n+1} = (I − ηG)θ_n + η·target, which is a contraction when η is small enough relative to the spectral radius of G.

## The Math

### Norm Axioms

A norm on a vector space X satisfies:
1. ‖x‖ ≥ 0, with ‖x‖ = 0 ⟺ x = 0
2. ‖αx‖ = |α|·‖x‖
3. ‖x + y‖ ≤ ‖x‖ + ‖y‖

### Banach Fixed Point Theorem

If (X, ‖·‖) is complete and T: X → X satisfies ‖T(x) − T(y)‖ ≤ k·‖x − y‖ for some k < 1, then T has a unique fixed point x*, and:

$$x^* = \lim_{n \to \infty} T^n(x_0), \quad \|x_n - x^*\| \leq \frac{k^n}{1-k} \|x_1 - x_0\|$$

### Operator Norm

$$\|T\| = \sup_{\|x\|=1} \|Tx\|$$

### Hahn-Banach Theorem

Every bounded linear functional on a subspace M ⊂ X can be extended to X without increasing its norm.

### Open Mapping Theorem

If T: X → Y is a surjective bounded linear operator between Banach spaces, then T maps open sets to open sets.

### Closed Graph Theorem

A linear operator T: X → Y between Banach spaces is bounded if and only if its graph G(T) = {(x, Tx)} is closed in X × Y.

### Uniform Boundedness Principle

If {T_i} is a family of bounded operators and sup_i ‖T_i x‖ < ∞ for each x, then sup_i ‖T_i‖ < ∞.

### Riesz Representation (Lᵖ Duality)

Every f ∈ (Lᵖ)* has the form f(x) = ∫ x(t)g(t) dt for some g ∈ Lᵍ where 1/p + 1/q = 1.

### Hölder's Inequality

$$|\langle x, y \rangle| \leq \|x\|_p \cdot \|y\|_q \quad \text{where } \frac{1}{p} + \frac{1}{q} = 1$$

### Spectral Radius Formula

$$r(T) = \max|\sigma(T)| = \lim_{n \to \infty} \|T^n\|^{1/n}$$

For normal operators: r(T) = ‖T‖.

## License

MIT
