# lau-banach-agents

> Contraction mappings and fixed-point theorems for provably convergent agent learning

## What This Does

This crate implements Banach fixed-point theory applied to agent learning. A `LearningAgent` uses contraction mappings — functions with Lipschitz constant < 1 — to iteratively update its state. Because Banach's theorem guarantees every contraction on a complete metric space has a unique fixed point, the agent's learning process provably converges regardless of starting state.

## The Key Idea

Most machine learning has no convergence guarantees. This crate provides one: the learning update rule is a contraction map (Lipschitz constant L < 1), meaning each step brings the state strictly closer to the unique fixed point. The rate is exponential: error shrinks by factor L each iteration. No learning rate tuning. No divergence. No local minima. The fixed point *is* the solution.

## Install

```toml
[dependencies]
lau-banach-agents = { git = "https://github.com/SuperInstance/lau-banach-agents" }
```

## Quick Start

```rust
use lau_banach_agents::*;
use nalgebra::DVector;

// Create a learning agent with contraction rate 0.5
let mut agent = LearningAgent::new(3, 0.5);

// Set initial state
agent.set_state(DVector::from_vec(vec![1.0, 2.0, 3.0]));

// Define the contraction map (update rule)
let update = |state: &DVector<f64>| -> DVector<f64> {
    state.scale(0.5) // Halving is a contraction with L = 0.5
};

// Iterate to convergence
let fixed_point = agent.iterate(update, 1e-12, 1000);
println!("Fixed point: {:?}", fixed_point);

// Verify contraction property
assert!(agent.verify_contraction(&update));
```

## API Reference

### `LearningAgent`

| Method | Description |
|--------|-------------|
| `new(dim, lipschitz)` | Create agent with state dimension and contraction rate L. |
| `set_state(state)` | Set current state. |
| `state()` | Get current state. |
| `iterate(f, tol, max_iter)` | Apply f until ‖f(x)-x‖ < tol. Returns fixed point. |
| `verify_contraction(f)` | Check that f is a contraction. |
| `convergence_rate()` | Rate = L^n (error after n iterations). |

### `ContractionMap`

| Method | Description |
|--------|-------------|
| `new(f, lipschitz)` | Wrap a closure as a verified contraction map. |
| `apply(x)` | Apply one iteration. |
| `fixed_point(x0, tol)` | Iterate to fixed point. |
| `lipschitz_constant()` | The contraction rate. |

## How It Works

1. **Contraction Definition**: A map f: X→X is a contraction if d(f(x), f(y)) ≤ L·d(x,y) for L < 1.
2. **Iteration**: Start from any x₀, compute xₙ₊₁ = f(xₙ).
3. **Convergence**: By Banach's theorem, xₙ → x* (unique fixed point) exponentially fast.
4. **Error Bound**: After n iterations, d(xₙ, x*) ≤ Lⁿ/(1-L) · d(x₀, x₁).

## The Math

- **Banach Fixed-Point Theorem**: Let (X, d) be a complete metric space and f: X→X a contraction with Lipschitz constant L < 1. Then f has a unique fixed point x* and the iteration xₙ₊₁ = f(xₙ) converges to x* for any x₀.
- **A Priori Error**: d(xₙ, x*) ≤ Lⁿ · d(x₀, x*).
- **A Posteriori Error**: d(xₙ, x*) ≤ L/(1-L) · d(xₙ, xₙ₋₁).

## Testing

96 tests covering contraction map verification, fixed-point convergence, error bounds, and edge cases.

## License

MIT
