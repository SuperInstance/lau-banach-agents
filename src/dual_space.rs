//! Dual space: B(X*) = space of bounded linear functionals.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::banach_space::NormType;
use crate::hahn_banach::LinearFunctional;

/// The dual space X* of a Banach space X.
/// Consists of all bounded linear functionals f: X → ℝ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualSpace {
    pub dimension: usize,
    pub norm_type: NormType,
}

impl DualSpace {
    pub fn new(dimension: usize, norm_type: NormType) -> Self {
        Self { dimension, norm_type }
    }

    /// A functional in the dual space.
    pub fn functional(&self, coefficients: Vec<f64>) -> LinearFunctional {
        assert_eq!(coefficients.len(), self.dimension);
        LinearFunctional::new(coefficients, self.dual_norm_type())
    }

    /// The dual norm type.
    /// (Lᵖ)* = Lᵍ where 1/p + 1/q = 1, with convention L∞* = L1, L1* = L∞.
    pub fn dual_norm_type(&self) -> NormType {
        match &self.norm_type {
            NormType::L1 => NormType::LInf,
            NormType::L2 => NormType::L2, // Hilbert space: self-dual
            NormType::LInf => NormType::L1,
            NormType::Lp(p) => {
                let q = 1.0 / (1.0 - 1.0 / p);
                NormType::Lp(q)
            }
        }
    }

    /// Dimension of the dual space (same as X for finite-dimensional spaces).
    pub fn dual_dimension(&self) -> usize {
        self.dimension
    }

    /// The canonical embedding J: X → X** (double dual).
    /// J(x)(f) = f(x).
    pub fn canonical_embedding(&self, x: &DVector<f64>, f: &LinearFunctional) -> f64 {
        f.evaluate(x)
    }

    /// Reflexivity check: X is reflexive if the canonical embedding is surjective.
    /// All finite-dimensional spaces are reflexive.
    pub fn is_reflexive(&self) -> bool {
        true // Finite-dimensional spaces are always reflexive
    }
}

/// Annihilator of a subspace M ⊂ X: M⊥ = {f ∈ X* : f(m) = 0 for all m ∈ M}.
pub fn annihilator(
    subspace_basis: &[DVector<f64>],
    dimension: usize,
    norm_type: NormType,
) -> Vec<LinearFunctional> {
    // Find all f such that f(b) = 0 for all basis vectors b.
    // f(b) = c^T b = 0 means c is orthogonal to all basis vectors.
    if subspace_basis.is_empty() {
        // Annihilator is the whole dual space
        return (0..dimension)
            .map(|i| {
                let mut coeffs = vec![0.0; dimension];
                coeffs[i] = 1.0;
                LinearFunctional::new(coeffs, norm_type.clone())
            })
            .collect();
    }

    // Build matrix B where rows are basis vectors
    let k = subspace_basis.len();
    let n = dimension;
    let mut b_matrix = vec![0.0; k * n];
    for (i, v) in subspace_basis.iter().enumerate() {
        for j in 0..n {
            b_matrix[i * n + j] = v[j];
        }
    }
    let b = nalgebra::DMatrix::from_row_slice(k, n, &b_matrix);

    // Find null space of B (vectors c such that Bc = 0)
    let rank = b.rank(1e-10);
    let svd = b.svd(true, true);
    let nullity = n - rank;

    // Approximate: return basis for null space
    let mut annihilators = Vec::new();
    for col in 0..nullity {
        let mut coeffs = vec![0.0; n];
        // Use right singular vectors corresponding to zero singular values
        if let Some(vt) = &svd.v_t {
            for row in 0..n {
                coeffs[row] = vt[(n - nullity + col, row)];
            }
        }
        annihilators.push(LinearFunctional::new(coeffs, norm_type.clone()));
    }

    annihilators
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dual_space_dimension() {
        let dual = DualSpace::new(3, NormType::L2);
        assert_eq!(dual.dual_dimension(), 3);
    }

    #[test]
    fn test_dual_norm_l2() {
        let dual = DualSpace::new(3, NormType::L2);
        assert_eq!(dual.dual_norm_type(), NormType::L2);
    }

    #[test]
    fn test_dual_norm_l1() {
        let dual = DualSpace::new(3, NormType::L1);
        assert_eq!(dual.dual_norm_type(), NormType::LInf);
    }

    #[test]
    fn test_dual_norm_linf() {
        let dual = DualSpace::new(3, NormType::LInf);
        assert_eq!(dual.dual_norm_type(), NormType::L1);
    }

    #[test]
    fn test_functional_in_dual() {
        let dual = DualSpace::new(3, NormType::L2);
        let f = dual.functional(vec![1.0, 2.0, 3.0]);
        let x = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        assert_relative_eq!(f.evaluate(&x), 1.0);
    }

    #[test]
    fn test_reflexive() {
        let dual = DualSpace::new(5, NormType::L2);
        assert!(dual.is_reflexive());
    }

    #[test]
    fn test_canonical_embedding() {
        let dual = DualSpace::new(2, NormType::L2);
        let x = DVector::from_vec(vec![3.0, 4.0]);
        let f = dual.functional(vec![1.0, 0.0]);
        let val = dual.canonical_embedding(&x, &f);
        assert_relative_eq!(val, 3.0);
    }

    #[test]
    fn test_annihilator_empty() {
        let an = annihilator(&[], 3, NormType::L2);
        assert_eq!(an.len(), 3);
    }
}
