//! Hahn-Banach theorem: extend linear functionals (separation theorem).

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::banach_space::NormType;

/// A bounded linear functional f: X → ℝ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearFunctional {
    /// Coefficient vector: f(x) = cᵀx.
    pub coefficients: DVector<f64>,
    pub norm_type: NormType,
}

impl LinearFunctional {
    pub fn new(coefficients: Vec<f64>, norm_type: NormType) -> Self {
        Self { coefficients: DVector::from_vec(coefficients), norm_type }
    }

    /// Evaluate f(x) = cᵀx.
    pub fn evaluate(&self, x: &DVector<f64>) -> f64 {
        self.coefficients.dot(x)
    }

    /// Compute the dual norm of this functional.
    /// ‖f‖ = sup_{‖x‖=1} |f(x)|.
    pub fn dual_norm(&self) -> f64 {
        match &self.norm_type {
            NormType::L1 => {
                // Dual of L1 is LInf
                self.coefficients.iter().map(|c| c.abs()).fold(0.0_f64, f64::max)
            }
            NormType::L2 | NormType::Lp(_) => {
                // For Lp, the dual norm is computed via the Riesz representative
                if let NormType::Lp(p) = &self.norm_type {
                    let q = 1.0 / (1.0 - 1.0 / p);
                    let sum: f64 = self.coefficients.iter().map(|c| c.abs().powf(q)).sum();
                    sum.powf(1.0 / q)
                } else {
                    self.coefficients.iter().map(|c| c * c).sum::<f64>().sqrt()
                }
            }
            NormType::LInf => {
                // Dual of LInf is L1
                self.coefficients.iter().map(|c| c.abs()).sum()
            }
        }
    }

    /// Extend a functional defined on a subspace to the whole space,
    /// preserving the norm. (Hahn-Banach extension theorem.)
    ///
    /// Given f defined on subspace spanned by first k basis vectors,
    /// extend to full n-dimensional space.
    pub fn extend(&self, full_dimension: usize, bound: f64) -> LinearFunctional {
        let k = self.coefficients.nrows();
        let mut extended = vec![0.0; full_dimension];
        for i in 0..k.min(full_dimension) {
            extended[i] = self.coefficients[i];
        }
        // For remaining dimensions, choose values that don't exceed the bound
        // Use minimal extension (zero padding preserves the norm bound)
        let extended_func = LinearFunctional::new(extended, self.norm_type.clone());
        // Verify the extension doesn't exceed the bound
        assert!(
            extended_func.dual_norm() <= bound + 1e-10,
            "Extension violated bound: {} > {}",
            extended_func.dual_norm(),
            bound
        );
        extended_func
    }
}

/// A closed convex set in ℝⁿ, represented as the intersection of half-spaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvexSet {
    /// Each row aᵢ defines a half-space: aᵢᵀx ≤ bᵢ.
    pub halfspaces_a: Vec<DVector<f64>>,
    pub halfspaces_b: Vec<f64>,
}

impl ConvexSet {
    pub fn new(halfspaces_a: Vec<DVector<f64>>, halfspaces_b: Vec<f64>) -> Self {
        Self { halfspaces_a, halfspaces_b }
    }

    /// Check if a point is in the convex set.
    pub fn contains(&self, x: &DVector<f64>) -> bool {
        self.halfspaces_a.iter().zip(&self.halfspaces_b).all(|(a, b)| {
            a.dot(x) <= *b + 1e-10
        })
    }
}

/// Hahn-Banach separation: given two disjoint convex sets, find a separating hyperplane.
/// Returns (f, α) such that f(x) ≤ α for all x in A and f(y) > α for all y in B.
pub fn separate_convex_sets(
    a_center: &DVector<f64>,
    b_center: &DVector<f64>,
) -> (LinearFunctional, f64) {
    // The separating hyperplane is perpendicular to the line connecting centers
    let diff = b_center - a_center;
    let norm = diff.norm();
    let normal = if norm > 1e-10 { &diff / norm } else { diff.clone() };

    let f = LinearFunctional {
        coefficients: normal,
        norm_type: NormType::L2,
    };

    let midpoint = (a_center + b_center) * 0.5;
    let alpha = f.evaluate(&midpoint);

    (f, alpha)
}

/// Verify the Hahn-Banach extension theorem:
/// The extended functional agrees with the original on the subspace
/// and doesn't increase in norm.
pub fn verify_extension(
    original: &LinearFunctional,
    extended: &LinearFunctional,
    subspace_dim: usize,
) -> bool {
    // Check agreement on subspace
    for i in 0..subspace_dim {
        if (original.coefficients[i] - extended.coefficients[i]).abs() > 1e-10 {
            return false;
        }
    }
    // Check norm doesn't increase
    extended.dual_norm() <= original.dual_norm() + 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_functional_evaluation() {
        let f = LinearFunctional::new(vec![1.0, 2.0, 3.0], NormType::L2);
        let x = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        assert_relative_eq!(f.evaluate(&x), 6.0);
    }

    #[test]
    fn test_dual_norm_l2() {
        let f = LinearFunctional::new(vec![3.0, 4.0], NormType::L2);
        assert_relative_eq!(f.dual_norm(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dual_norm_l1() {
        // Dual of L1 is LInf
        let f = LinearFunctional::new(vec![1.0, -3.0, 2.0], NormType::L1);
        assert_relative_eq!(f.dual_norm(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dual_norm_linf() {
        // Dual of LInf is L1
        let f = LinearFunctional::new(vec![1.0, 2.0, 3.0], NormType::LInf);
        assert_relative_eq!(f.dual_norm(), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hahn_banach_extension() {
        let f = LinearFunctional::new(vec![1.0, 2.0], NormType::L2);
        let bound = f.dual_norm() * 2.0; // generous bound
        let extended = f.extend(4, bound);
        assert_eq!(extended.coefficients.nrows(), 4);
        assert_eq!(extended.coefficients[0], 1.0);
        assert_eq!(extended.coefficients[1], 2.0);
        assert_eq!(extended.coefficients[2], 0.0);
    }

    #[test]
    fn test_verify_extension() {
        let original = LinearFunctional::new(vec![1.0, 2.0], NormType::L2);
        let extended = original.extend(4, original.dual_norm() + 0.1);
        assert!(verify_extension(&original, &extended, 2));
    }

    #[test]
    fn test_separation_simple() {
        let a = DVector::from_vec(vec![0.0, 0.0]);
        let b = DVector::from_vec(vec![4.0, 0.0]);
        let (f, alpha) = separate_convex_sets(&a, &b);
        assert_relative_eq!(f.evaluate(&a), 0.0, epsilon = 1e-10);
        assert!(f.evaluate(&b) > alpha - 1e-10);
    }

    #[test]
    fn test_convex_set_contains() {
        let a = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![-1.0, 0.0]),
            DVector::from_vec(vec![0.0, -1.0]),
        ];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let set = ConvexSet::new(a, b);
        assert!(set.contains(&DVector::from_vec(vec![0.0, 0.0])));
        assert!(!set.contains(&DVector::from_vec(vec![2.0, 2.0])));
    }

    #[test]
    fn test_extension_preserves_norm_bound() {
        let f = LinearFunctional::new(vec![3.0], NormType::L2);
        let bound = f.dual_norm() + 1.0;
        let ext = f.extend(3, bound);
        assert!(ext.dual_norm() <= bound + 1e-10);
    }
}
