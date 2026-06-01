//! Banach fixed point theorem: contractions have unique fixed points.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::banach_space::{BanachElement, NormType, compute_norm};
use crate::bounded_operator::BoundedOperator;

/// Result of a fixed point iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPointResult {
    /// The fixed point (or approximation).
    pub fixed_point: DVector<f64>,
    /// Number of iterations taken.
    pub iterations: usize,
    /// Final error ‖x_{n+1} - x_n‖.
    pub final_error: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// History of errors.
    pub error_history: Vec<f64>,
}

/// A contraction mapping T with Lipschitz constant k < 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractionMapping {
    /// The affine map: T(x) = A*x + b.
    pub matrix: DMatrix<f64>,
    pub bias: DVector<f64>,
    pub norm_type: NormType,
}

impl ContractionMapping {
    pub fn new(matrix: DMatrix<f64>, bias: DVector<f64>, norm_type: NormType) -> Self {
        Self { matrix, bias, norm_type }
    }

    /// Apply the contraction: T(x) = Ax + b.
    pub fn apply(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.matrix * x + &self.bias
    }

    /// Compute the contraction constant (operator norm of the matrix part).
    /// For T to be a contraction, this must be < 1.
    pub fn contraction_constant(&self) -> f64 {
        let op = BoundedOperator::new(self.matrix.clone(), self.norm_type.clone(), self.norm_type.clone());
        op.operator_norm_l2()
    }

    /// Check if this is actually a contraction.
    pub fn is_contraction(&self) -> bool {
        self.contraction_constant() < 1.0
    }

    /// Find the fixed point by iteration: x_{n+1} = T(x_n).
    /// Banach fixed point theorem guarantees convergence.
    pub fn find_fixed_point(
        &self,
        initial: &DVector<f64>,
        max_iterations: usize,
        tolerance: f64,
    ) -> FixedPointResult {
        let mut x = initial.clone();
        let mut error_history = Vec::new();
        let mut converged = false;
        let mut iterations = 0;

        for i in 0..max_iterations {
            let x_next = self.apply(&x);
            let diff = &x_next - &x;
            let error = compute_norm(&diff, &self.norm_type);
            error_history.push(error);
            iterations = i + 1;

            if error < tolerance {
                converged = true;
                x = x_next;
                break;
            }
            x = x_next;
        }

        let final_error = error_history.last().copied().unwrap_or(0.0);

        FixedPointResult {
            fixed_point: x,
            iterations,
            final_error,
            converged,
            error_history,
        }
    }

    /// Compute the exact fixed point analytically: (I - A)^{-1} * b.
    pub fn exact_fixed_point(&self) -> Option<DVector<f64>> {
        let n = self.matrix.nrows();
        let identity = DMatrix::identity(n, n);
        let i_minus_a = &identity - &self.matrix;
        i_minus_a.try_inverse().map(|inv| inv * &self.bias)
    }
}

/// Verify Banach fixed point theorem: contraction mapping has unique fixed point.
pub fn verify_fixed_point_theorem(
    contraction: &ContractionMapping,
    tolerance: f64,
    max_iterations: usize,
) -> bool {
    if !contraction.is_contraction() {
        return false;
    }
    let n = contraction.matrix.ncols();
    let initial = DVector::zeros(n);
    let result = contraction.find_fixed_point(&initial, max_iterations, tolerance);

    // Check convergence
    if !result.converged {
        return false;
    }

    // Check it's actually a fixed point: T(x*) ≈ x*
    let tx = contraction.apply(&result.fixed_point);
    let diff = &tx - &result.fixed_point;
    compute_norm(&diff, &contraction.norm_type) < tolerance * 10.0
}

/// Verify uniqueness: starting from different initial points converges to same fixed point.
pub fn verify_uniqueness(
    contraction: &ContractionMapping,
    initial_points: &[DVector<f64>],
    tolerance: f64,
    max_iterations: usize,
) -> bool {
    let fixed_points: Vec<_> = initial_points
        .iter()
        .map(|x0| contraction.find_fixed_point(x0, max_iterations, tolerance))
        .filter(|r| r.converged)
        .map(|r| r.fixed_point)
        .collect();

    if fixed_points.len() < 2 {
        return true;
    }

    let first = &fixed_points[0];
    fixed_points[1..].iter().all(|fp| {
        let diff = fp - first;
        compute_norm(&diff, &contraction.norm_type) < tolerance * 100.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_contraction_fixed_point() {
        // T(x) = 0.5*x + 1 => fixed point at x = 2
        let m = DMatrix::from_row_slice(1, 1, &[0.5]);
        let b = DVector::from_vec(vec![1.0]);
        let c = ContractionMapping::new(m, b, NormType::L2);
        let x0 = DVector::from_vec(vec![0.0]);
        let result = c.find_fixed_point(&x0, 100, 1e-10);
        assert!(result.converged);
        assert_relative_eq!(result.fixed_point[0], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_contraction_constant() {
        let m = DMatrix::from_row_slice(2, 2, &[0.3, 0.0, 0.0, 0.4]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let c = ContractionMapping::new(m, b, NormType::L2);
        assert!(c.is_contraction());
    }

    #[test]
    fn test_not_contraction() {
        let m = DMatrix::from_row_slice(1, 1, &[2.0]);
        let b = DVector::from_vec(vec![1.0]);
        let c = ContractionMapping::new(m, b, NormType::L2);
        assert!(!c.is_contraction());
    }

    #[test]
    fn test_exact_fixed_point() {
        let m = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let c = ContractionMapping::new(m, b, NormType::L2);
        let exact = c.exact_fixed_point().unwrap();
        // (I - 0.5I)^{-1} * b = 2I * b = [2, 4]
        assert_relative_eq!(exact[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(exact[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_verify_theorem() {
        let m = DMatrix::from_row_slice(2, 2, &[0.5, 0.1, 0.0, 0.5]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let c = ContractionMapping::new(m, b, NormType::L2);
        assert!(verify_fixed_point_theorem(&c, 1e-10, 1000));
    }

    #[test]
    fn test_uniqueness() {
        let m = DMatrix::from_row_slice(2, 2, &[0.3, 0.0, 0.0, 0.4]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let c = ContractionMapping::new(m, b, NormType::L2);
        let starts = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![10.0, -10.0]),
            DVector::from_vec(vec![-5.0, 5.0]),
        ];
        assert!(verify_uniqueness(&c, &starts, 1e-10, 1000));
    }

    #[test]
    fn test_linear_convergence_rate() {
        let m = DMatrix::from_row_slice(1, 1, &[0.5]);
        let b = DVector::from_vec(vec![1.0]);
        let c = ContractionMapping::new(m, b, NormType::L2);
        let x0 = DVector::from_vec(vec![0.0]);
        let result = c.find_fixed_point(&x0, 100, 1e-12);
        // Errors should decrease roughly geometrically
        assert!(result.error_history.len() > 2);
        for i in 2..result.error_history.len() {
            assert!(result.error_history[i] < result.error_history[i - 1]);
        }
    }

    #[test]
    fn test_2d_fixed_point() {
        // System: x = 0.5x + 0.1y + 1, y = 0.1x + 0.5y + 2
        let m = DMatrix::from_row_slice(2, 2, &[0.5, 0.1, 0.1, 0.5]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let c = ContractionMapping::new(m, b, NormType::L2);
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let result = c.find_fixed_point(&x0, 1000, 1e-10);
        assert!(result.converged);
        // Verify T(x*) = x*
        let tx = c.apply(&result.fixed_point);
        let err = (&tx - &result.fixed_point).norm();
        assert!(err < 1e-8);
    }
}
