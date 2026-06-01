//! Spectral theory: spectrum of bounded operators, spectral radius formula.

use nalgebra::{DMatrix, DVector, ComplexField};
use serde::{Serialize, Deserialize};
use crate::bounded_operator::BoundedOperator;
use crate::banach_space::NormType;

/// Spectral analysis of a bounded linear operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    /// Eigenvalues (for finite-dimensional operators).
    pub eigenvalues: Vec<num_complex::Complex64>,
    /// Spectral radius: r(T) = max|λ|.
    pub spectral_radius: f64,
    /// Operator norm.
    pub operator_norm: f64,
}

impl SpectralAnalysis {
    /// Perform spectral analysis of a matrix operator.
    pub fn analyze(matrix: &DMatrix<f64>) -> Self {
        let eigen = matrix.complex_eigenvalues();
        let spectral_radius = eigen.iter().map(|e| e.norm()).fold(0.0_f64, f64::max);

        let op = BoundedOperator::new(matrix.clone(), NormType::L2, NormType::L2);
        let operator_norm = op.operator_norm_l2();

        Self {
            eigenvalues: eigen.iter().cloned().collect(),
            spectral_radius,
            operator_norm,
        }
    }

    /// Verify the spectral radius formula: r(T) = lim_{n→∞} ‖Tⁿ‖^{1/n}.
    pub fn verify_spectral_radius_formula(&self, matrix: &DMatrix<f64>, max_power: usize) -> bool {
        let _n = matrix.nrows();
        let mut current = matrix.clone();
        for k in 1..=max_power {
            let op = BoundedOperator::new(current.clone(), NormType::L2, NormType::L2);
            let norm_n = op.operator_norm_l2();
            let root = norm_n.powf(1.0 / k as f64);
            if (root - self.spectral_radius).abs() < 0.01 {
                return true;
            }
            current = &current * matrix;
        }
        false
    }

    /// Verify: r(T) ≤ ‖T‖ (spectral radius bounded by operator norm).
    pub fn verify_spectral_bound(&self) -> bool {
        self.spectral_radius <= self.operator_norm + 1e-10
    }

    /// Check if the operator is normal: T*T = TT*.
    pub fn is_normal(matrix: &DMatrix<f64>) -> bool {
        let t_star = matrix.transpose();
        let left = matrix * &t_star;
        let right = &t_star * matrix;
        (left - right).iter().all(|v| v.abs() < 1e-10)
    }

    /// For normal operators: r(T) = ‖T‖.
    pub fn verify_normal_spectral_equality(&self) -> bool {
        (self.spectral_radius - self.operator_norm).abs() < 1e-10
    }

    /// The spectrum σ(T) = {λ ∈ ℂ : T - λI is not invertible}.
    /// For matrices, this is just the set of eigenvalues.
    pub fn spectrum(&self) -> &[num_complex::Complex64] {
        &self.eigenvalues
    }

    /// Check if an operator is self-adjoint (Hermitian for real matrices).
    pub fn is_self_adjoint(matrix: &DMatrix<f64>) -> bool {
        if matrix.nrows() != matrix.ncols() {
            return false;
        }
        (matrix - matrix.transpose()).iter().all(|v| v.abs() < 1e-10)
    }

    /// Resolvent set: ℂ \ σ(T).
    pub fn is_in_resolvent(&self, lambda: num_complex::Complex64) -> bool {
        !self.eigenvalues.iter().any(|e| (e - lambda).norm() < 1e-10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_eigenvalues_diagonal() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 3.0]));
        let analysis = SpectralAnalysis::analyze(&m);
        assert_relative_eq!(analysis.spectral_radius, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_bound() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let analysis = SpectralAnalysis::analyze(&m);
        assert!(analysis.verify_spectral_bound());
    }

    #[test]
    fn test_spectral_radius_formula() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![0.5, 0.3]));
        let analysis = SpectralAnalysis::analyze(&m);
        assert!(analysis.verify_spectral_radius_formula(&m, 20));
    }

    #[test]
    fn test_normal_matrix() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, -2.0, 1.0]);
        assert!(SpectralAnalysis::is_normal(&m));
    }

    #[test]
    fn test_not_normal() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(!SpectralAnalysis::is_normal(&m));
    }

    #[test]
    fn test_self_adjoint() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 3.0]);
        assert!(SpectralAnalysis::is_self_adjoint(&m));
    }

    #[test]
    fn test_identity_spectrum() {
        let m = DMatrix::identity(3, 3);
        let analysis = SpectralAnalysis::analyze(&m);
        assert_eq!(analysis.eigenvalues.len(), 3);
        for e in &analysis.eigenvalues {
            assert_relative_eq!(e.re, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_resolvent() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0]));
        let analysis = SpectralAnalysis::analyze(&m);
        assert!(!analysis.is_in_resolvent(num_complex::Complex64::new(1.0, 0.0)));
        assert!(analysis.is_in_resolvent(num_complex::Complex64::new(0.0, 1.0)));
    }

    #[test]
    fn test_spectral_radius_zero_matrix() {
        let m = DMatrix::zeros(2, 2);
        let analysis = SpectralAnalysis::analyze(&m);
        assert_relative_eq!(analysis.spectral_radius, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_spectral_equality() {
        // Symmetric → normal → r(T) = ‖T‖
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]);
        let analysis = SpectralAnalysis::analyze(&m);
        assert!(SpectralAnalysis::is_normal(&m));
        assert!(analysis.verify_normal_spectral_equality());
    }
}
