//! Bounded linear operators: ‖T‖ = sup‖Tx‖/‖x‖.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::banach_space::{BanachElement, NormType, compute_norm};

/// A bounded linear operator T: X → Y between Banach spaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedOperator {
    /// Matrix representation of the operator.
    pub matrix: DMatrix<f64>,
    /// Domain norm type.
    pub domain_norm: NormType,
    /// Codomain norm type.
    pub codomain_norm: NormType,
}

impl BoundedOperator {
    pub fn new(matrix: DMatrix<f64>, domain_norm: NormType, codomain_norm: NormType) -> Self {
        Self { matrix, domain_norm, codomain_norm }
    }

    /// Apply the operator to a vector.
    pub fn apply(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.matrix * x
    }

    /// Apply to a BanachElement.
    pub fn apply_element(&self, x: &BanachElement) -> BanachElement {
        let result = self.apply(&x.data);
        BanachElement { data: result, norm_type: self.codomain_norm.clone() }
    }

    /// Number of rows (codomain dimension).
    pub fn nrows(&self) -> usize { self.matrix.nrows() }

    /// Number of columns (domain dimension).
    pub fn ncols(&self) -> usize { self.matrix.ncols() }

    /// Compute the operator norm: ‖T‖ = sup_{‖x‖=1} ‖Tx‖.
    /// Estimated via sampling over unit vectors.
    pub fn operator_norm(&self, samples: usize) -> f64 {
        let mut max_norm = 0.0_f64;
        let n = self.ncols();
        for _ in 0..samples {
            // Generate random unit vector (not truly random, use deterministic)
            let mut v = DVector::zeros(n);
            for i in 0..n {
                v[i] = 1.0;
                let norm = compute_norm(&v, &self.domain_norm);
                if norm > 1e-10 {
                    let unit = &v / norm;
                    let result = self.apply(&unit);
                    let result_norm = compute_norm(&result, &self.codomain_norm);
                    max_norm = max_norm.max(result_norm);
                }
                v[i] = -1.0;
                let norm = compute_norm(&v, &self.domain_norm);
                if norm > 1e-10 {
                    let unit = &v / norm;
                    let result = self.apply(&unit);
                    let result_norm = compute_norm(&result, &self.codomain_norm);
                    max_norm = max_norm.max(result_norm);
                }
                v[i] = 0.0;
            }
            // Also try all-ones
            for i in 0..n { v[i] = 1.0; }
            let norm = compute_norm(&v, &self.domain_norm);
            if norm > 1e-10 {
                let unit = &v / norm;
                let result = self.apply(&unit);
                let result_norm = compute_norm(&result, &self.codomain_norm);
                max_norm = max_norm.max(result_norm);
            }
        }
        max_norm
    }

    /// Exact operator norm for the L2 norm via singular values.
    pub fn operator_norm_l2(&self) -> f64 {
        let svd = self.matrix.clone().svd(true, true);
        svd.singular_values[0]
    }

    /// Compose two operators: S ∘ T.
    pub fn compose(&self, other: &BoundedOperator) -> Result<BoundedOperator, String> {
        if self.ncols() != other.nrows() {
            return Err("Dimension mismatch for composition".into());
        }
        let product = &self.matrix * &other.matrix;
        Ok(BoundedOperator::new(product, other.domain_norm.clone(), self.codomain_norm.clone()))
    }

    /// Add two operators (must have same dimensions).
    pub fn add(&self, other: &BoundedOperator) -> Result<BoundedOperator, String> {
        if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
            return Err("Dimension mismatch".into());
        }
        Ok(BoundedOperator::new(
            &self.matrix + &other.matrix,
            self.domain_norm.clone(),
            self.codomain_norm.clone(),
        ))
    }

    /// Scalar multiplication of operator.
    pub fn scale(&self, scalar: f64) -> BoundedOperator {
        BoundedOperator::new(
            self.matrix.scale(scalar),
            self.domain_norm.clone(),
            self.codomain_norm.clone(),
        )
    }

    /// Identity operator.
    pub fn identity(n: usize, norm: NormType) -> Self {
        Self::new(DMatrix::identity(n, n), norm.clone(), norm)
    }

    /// Zero operator.
    pub fn zero(m: usize, n: usize, domain_norm: NormType, codomain_norm: NormType) -> Self {
        Self::new(DMatrix::zeros(m, n), domain_norm, codomain_norm)
    }

    /// Verify linearity: T(αx + βy) = αT(x) + βT(y).
    pub fn verify_linearity(&self, x: &DVector<f64>, y: &DVector<f64>, alpha: f64, beta: f64) -> bool {
        let lhs = self.apply(&(x * alpha + y * beta));
        let rhs = self.apply(x) * alpha + self.apply(y) * beta;
        (lhs - rhs).iter().all(|v| v.abs() < 1e-10)
    }

    /// Verify boundedness: ‖Tx‖ ≤ ‖T‖·‖x‖.
    pub fn verify_boundedness(&self, x: &DVector<f64>) -> bool {
        let op_norm = self.operator_norm(100);
        let tx_norm = compute_norm(&self.apply(x), &self.codomain_norm);
        let x_norm = compute_norm(x, &self.domain_norm);
        tx_norm <= op_norm * x_norm + 1e-8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_operator_apply() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        let x = DVector::from_vec(vec![3.0, 4.0]);
        let result = op.apply(&x);
        assert_relative_eq!(result[0], 3.0);
        assert_relative_eq!(result[1], 8.0);
    }

    #[test]
    fn test_operator_norm_l2_diagonal() {
        let m = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 4.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        assert_relative_eq!(op.operator_norm_l2(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linearity() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        let x = DVector::from_vec(vec![1.0, 0.0]);
        let y = DVector::from_vec(vec![0.0, 1.0]);
        assert!(op.verify_linearity(&x, &y, 2.0, 3.0));
    }

    #[test]
    fn test_boundedness() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        let x = DVector::from_vec(vec![1.0, 1.0]);
        assert!(op.verify_boundedness(&x));
    }

    #[test]
    fn test_identity_operator() {
        let id = BoundedOperator::identity(3, NormType::L2);
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = id.apply(&x);
        assert_relative_eq!(result[0], 1.0);
        assert_relative_eq!(result[1], 2.0);
        assert_relative_eq!(result[2], 3.0);
    }

    #[test]
    fn test_composition() {
        let a = BoundedOperator::new(
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]),
            NormType::L2, NormType::L2,
        );
        let b = BoundedOperator::new(
            DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]),
            NormType::L2, NormType::L2,
        );
        let c = a.compose(&b).unwrap();
        assert_relative_eq!(c.matrix[(0, 0)], 2.0);
        assert_relative_eq!(c.matrix[(1, 1)], 6.0);
    }

    #[test]
    fn test_zero_operator() {
        let z = BoundedOperator::zero(2, 3, NormType::L2, NormType::L2);
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = z.apply(&x);
        assert_relative_eq!(result[0], 0.0);
        assert_relative_eq!(result[1], 0.0);
    }

    #[test]
    fn test_scalar_mult_operator() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        let scaled = op.scale(3.0);
        assert_relative_eq!(scaled.matrix[(0, 0)], 3.0);
    }
}
