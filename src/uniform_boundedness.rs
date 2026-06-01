//! Uniform boundedness principle: pointwise bounded → uniformly bounded.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::bounded_operator::BoundedOperator;
use crate::banach_space::{NormType, compute_norm};

/// A family of bounded linear operators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorFamily {
    pub operators: Vec<BoundedOperator>,
}

impl OperatorFamily {
    pub fn new(operators: Vec<BoundedOperator>) -> Self {
        Self { operators }
    }

    /// Check pointwise boundedness: for each x, sup_i ‖T_i(x)‖ < ∞.
    /// We test on a finite set of test vectors.
    pub fn is_pointwise_bounded(&self, test_vectors: &[DVector<f64>], bound: f64) -> bool {
        test_vectors.iter().all(|x| {
            self.operators.iter().all(|op| {
                let result = op.apply(x);
                compute_norm(&result, &op.codomain_norm) <= bound
            })
        })
    }

    /// Compute the uniform bound: sup_i ‖T_i‖.
    pub fn uniform_bound(&self, samples: usize) -> f64 {
        self.operators
            .iter()
            .map(|op| op.operator_norm(samples))
            .fold(0.0_f64, f64::max)
    }

    /// The uniform boundedness principle states:
    /// If {T_i} is a family of bounded operators and for each x, sup_i ‖T_i x‖ < ∞,
    /// then sup_i ‖T_i‖ < ∞.
    ///
    /// In finite dimensions, this is always true since operator norms are finite.
    pub fn verify_uniform_boundedness(&self, samples: usize) -> UniformBoundednessResult {
        let uniform_bound = self.uniform_bound(samples);
        UniformBoundednessResult {
            num_operators: self.operators.len(),
            uniform_bound,
            is_uniformly_bounded: uniform_bound.is_finite(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformBoundednessResult {
    pub num_operators: usize,
    pub uniform_bound: f64,
    pub is_uniformly_bounded: bool,
}

/// Counterexample: pointwise bounded but NOT uniformly bounded.
/// This can only happen in infinite dimensions. In finite dimensions,
/// we demonstrate that pointwise bounded always implies uniform bounded.
pub fn verify_finite_dim_uniform_boundedness() -> bool {
    // Generate a family of operators with bounded entries
    let ops: Vec<BoundedOperator> = (1..=10)
        .map(|k| {
            let m = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0 / k as f64]));
            BoundedOperator::new(m, NormType::L2, NormType::L2)
        })
        .collect();
    let family = OperatorFamily::new(ops);
    let result = family.verify_uniform_boundedness(10);
    result.is_uniformly_bounded
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pointwise_bounded() {
        let ops: Vec<BoundedOperator> = (1..=5)
            .map(|k| {
                let m = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0 / k as f64]));
                BoundedOperator::new(m, NormType::L2, NormType::L2)
            })
            .collect();
        let family = OperatorFamily::new(ops);
        let test = vec![DVector::from_vec(vec![1.0])];
        assert!(family.is_pointwise_bounded(&test, 2.0));
    }

    #[test]
    fn test_uniform_bound() {
        let ops: Vec<BoundedOperator> = (1..=5)
            .map(|k| {
                let m = DMatrix::from_diagonal(&DVector::from_vec(vec![k as f64]));
                BoundedOperator::new(m, NormType::L2, NormType::L2)
            })
            .collect();
        let family = OperatorFamily::new(ops);
        let ub = family.uniform_bound(10);
        assert_relative_eq!(ub, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_boundedness_principle() {
        let ops: Vec<BoundedOperator> = (1..=10)
            .map(|k| {
                let m = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0 / k as f64]));
                BoundedOperator::new(m, NormType::L2, NormType::L2)
            })
            .collect();
        let family = OperatorFamily::new(ops);
        let result = family.verify_uniform_boundedness(10);
        assert!(result.is_uniformly_bounded);
    }

    #[test]
    fn test_finite_dim_always_uniformly_bounded() {
        assert!(verify_finite_dim_uniform_boundedness());
    }

    #[test]
    fn test_identity_family() {
        let ops = vec![BoundedOperator::identity(2, NormType::L2); 5];
        let family = OperatorFamily::new(ops);
        let ub = family.uniform_bound(10);
        assert_relative_eq!(ub, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_decreasing_norms() {
        let ops: Vec<BoundedOperator> = (1..=5)
            .map(|k| {
                let scale = 1.0 / (2.0_f64.powi(k));
                let m = DMatrix::from_diagonal(&DVector::from_vec(vec![scale]));
                BoundedOperator::new(m, NormType::L2, NormType::L2)
            })
            .collect();
        let family = OperatorFamily::new(ops);
        let ub = family.uniform_bound(10);
        assert!(ub < 1.0);
    }
}
