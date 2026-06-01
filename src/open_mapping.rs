//! Open mapping theorem: surjective bounded operators are open maps.

use nalgebra::{DMatrix, DVector};
use crate::bounded_operator::BoundedOperator;
use crate::banach_space::NormType;

/// Verify the open mapping theorem: if T: X → Y is a surjective bounded linear
/// operator between Banach spaces, then T is an open map (maps open sets to open sets).
pub fn verify_open_mapping(operator: &BoundedOperator) -> OpenMappingResult {
    let m = operator.nrows();
    let _n = operator.ncols();

    // A bounded operator is open iff it's surjective
    // For matrices: surjective iff rank = m (full row rank)
    let rank = operator.matrix.rank(1e-10);
    let is_surjective = rank == m;

    // An open map maps the open unit ball to a neighborhood of 0 in Y
    // This means: there exists δ > 0 such that δ·B_Y ⊂ T(B_X)
    let min_singular = {
        let svd = operator.matrix.clone().svd(false, false);
        svd.singular_values.iter()
            .filter(|s| **s > 1e-10)
            .fold(f64::INFINITY, |a, &b| a.min(b))
    };

    OpenMappingResult {
        is_surjective,
        rank,
        is_open_map: is_surjective && min_singular > 0.0,
        min_singular_value: min_singular,
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenMappingResult {
    pub is_surjective: bool,
    pub rank: usize,
    pub is_open_map: bool,
    pub min_singular_value: f64,
}

/// Corollary: If T is bijective and bounded, then T⁻¹ is bounded.
pub fn bounded_inverse(operator: &BoundedOperator) -> Option<f64> {
    if operator.nrows() != operator.ncols() {
        return None;
    }

    let rank = operator.matrix.rank(1e-10);
    if rank != operator.nrows() {
        return None;
    }

    let svd = operator.matrix.clone().svd(false, false);
    let min_sv = svd.singular_values.iter().cloned().fold(f64::INFINITY, f64::min);

    if min_sv < 1e-10 {
        return None;
    }

    // ‖T⁻¹‖ = 1/σ_min
    Some(1.0 / min_sv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_surjective_operator_is_open() {
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        let result = verify_open_mapping(&op);
        assert!(result.is_surjective);
        assert!(result.is_open_map);
    }

    #[test]
    fn test_non_surjective_operator() {
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        let result = verify_open_mapping(&op);
        assert!(!result.is_surjective);
    }

    #[test]
    fn test_bounded_inverse() {
        let m = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        let inv_norm = bounded_inverse(&op).unwrap();
        assert_relative_eq!(inv_norm, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_open_mapping() {
        let id = BoundedOperator::identity(3, NormType::L2);
        let result = verify_open_mapping(&id);
        assert!(result.is_surjective);
        assert!(result.is_open_map);
        assert_relative_eq!(result.min_singular_value, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_non_square_no_inverse_bound() {
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        assert!(bounded_inverse(&op).is_none());
    }

    #[test]
    fn test_singular_no_inverse() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let op = BoundedOperator::new(m, NormType::L2, NormType::L2);
        assert!(bounded_inverse(&op).is_none());
    }
}
