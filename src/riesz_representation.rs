//! Riesz representation: identify dual of Lᵖ as Lᵍ (1/p + 1/q = 1).

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::banach_space::NormType;
use crate::hahn_banach::LinearFunctional;

/// Riesz representation theorem for Lᵖ spaces.
/// Every bounded linear functional f on Lᵖ can be represented as:
///   f(x) = ∫ x(t)·g(t) dt
/// where g ∈ Lᵍ with 1/p + 1/q = 1, and ‖f‖ = ‖g‖_q.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RieszRepresentation {
    /// The conjugate exponent q such that 1/p + 1/q = 1.
    pub conjugate_exponent: f64,
    /// The representing element g in Lᵍ.
    pub representing_element: LinearFunctional,
}

impl RieszRepresentation {
    /// Compute the conjugate exponent q from p.
    pub fn conjugate_exponent(p: f64) -> f64 {
        1.0 / (1.0 - 1.0 / p)
    }

    /// Given a functional on Lᵖ, find its Riesz representation g ∈ Lᵍ.
    pub fn represent(f: &LinearFunctional) -> Self {
        let q = match &f.norm_type {
            NormType::Lp(p) => Self::conjugate_exponent(*p),
            NormType::L1 => f64::INFINITY,
            NormType::L2 => 2.0,
            NormType::LInf => 1.0,
        };

        // For the discrete (finite-dimensional) case:
        // The Riesz representative is just the coefficient vector
        // but viewed in the conjugate space Lᵍ
        Self {
            conjugate_exponent: q,
            representing_element: f.clone(),
        }
    }

    /// Verify ‖f‖ = ‖g‖_q (norm equality from Riesz representation).
    pub fn verify_norm_equality(&self) -> bool {
        let f_norm = self.representing_element.dual_norm();
        // For discrete Lᵖ, the dual norm is the Lᵍ norm of coefficients
        let q_norm = match self.conjugate_exponent {
            q if q.is_infinite() => {
                // LInf norm
                self.representing_element.coefficients.iter().map(|c| c.abs()).fold(0.0_f64, f64::max)
            }
            q => {
                let sum: f64 = self.representing_element.coefficients.iter()
                    .map(|c| c.abs().powf(q)).sum();
                sum.powf(1.0 / q)
            }
        };
        (f_norm - q_norm).abs() < 1e-8
    }
}

/// Verify Lᵖ duality: (Lᵖ)* = Lᵍ where 1/p + 1/q = 1.
pub fn verify_lp_duality(p: f64, coefficients: &[f64]) -> bool {
    let q = RieszRepresentation::conjugate_exponent(p);
    let f = LinearFunctional::new(coefficients.to_vec(), NormType::Lp(p));
    let rep = RieszRepresentation::represent(&f);

    // Check 1/p + 1/q = 1
    if (1.0 / p + 1.0 / q - 1.0).abs() > 1e-10 {
        return false;
    }

    rep.verify_norm_equality()
}

/// Hölder's inequality: |∑ xᵢyᵢ| ≤ ‖x‖_p · ‖y‖_q where 1/p + 1/q = 1.
pub fn holder_inequality(x: &DVector<f64>, y: &DVector<f64>, p: f64) -> bool {
    let q = RieszRepresentation::conjugate_exponent(p);
    let inner: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>().abs();
    let xp = x.iter().map(|v| v.abs().powf(p)).sum::<f64>().powf(1.0 / p);
    let yq = y.iter().map(|v| v.abs().powf(q)).sum::<f64>().powf(1.0 / q);
    inner <= xp * yq + 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_conjugate_exponent() {
        assert_relative_eq!(RieszRepresentation::conjugate_exponent(2.0), 2.0);
        assert_relative_eq!(RieszRepresentation::conjugate_exponent(3.0), 1.5);
        assert_relative_eq!(RieszRepresentation::conjugate_exponent(1.5), 3.0);
    }

    #[test]
    fn test_conjugate_reciprocal() {
        for p in [1.5, 2.0, 3.0, 4.0, 5.0] {
            let q = RieszRepresentation::conjugate_exponent(p);
            assert_relative_eq!(1.0 / p + 1.0 / q, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_riesz_representation_l2() {
        let f = LinearFunctional::new(vec![3.0, 4.0], NormType::L2);
        let rep = RieszRepresentation::represent(&f);
        assert_relative_eq!(rep.conjugate_exponent, 2.0);
        assert!(rep.verify_norm_equality());
    }

    #[test]
    fn test_holder_inequality() {
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let y = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        assert!(holder_inequality(&x, &y, 2.0));
    }

    #[test]
    fn test_holder_inequality_tight() {
        // Equality when x and y are proportional in the right sense
        let x = DVector::from_vec(vec![1.0, 0.0]);
        let y = DVector::from_vec(vec![1.0, 0.0]);
        assert!(holder_inequality(&x, &y, 2.0));
    }

    #[test]
    fn test_lp_duality() {
        assert!(verify_lp_duality(2.0, &[1.0, 2.0, 3.0]));
        assert!(verify_lp_duality(3.0, &[1.0, 2.0]));
    }

    #[test]
    fn test_holder_p3() {
        let x = DVector::from_vec(vec![2.0, 1.0, 3.0]);
        let y = DVector::from_vec(vec![1.0, 2.0, 1.0]);
        assert!(holder_inequality(&x, &y, 3.0));
    }
}
