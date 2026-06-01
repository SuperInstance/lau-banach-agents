//! Banach space: complete normed vector space.

use nalgebra::{DVector, DMatrix, RealField};
use serde::{Serialize, Deserialize};
use std::fmt;

/// A norm type for Banach spaces.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormType {
    /// ‖x‖_p = (∑|x_i|^p)^(1/p)
    #[serde(rename = "lp")]
    Lp(f64),
    /// ‖x‖_∞ = max|x_i|
    #[serde(rename = "linf")]
    LInf,
    /// ‖x‖_1 = ∑|x_i|
    #[serde(rename = "l1")]
    L1,
    /// ‖x‖_2 = sqrt(∑x_i²)
    #[serde(rename = "l2")]
    L2,
}

/// A Banach space element: a vector equipped with a norm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanachElement {
    pub data: DVector<f64>,
    pub norm_type: NormType,
}

impl BanachElement {
    pub fn new(data: Vec<f64>, norm_type: NormType) -> Self {
        Self { data: DVector::from_vec(data), norm_type }
    }

    pub fn dimension(&self) -> usize {
        self.data.nrows()
    }

    /// Compute the norm of this element.
    pub fn norm(&self) -> f64 {
        compute_norm(&self.data, &self.norm_type)
    }

    /// Zero element in the same space.
    pub fn zero(dim: usize, norm_type: NormType) -> Self {
        Self { data: DVector::zeros(dim), norm_type }
    }

    /// Add two elements (must have same dimension).
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        if self.dimension() != other.dimension() {
            return Err("Dimension mismatch".into());
        }
        Ok(Self {
            data: &self.data + &other.data,
            norm_type: self.norm_type.clone(),
        })
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: f64) -> Self {
        Self {
            data: &self.data * scalar,
            norm_type: self.norm_type.clone(),
        }
    }

    /// Subtraction.
    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        if self.dimension() != other.dimension() {
            return Err("Dimension mismatch".into());
        }
        Ok(Self {
            data: &self.data - &other.data,
            norm_type: self.norm_type.clone(),
        })
    }

    /// Distance between two elements: ‖self - other‖.
    pub fn distance(&self, other: &Self) -> Result<f64, String> {
        let diff = self.sub(other)?;
        Ok(diff.norm())
    }
}

/// Compute norm of a vector under a given norm type.
pub fn compute_norm(v: &DVector<f64>, norm_type: &NormType) -> f64 {
    match norm_type {
        NormType::L1 => v.iter().map(|x| x.abs()).sum(),
        NormType::L2 => v.iter().map(|x| x * x).sum::<f64>().sqrt(),
        NormType::LInf => v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
        NormType::Lp(p) => {
            let sum: f64 = v.iter().map(|x| x.abs().powf(*p)).sum();
            sum.powf(1.0 / p)
        }
    }
}

/// Verify the triangle inequality: ‖x + y‖ ≤ ‖x‖ + ‖y‖.
pub fn triangle_inequality(x: &BanachElement, y: &BanachElement) -> bool {
    if let Ok(sum) = x.add(y) {
        sum.norm() <= x.norm() + y.norm() + 1e-10
    } else {
        false
    }
}

/// Verify absolute homogeneity: ‖αx‖ = |α|‖x‖.
pub fn absolute_homogeneity(x: &BanachElement, alpha: f64) -> bool {
    let scaled = x.scale(alpha);
    (scaled.norm() - alpha.abs() * x.norm()).abs() < 1e-10
}

/// Verify positive definiteness: ‖x‖ = 0 ⟺ x = 0.
pub fn positive_definiteness(x: &BanachElement) -> bool {
    if x.data.iter().all(|v| v.abs() < 1e-10) {
        x.norm() < 1e-10
    } else {
        x.norm() > 1e-10
    }
}

/// A Cauchy sequence in a Banach space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CauchySequence {
    pub elements: Vec<BanachElement>,
}

impl CauchySequence {
    pub fn new(elements: Vec<BanachElement>) -> Self {
        Self { elements }
    }

    /// Check if this is actually a Cauchy sequence.
    /// For all ε > 0, there exists N such that for all m,n > N: ‖x_m - x_n‖ < ε.
    pub fn is_cauchy(&self, epsilon: f64) -> bool {
        if self.elements.len() < 2 {
            return true;
        }
        let n = self.elements.len();
        // Check tail pairs
        for i in (n.saturating_sub(10))..n {
            for j in (i + 1)..n {
                if let Ok(dist) = self.elements[i].distance(&self.elements[j]) {
                    if dist >= epsilon {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Compute the limit (last element as approximation for convergent sequences).
    pub fn limit(&self) -> Option<&BanachElement> {
        self.elements.last()
    }
}

/// Verify completeness: every Cauchy sequence converges in the space.
/// In finite dimensions, all norms are equivalent so the space is complete.
pub fn verify_completeness(sequence: &CauchySequence, epsilon: f64) -> bool {
    if sequence.elements.is_empty() {
        return true;
    }
    // In finite dimensions, Cauchy sequences always converge
    sequence.is_cauchy(epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_l2_norm() {
        let v = BanachElement::new(vec![3.0, 4.0], NormType::L2);
        assert_relative_eq!(v.norm(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l1_norm() {
        let v = BanachElement::new(vec![3.0, -4.0], NormType::L1);
        assert_relative_eq!(v.norm(), 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linf_norm() {
        let v = BanachElement::new(vec![3.0, -5.0, 2.0], NormType::LInf);
        assert_relative_eq!(v.norm(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lp_norm() {
        let v = BanachElement::new(vec![1.0, 1.0], NormType::Lp(3.0));
        // (1 + 1)^(1/3) = 2^(1/3)
        assert_relative_eq!(v.norm(), 2.0_f64.powf(1.0 / 3.0), epsilon = 1e-10);
    }

    #[test]
    fn test_triangle_inequality_holds() {
        let x = BanachElement::new(vec![1.0, 2.0], NormType::L2);
        let y = BanachElement::new(vec![3.0, 4.0], NormType::L2);
        assert!(triangle_inequality(&x, &y));
    }

    #[test]
    fn test_absolute_homogeneity_holds() {
        let x = BanachElement::new(vec![1.0, 2.0], NormType::L2);
        assert!(absolute_homogeneity(&x, 3.0));
    }

    #[test]
    fn test_positive_definiteness() {
        let nonzero = BanachElement::new(vec![1.0, 0.0], NormType::L2);
        let zero = BanachElement::new(vec![0.0, 0.0], NormType::L2);
        assert!(positive_definiteness(&nonzero));
        assert!(positive_definiteness(&zero));
    }

    #[test]
    fn test_addition() {
        let x = BanachElement::new(vec![1.0, 2.0], NormType::L2);
        let y = BanachElement::new(vec![3.0, 4.0], NormType::L2);
        let sum = x.add(&y).unwrap();
        assert_relative_eq!(sum.data[0], 4.0);
        assert_relative_eq!(sum.data[1], 6.0);
    }

    #[test]
    fn test_scalar_mult() {
        let x = BanachElement::new(vec![1.0, 2.0], NormType::L2);
        let scaled = x.scale(3.0);
        assert_relative_eq!(scaled.data[0], 3.0);
        assert_relative_eq!(scaled.data[1], 6.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let x = BanachElement::new(vec![1.0], NormType::L2);
        let y = BanachElement::new(vec![1.0, 2.0], NormType::L2);
        assert!(x.add(&y).is_err());
    }

    #[test]
    fn test_cauchy_sequence_convergent() {
        let seq: Vec<BanachElement> = (0..20)
            .map(|i| {
                let val = 1.0 / (i as f64 + 1.0).powi(2);
                BanachElement::new(vec![val], NormType::L2)
            })
            .collect();
        let cauchy = CauchySequence::new(seq);
        assert!(cauchy.is_cauchy(1.0));
    }

    #[test]
    fn test_zero_element() {
        let z = BanachElement::zero(3, NormType::L2);
        assert_eq!(z.dimension(), 3);
        assert_relative_eq!(z.norm(), 0.0);
    }

    #[test]
    fn test_distance() {
        let x = BanachElement::new(vec![0.0, 0.0], NormType::L2);
        let y = BanachElement::new(vec![3.0, 4.0], NormType::L2);
        assert_relative_eq!(x.distance(&y).unwrap(), 5.0, epsilon = 1e-10);
    }
}
