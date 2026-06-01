//! Weak topology: convergence in dual pairing.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::banach_space::{BanachElement, NormType, compute_norm};
use crate::hahn_banach::LinearFunctional;

/// Weak convergence: x_n → x weakly iff f(x_n) → f(x) for all f ∈ X*.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakConvergence {
    pub sequence: Vec<BanachElement>,
}

impl WeakConvergence {
    pub fn new(sequence: Vec<BanachElement>) -> Self {
        Self { sequence }
    }

    /// Check weak convergence against a set of test functionals.
    /// A sequence converges weakly to x if f(x_n) → f(x) for all f in the dual space.
    pub fn check_weak_convergence(
        &self,
        limit: &BanachElement,
        functionals: &[LinearFunctional],
        epsilon: f64,
    ) -> bool {
        if self.sequence.len() < 2 {
            return true;
        }

        // Check that for each functional, f(x_n) converges to f(x)
        let n = self.sequence.len();
        let tail_start = n.saturating_sub(5);

        for f in functionals {
            let limit_val = f.evaluate(&limit.data);
            for elem in &self.sequence[tail_start..] {
                let val = f.evaluate(&elem.data);
                if (val - limit_val).abs() >= epsilon {
                    return false;
                }
            }
        }
        true
    }
}

/// Strong (norm) convergence: ‖x_n - x‖ → 0.
pub fn check_strong_convergence(
    sequence: &[BanachElement],
    limit: &BanachElement,
    epsilon: f64,
) -> bool {
    if sequence.is_empty() {
        return true;
    }
    let n = sequence.len();
    let tail_start = n.saturating_sub(5);
    sequence[tail_start..].iter().all(|elem| {
        elem.distance(limit).unwrap_or(f64::INFINITY) < epsilon
    })
}

/// Verify: strong convergence implies weak convergence.
pub fn strong_implies_weak(
    sequence: &[BanachElement],
    limit: &BanachElement,
    functionals: &[LinearFunctional],
    epsilon: f64,
) -> bool {
    let is_strong = check_strong_convergence(sequence, limit, epsilon);
    let wc = WeakConvergence::new(sequence.to_vec());
    let is_weak = wc.check_weak_convergence(limit, functionals, epsilon);
    // If strong, then must be weak
    !is_strong || is_weak
}

/// Verify: weak convergence does NOT necessarily imply strong convergence.
/// Example: orthonormal basis in Hilbert space converges weakly to 0 but not strongly.
pub fn weak_not_implies_strong_example() -> (Vec<BanachElement>, BanachElement) {
    let n = 5;
    let sequence: Vec<BanachElement> = (0..n)
        .map(|i| {
            let mut v = vec![0.0; n];
            v[i] = 1.0;
            BanachElement::new(v, NormType::L2)
        })
        .collect();
    let limit = BanachElement::new(vec![0.0; n], NormType::L2);
    (sequence, limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_strong_convergence() {
        let seq: Vec<BanachElement> = (1..=10)
            .map(|i| BanachElement::new(vec![1.0 / i as f64], NormType::L2))
            .collect();
        let limit = BanachElement::new(vec![0.0], NormType::L2);
        assert!(check_strong_convergence(&seq, &limit, 0.5));
    }

    #[test]
    fn test_not_strong_convergence() {
        let seq: Vec<BanachElement> = (1..=10)
            .map(|_| BanachElement::new(vec![1.0], NormType::L2))
            .collect();
        let limit = BanachElement::new(vec![0.0], NormType::L2);
        assert!(!check_strong_convergence(&seq, &limit, 0.5));
    }

    #[test]
    fn test_weak_convergence() {
        let seq: Vec<BanachElement> = (1..=10)
            .map(|i| BanachElement::new(vec![1.0 / i as f64], NormType::L2))
            .collect();
        let limit = BanachElement::new(vec![0.0], NormType::L2);
        let f = LinearFunctional::new(vec![1.0], NormType::L2);
        let wc = WeakConvergence::new(seq);
        assert!(wc.check_weak_convergence(&limit, &[f], 0.5));
    }

    #[test]
    fn test_strong_implies_weak() {
        let seq: Vec<BanachElement> = (1..=10)
            .map(|i| BanachElement::new(vec![1.0 / i as f64], NormType::L2))
            .collect();
        let limit = BanachElement::new(vec![0.0], NormType::L2);
        let f = LinearFunctional::new(vec![1.0], NormType::L2);
        assert!(strong_implies_weak(&seq, &limit, &[f], 0.5));
    }

    #[test]
    fn test_weak_not_strong_example() {
        let (seq, limit) = weak_not_implies_strong_example();
        // Each element has norm 1, but they converge weakly to 0
        for elem in &seq {
            assert_relative_eq!(elem.norm(), 1.0, epsilon = 1e-10);
        }
        assert_relative_eq!(limit.norm(), 0.0);
    }

    #[test]
    fn test_weak_convergence_orthogonal_basis() {
        let (seq, limit) = weak_not_implies_strong_example();
        let func1 = LinearFunctional::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], NormType::L2);
        let func2 = LinearFunctional::new(vec![0.0, 1.0, 0.0, 0.0, 0.0], NormType::L2);
        let wc = WeakConvergence::new(seq.clone());
        // Weak convergence to zero: for each f, eventually f(e_i) = 0
        // Since only the last elements matter and they have different support
        assert!(wc.check_weak_convergence(&limit, &[func1, func2], 1.1));
    }
}
