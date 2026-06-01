//! Application: Banach fixed point theorem for iterative agent learning convergence.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::banach_space::NormType;
use crate::fixed_point::{ContractionMapping, FixedPointResult};

/// An agent that learns by iterating a contraction mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningAgent {
    /// The agent's parameter vector (state).
    pub parameters: DVector<f64>,
    /// The learning operator: T(θ) = Aθ + b.
    pub learning_map: ContractionMapping,
    /// Learning rate (affects contraction constant).
    pub learning_rate: f64,
}

impl LearningAgent {
    /// Create a new learning agent with parameters θ, learning rate η,
    /// and gradient operator G (the linear part of the update rule).
    /// Update rule: θ_{n+1} = θ_n - η·G·θ_n + η·target
    /// = (I - ηG)·θ_n + η·target
    pub fn new(
        initial_params: Vec<f64>,
        gradient_matrix: DMatrix<f64>,
        target: DVector<f64>,
        learning_rate: f64,
    ) -> Self {
        let n = initial_params.len();
        let identity = DMatrix::identity(n, n);
        let a = &identity - &gradient_matrix.scale(learning_rate);
        let b = target.scale(learning_rate);

        let learning_map = ContractionMapping::new(a, b, NormType::L2);

        Self {
            parameters: DVector::from_vec(initial_params),
            learning_map,
            learning_rate,
        }
    }

    /// Check if the learning process is a contraction (guaranteed to converge).
    pub fn is_contractive(&self) -> bool {
        self.learning_map.is_contraction()
    }

    /// Run one learning iteration.
    pub fn step(&mut self) -> f64 {
        let new_params = self.learning_map.apply(&self.parameters);
        let diff = &new_params - &self.parameters;
        let error = diff.norm();
        self.parameters = new_params;
        error
    }

    /// Train until convergence.
    pub fn train(&mut self, max_iterations: usize, tolerance: f64) -> FixedPointResult {
        let result = self.learning_map.find_fixed_point(
            &self.parameters,
            max_iterations,
            tolerance,
        );
        self.parameters = result.fixed_point.clone();
        result
    }

    /// Get the optimal parameters (fixed point of the learning map).
    pub fn optimal_parameters(&self) -> Option<DVector<f64>> {
        self.learning_map.exact_fixed_point()
    }
}

/// A multi-agent system where agents converge through shared contraction mappings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentSystem {
    pub agents: Vec<LearningAgent>,
}

impl MultiAgentSystem {
    pub fn new(agents: Vec<LearningAgent>) -> Self {
        Self { agents }
    }

    /// Check if all agents converge (all are contractive).
    pub fn all_converge(&self) -> bool {
        self.agents.iter().all(|a| a.is_contractive())
    }

    /// Compute the joint contraction rate (max of individual rates).
    pub fn joint_contraction_rate(&self) -> f64 {
        self.agents
            .iter()
            .map(|a| a.learning_map.contraction_constant())
            .fold(0.0_f64, f64::max)
    }

    /// Train all agents and return results.
    pub fn train_all(&mut self, max_iterations: usize, tolerance: f64) -> Vec<FixedPointResult> {
        self.agents
            .iter_mut()
            .map(|a| a.train(max_iterations, tolerance))
            .collect()
    }
}

/// Verify the convergence theorem for iterative agent learning:
/// If the learning operator is a contraction, then:
/// 1. The learning process converges to a unique fixed point
/// 2. The convergence rate is determined by the contraction constant
/// 3. The error bound after n steps is k^n/(1-k) * ‖x_1 - x_0‖
pub fn verify_convergence_theorem(
    agent: &LearningAgent,
    max_iterations: usize,
    tolerance: f64,
) -> ConvergenceVerification {
    let k = agent.learning_map.contraction_constant();
    let is_contraction = k < 1.0;

    if !is_contraction {
        return ConvergenceVerification {
            is_contraction: false,
            converged: false,
            error_bound_holds: false,
            rate_monotone: false,
        };
    }

    let mut test_agent = agent.clone();
    let initial_error = {
        let next = test_agent.learning_map.apply(&test_agent.parameters);
        (&next - &test_agent.parameters).norm()
    };

    let result = test_agent.train(max_iterations, tolerance);

    // Check error bound: final error ≤ k^n * initial_error
    let n = result.iterations as f64;
    let theoretical_bound = k.powf(n) / (1.0 - k) * initial_error;
    let error_bound_holds = result.final_error <= theoretical_bound + tolerance * 100.0;

    // Check monotone convergence (errors decrease)
    let rate_monotone = result.error_history.windows(2).all(|w| w[0] >= w[1] - 1e-12);

    ConvergenceVerification {
        is_contraction: true,
        converged: result.converged,
        error_bound_holds,
        rate_monotone,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceVerification {
    pub is_contraction: bool,
    pub converged: bool,
    pub error_bound_holds: bool,
    pub rate_monotone: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_learning_agent_contraction() {
        // Gradient descent on a strongly convex function
        // G = positive definite, η small enough → (I - ηG) has spectral radius < 1
        let g = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let target = DVector::from_vec(vec![1.0, 1.0]);
        let agent = LearningAgent::new(vec![0.0, 0.0], g, target, 0.1);
        assert!(agent.is_contractive());
    }

    #[test]
    fn test_learning_agent_converges() {
        let g = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let target = DVector::from_vec(vec![1.0, 1.0]);
        let mut agent = LearningAgent::new(vec![0.0, 0.0], g, target, 0.1);
        let result = agent.train(1000, 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_optimal_parameters() {
        // For G = I, target = [1,2], η = 0.5:
        // A = (I - 0.5I) = 0.5I, b = 0.5 * [1,2]
        // Fixed point: (I - 0.5I)^{-1} * 0.5 * [1,2] = 2 * 0.5 * [1,2] = [1,2]
        let g = DMatrix::identity(2, 2);
        let target = DVector::from_vec(vec![1.0, 2.0]);
        let agent = LearningAgent::new(vec![0.0, 0.0], g, target, 0.5);
        let optimal = agent.optimal_parameters().unwrap();
        assert_relative_eq!(optimal[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(optimal[1], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_step_decreases_error() {
        let g = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let target = DVector::from_vec(vec![1.0, 1.0]);
        let mut agent = LearningAgent::new(vec![0.0, 0.0], g, target, 0.3);
        let e1 = agent.step();
        let e2 = agent.step();
        assert!(e2 < e1);
    }

    #[test]
    fn test_multi_agent_system() {
        let agents: Vec<LearningAgent> = (0..3)
            .map(|_| {
                let g = DMatrix::from_row_slice(1, 1, &[2.0]);
                let target = DVector::from_vec(vec![1.0]);
                LearningAgent::new(vec![0.0], g, target, 0.1)
            })
            .collect();
        let system = MultiAgentSystem::new(agents);
        assert!(system.all_converge());
    }

    #[test]
    fn test_multi_agent_train() {
        let agents: Vec<LearningAgent> = (0..3)
            .map(|i| {
                let g = DMatrix::from_row_slice(1, 1, &[2.0]);
                let target = DVector::from_vec(vec![i as f64]);
                LearningAgent::new(vec![0.0], g, target, 0.1)
            })
            .collect();
        let mut system = MultiAgentSystem::new(agents);
        let results = system.train_all(1000, 1e-10);
        assert!(results.iter().all(|r| r.converged));
    }

    #[test]
    fn test_convergence_theorem_verification() {
        let g = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let target = DVector::from_vec(vec![1.0, 1.0]);
        let agent = LearningAgent::new(vec![0.0, 0.0], g, target, 0.5);
        let verification = verify_convergence_theorem(&agent, 1000, 1e-10);
        assert!(verification.is_contraction);
        assert!(verification.converged);
    }

    #[test]
    fn test_non_contractive_does_not_verify() {
        // η too large → not a contraction
        let g = DMatrix::from_row_slice(1, 1, &[1.0]);
        let target = DVector::from_vec(vec![1.0]);
        let agent = LearningAgent::new(vec![0.0], g, target, 3.0);
        let verification = verify_convergence_theorem(&agent, 1000, 1e-10);
        assert!(!verification.is_contraction);
    }

    #[test]
    fn test_joint_contraction_rate() {
        let agents: Vec<LearningAgent> = (1..=3)
            .map(|k| {
                let g = DMatrix::from_row_slice(1, 1, &[k as f64]);
                let target = DVector::from_vec(vec![1.0]);
                LearningAgent::new(vec![0.0], g, target, 0.1)
            })
            .collect();
        let system = MultiAgentSystem::new(agents);
        let rate = system.joint_contraction_rate();
        assert!(rate < 1.0);
        assert!(rate > 0.0);
    }
}
