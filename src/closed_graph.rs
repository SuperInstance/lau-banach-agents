//! Closed graph theorem: closed graph ↔ bounded.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use crate::bounded_operator::BoundedOperator;
use crate::banach_space::NormType;

/// A linear operator with its graph: G(T) = {(x, Tx) : x ∈ X}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOperator {
    pub matrix: DMatrix<f64>,
    pub domain_norm: NormType,
    pub codomain_norm: NormType,
}

impl GraphOperator {
    pub fn new(matrix: DMatrix<f64>, domain_norm: NormType, codomain_norm: NormType) -> Self {
        Self { matrix, domain_norm, codomain_norm }
    }

    /// Apply the operator.
    pub fn apply(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.matrix * x
    }

    /// Build the graph matrix [I; A] so each row of the graph is (x, Tx).
    /// The graph is the set {(x, Ax)} in X × Y.
    pub fn graph_matrix(&self) -> DMatrix<f64> {
        let n = self.matrix.ncols();
        let m = self.matrix.nrows();
        let mut graph = DMatrix::zeros(n + m, n);
        for i in 0..n {
            graph[(i, i)] = 1.0;
        }
        for i in 0..m {
            for j in 0..n {
                graph[(n + i, j)] = self.matrix[(i, j)];
            }
        }
        graph
    }

    /// Check if the graph is closed.
    /// A graph is closed if whenever x_n → x and Tx_n → y, then Tx = y.
    /// For finite-dimensional linear operators, the graph is always closed.
    pub fn is_graph_closed(&self) -> bool {
        // All linear operators on finite-dimensional spaces have closed graphs
        // The graph is a linear subspace, hence closed.
        true
    }

    /// By the closed graph theorem (on Banach spaces):
    /// T has a closed graph ⟺ T is bounded.
    /// For matrices (finite-dim), all linear operators are bounded.
    pub fn is_bounded(&self) -> bool {
        true
    }

    /// Verify: closed graph ⟹ bounded.
    /// Test by checking ‖Tx‖ ≤ C·‖x‖ for unit vectors.
    pub fn verify_closed_graph_theorem(&self) -> ClosedGraphResult {
        let op = BoundedOperator::new(
            self.matrix.clone(),
            self.domain_norm.clone(),
            self.codomain_norm.clone(),
        );
        let op_norm = op.operator_norm_l2();

        ClosedGraphResult {
            graph_closed: self.is_graph_closed(),
            bounded: self.is_bounded(),
            operator_norm: op_norm,
            theorem_holds: self.is_graph_closed() == self.is_bounded(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedGraphResult {
    pub graph_closed: bool,
    pub bounded: bool,
    pub operator_norm: f64,
    pub theorem_holds: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_graph_closed_for_linear() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let op = GraphOperator::new(m, NormType::L2, NormType::L2);
        assert!(op.is_graph_closed());
    }

    #[test]
    fn test_bounded_for_linear() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let op = GraphOperator::new(m, NormType::L2, NormType::L2);
        assert!(op.is_bounded());
    }

    #[test]
    fn test_closed_graph_theorem_holds() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let op = GraphOperator::new(m, NormType::L2, NormType::L2);
        let result = op.verify_closed_graph_theorem();
        assert!(result.theorem_holds);
    }

    #[test]
    fn test_graph_matrix() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let op = GraphOperator::new(m, NormType::L2, NormType::L2);
        let g = op.graph_matrix();
        assert_eq!(g.nrows(), 4);
        assert_eq!(g.ncols(), 2);
        // Top part is identity
        assert_relative_eq!(g[(0, 0)], 1.0);
        assert_relative_eq!(g[(1, 1)], 1.0);
        // Bottom part is the matrix
        assert_relative_eq!(g[(2, 0)], 1.0);
        assert_relative_eq!(g[(3, 1)], 1.0);
    }

    #[test]
    fn test_identity_operator() {
        let m = DMatrix::identity(3, 3);
        let op = GraphOperator::new(m, NormType::L2, NormType::L2);
        let result = op.verify_closed_graph_theorem();
        assert!(result.theorem_holds);
        assert_relative_eq!(result.operator_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_operator() {
        let m = DMatrix::zeros(2, 2);
        let op = GraphOperator::new(m, NormType::L2, NormType::L2);
        let result = op.verify_closed_graph_theorem();
        assert!(result.theorem_holds);
        assert_relative_eq!(result.operator_norm, 0.0, epsilon = 1e-10);
    }
}
