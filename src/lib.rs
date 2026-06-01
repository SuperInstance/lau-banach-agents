//! # lau-banach-agents
//!
//! Banach space theory for agents — the functional analysis framework.
//!
//! Provides core structures and theorems from functional analysis:
//! - Banach spaces (complete normed vector spaces)
//! - Bounded linear operators
//! - Banach fixed point theorem (contractions)
//! - Hahn-Banach theorem (functional extension & separation)
//! - Open mapping theorem
//! - Closed graph theorem
//! - Uniform boundedness principle
//! - Dual spaces and weak topology
//! - Riesz representation (Lᵖ duality)
//! - Spectral theory for bounded operators
//! - Application: iterative agent learning convergence

pub mod banach_space;
pub mod bounded_operator;
pub mod fixed_point;
pub mod hahn_banach;
pub mod open_mapping;
pub mod closed_graph;
pub mod uniform_boundedness;
pub mod dual_space;
pub mod weak_topology;
pub mod riesz_representation;
pub mod spectral;
pub mod agent_learning;

pub use banach_space::*;
pub use bounded_operator::*;
pub use fixed_point::*;
pub use hahn_banach::*;
pub use open_mapping::*;
pub use closed_graph::*;
pub use uniform_boundedness::*;
pub use dual_space::*;
pub use weak_topology::*;
pub use riesz_representation::*;
pub use spectral::*;
pub use agent_learning::*;
