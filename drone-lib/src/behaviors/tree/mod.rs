//! Behavior tree infrastructure.
//!
//! This module provides the core behavior tree types:
//! - `BehaviorNode` trait for all nodes
//! - `BehaviorStatus` for tick results
//! - `BehaviorContext` for passing data during ticks
//! - Composite nodes: `Sequence`, `Selector`, `Parallel`

pub mod composite;
pub mod node;

pub use composite::{Parallel, Selector, Sequence};
pub use node::{BehaviorContext, BehaviorNode, BehaviorStatus};
