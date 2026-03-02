//! Condition nodes for behavior trees.
//!
//! Condition nodes are leaf nodes that check conditions and return
//! Success or Failure without modifying the context.

pub mod collision;

pub use collision::{CollisionImminent, HasTarget};
