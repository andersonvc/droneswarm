//! Inter-drone communication types.
//!
//! Provides data types for swarm coordination:
//!
//! - [`FormationCommand`] - commands for formation control
//! - [`FormationSlot`] - position assignment within a formation

pub mod message;

pub use message::{FormationCommand, FormationSlot, PathProgress, VelocityConsensus};
