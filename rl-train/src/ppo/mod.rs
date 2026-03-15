//! Proximal Policy Optimization (PPO) algorithm components.

pub mod gae;
pub mod backward;

pub use gae::{compute_gae, Transition};
pub use backward::PolicyNetPPOExt;