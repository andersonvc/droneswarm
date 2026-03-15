//! Shared game engine: the authoritative game loop for drone swarm simulation.
//!
//! Both `sim_runner` (RL training) and `wasm-lib` (WASM frontend) delegate to
//! `GameEngine::tick()` for physics, detonation, collision, and win-condition logic.

pub mod collision;
pub mod config;
pub mod engine;
pub mod obs_layout;
pub mod patrol;
pub mod result;
pub mod rng;
pub mod state;
pub mod task_processing;

#[cfg(feature = "inference")]
pub mod action_mapping;
pub mod obs_encoding;
pub mod reward;

#[cfg(test)]
mod tests;

pub use config::GameConfig;
pub use engine::{GameEngine, TickResult};
pub use result::GameResult;
pub use state::{GameDrone, TargetState};
