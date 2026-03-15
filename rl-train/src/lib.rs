pub mod network;
pub mod ppo;

// Backward compatibility: re-export everything under the mlp module
pub mod mlp {
    pub use crate::network::*;
    pub use crate::ppo::*;
}
