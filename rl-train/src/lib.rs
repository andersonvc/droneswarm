pub mod checkpoint_pool;
pub mod curriculum;
pub mod network;
pub mod normalize;
pub mod ppo;

// Backward compatibility: re-export everything under the mlp module
pub mod mlp {
    pub use crate::network::*;
    pub use crate::normalize::*;
    pub use crate::ppo::*;
}
