//! Neural network components.

pub mod layer;
pub mod init;
pub mod policy;

pub use layer::DenseLayer;
pub use init::orthogonal_init;
pub use policy::PolicyNet;