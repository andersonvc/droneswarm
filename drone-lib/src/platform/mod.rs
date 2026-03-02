//! Platform module - physical drone characteristics and kinematics.
//!
//! This module defines the `Platform` trait and implementations for different
//! vehicle types. A platform handles the physics of the vehicle but does not
//! make decisions about navigation.

pub mod generic;
pub mod traits;

pub use generic::{
    GenericPlatform, DEFAULT_MAX_ACCELERATION, DEFAULT_MAX_TURN_RATE, DEFAULT_MAX_VELOCITY,
    MIN_TURN_RATE,
};
pub use traits::Platform;
