/// Errors that can occur in drone operations
#[derive(Debug, Clone, PartialEq)]
pub enum DroneError {
    /// Invalid bounds (width or height <= 0)
    InvalidBounds { width: f32, height: f32 },
    /// Invalid flight parameter
    InvalidFlightParam { param: &'static str, value: f32 },
}

impl std::fmt::Display for DroneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DroneError::InvalidBounds { width, height } => {
                write!(f, "Invalid bounds: width={}, height={}", width, height)
            }
            DroneError::InvalidFlightParam { param, value } => {
                write!(f, "Invalid flight parameter {}: {}", param, value)
            }
        }
    }
}

impl std::error::Error for DroneError {}

pub type DroneResult<T> = Result<T, DroneError>;
