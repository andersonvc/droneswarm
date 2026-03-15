use crate::agent::DroneAgent;
use crate::types::Position;

/// State of a ground target.
#[derive(Debug, Clone)]
pub struct TargetState {
    pub pos: Position,
    pub destroyed: bool,
}

/// A drone within the game engine.
pub struct GameDrone {
    pub id: usize,
    pub agent: DroneAgent,
    pub group: u32,
}
