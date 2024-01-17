#[derive(Debug)]
pub enum ObjectiveType {
    ReachWaypoint,
    FollowTarget,
    Loiter,
    Sleep,
}

#[derive(Debug)]
pub struct Objective {
    pub task: ObjectiveType,
    pub waypoints: Option<Vec<(f32, f32)>>,
    pub targets: Option<Vec<(f32, f32)>>,
}

#[derive(Debug)]
pub struct State {
    pub hdg: f32,
    pub pos: (f32, f32),
    pub vel: (f32, f32),
    pub acc: (f32, f32),
}
