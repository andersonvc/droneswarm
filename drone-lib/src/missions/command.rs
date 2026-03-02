//! Agent command queue for decoupled mission assignment.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::types::Position;

/// Commands that can be issued to an agent.
///
/// Commands are processed through a queue, enabling decoupled command
/// sources (UI, swarm coordinator, external API) from agent execution.
#[derive(Debug, Clone)]
pub enum AgentCommand {
    /// Navigate to waypoints in sequence, stop when done.
    GoToWaypoints(VecDeque<Position>),
    /// Follow a route indefinitely (looping).
    FollowRoute(Arc<[Position]>),
    /// Loiter at current position.
    Loiter,
    /// Stop all movement.
    Stop,
    /// Clear current mission.
    ClearMission,
    /// Set flight parameter: max velocity.
    SetMaxVelocity(f32),
    /// Set flight parameter: max acceleration.
    SetMaxAcceleration(f32),
    /// Set flight parameter: max turn rate.
    SetMaxTurnRate(f32),
    /// Set waypoint clearance distance.
    SetWaypointClearance(f32),
}

/// Queue for pending agent commands.
///
/// Commands are processed in FIFO order at the start of each tick.
#[derive(Debug, Clone, Default)]
pub struct CommandQueue {
    commands: VecDeque<AgentCommand>,
}

impl CommandQueue {
    /// Create a new empty command queue.
    pub fn new() -> Self {
        CommandQueue {
            commands: VecDeque::new(),
        }
    }

    /// Push a command to the back of the queue.
    pub fn push(&mut self, command: AgentCommand) {
        self.commands.push_back(command);
    }

    /// Pop a command from the front of the queue.
    pub fn pop(&mut self) -> Option<AgentCommand> {
        self.commands.pop_front()
    }

    /// Clear all pending commands.
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Get the number of pending commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Peek at the next command without removing it.
    pub fn peek(&self) -> Option<&AgentCommand> {
        self.commands.front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_queue_new() {
        let queue = CommandQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_command_queue_push_pop() {
        let mut queue = CommandQueue::new();

        queue.push(AgentCommand::Stop);
        queue.push(AgentCommand::Loiter);

        assert_eq!(queue.len(), 2);
        assert!(!queue.is_empty());

        // FIFO order
        assert!(matches!(queue.pop(), Some(AgentCommand::Stop)));
        assert!(matches!(queue.pop(), Some(AgentCommand::Loiter)));
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_command_queue_clear() {
        let mut queue = CommandQueue::new();

        queue.push(AgentCommand::Stop);
        queue.push(AgentCommand::Loiter);
        queue.clear();

        assert!(queue.is_empty());
    }

    #[test]
    fn test_command_queue_peek() {
        let mut queue = CommandQueue::new();

        queue.push(AgentCommand::Stop);

        // Peek doesn't remove
        assert!(matches!(queue.peek(), Some(AgentCommand::Stop)));
        assert!(matches!(queue.peek(), Some(AgentCommand::Stop)));
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_go_to_waypoints_command() {
        let waypoints: VecDeque<Position> = vec![
            Position::new(100.0, 100.0),
            Position::new(200.0, 200.0),
        ].into();

        let cmd = AgentCommand::GoToWaypoints(waypoints.clone());

        if let AgentCommand::GoToWaypoints(wps) = cmd {
            assert_eq!(wps.len(), 2);
        } else {
            panic!("Expected GoToWaypoints command");
        }
    }

    #[test]
    fn test_follow_route_command() {
        let route: Arc<[Position]> = vec![
            Position::new(100.0, 100.0),
            Position::new(200.0, 200.0),
        ].into();

        let cmd = AgentCommand::FollowRoute(route.clone());

        if let AgentCommand::FollowRoute(r) = cmd {
            assert_eq!(r.len(), 2);
        } else {
            panic!("Expected FollowRoute command");
        }
    }
}
