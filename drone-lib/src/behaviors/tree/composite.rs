//! Composite behavior tree nodes (Sequence, Selector, Parallel).

use super::node::{BehaviorContext, BehaviorNode, BehaviorStatus};

/// Sequence node: runs children in order until one fails.
///
/// - If a child returns Success, moves to the next child.
/// - If a child returns Running, returns Running.
/// - If a child returns Failure, returns Failure and resets.
/// - If all children succeed, returns Success.
#[derive(Debug)]
pub struct Sequence {
    name: String,
    children: Vec<Box<dyn BehaviorNode>>,
    current: usize,
}

impl Sequence {
    /// Create a new sequence node.
    pub fn new(name: impl Into<String>, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Sequence {
            name: name.into(),
            children,
            current: 0,
        }
    }
}

impl BehaviorNode for Sequence {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        while self.current < self.children.len() {
            match self.children[self.current].tick(ctx) {
                BehaviorStatus::Success => {
                    self.current += 1;
                }
                BehaviorStatus::Running => {
                    return BehaviorStatus::Running;
                }
                BehaviorStatus::Failure => {
                    self.current = 0;
                    return BehaviorStatus::Failure;
                }
            }
        }
        self.current = 0;
        BehaviorStatus::Success
    }

    fn reset(&mut self) {
        self.current = 0;
        for child in &mut self.children {
            child.reset();
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Selector node: runs children until one succeeds.
///
/// - If a child returns Failure, moves to the next child.
/// - If a child returns Running, returns Running.
/// - If a child returns Success, returns Success.
/// - If all children fail, returns Failure.
#[derive(Debug)]
pub struct Selector {
    name: String,
    children: Vec<Box<dyn BehaviorNode>>,
    current: usize,
}

impl Selector {
    /// Create a new selector node.
    pub fn new(name: impl Into<String>, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Selector {
            name: name.into(),
            children,
            current: 0,
        }
    }
}

impl BehaviorNode for Selector {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        while self.current < self.children.len() {
            match self.children[self.current].tick(ctx) {
                BehaviorStatus::Failure => {
                    self.current += 1;
                }
                BehaviorStatus::Running => {
                    return BehaviorStatus::Running;
                }
                BehaviorStatus::Success => {
                    self.current = 0;
                    return BehaviorStatus::Success;
                }
            }
        }
        self.current = 0;
        BehaviorStatus::Failure
    }

    fn reset(&mut self) {
        self.current = 0;
        for child in &mut self.children {
            child.reset();
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Parallel node: runs all children simultaneously.
///
/// Returns based on success/failure threshold:
/// - If success_threshold children succeed, returns Success.
/// - If more than (children - success_threshold) fail, returns Failure.
/// - Otherwise, returns Running.
#[derive(Debug)]
pub struct Parallel {
    name: String,
    children: Vec<Box<dyn BehaviorNode>>,
    success_threshold: usize,
}

impl Parallel {
    /// Create a new parallel node.
    ///
    /// # Arguments
    /// * `name` - Node name for debugging
    /// * `children` - Child nodes to run in parallel
    /// * `success_threshold` - How many children must succeed for overall success
    pub fn new(
        name: impl Into<String>,
        children: Vec<Box<dyn BehaviorNode>>,
        success_threshold: usize,
    ) -> Self {
        Parallel {
            name: name.into(),
            children,
            success_threshold,
        }
    }

    /// Create a parallel node that requires all children to succeed.
    pub fn all(name: impl Into<String>, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        let threshold = children.len();
        Self::new(name, children, threshold)
    }

    /// Create a parallel node that requires any child to succeed.
    pub fn any(name: impl Into<String>, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Self::new(name, children, 1)
    }
}

impl BehaviorNode for Parallel {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        let mut successes = 0;
        let mut failures = 0;

        for child in &mut self.children {
            match child.tick(ctx) {
                BehaviorStatus::Success => successes += 1,
                BehaviorStatus::Failure => failures += 1,
                BehaviorStatus::Running => {}
            }
        }

        if successes >= self.success_threshold {
            BehaviorStatus::Success
        } else if failures > self.children.len() - self.success_threshold {
            BehaviorStatus::Failure
        } else {
            BehaviorStatus::Running
        }
    }

    fn reset(&mut self) {
        for child in &mut self.children {
            child.reset();
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Bounds, DronePerfFeatures, Heading, Position, State, Velocity};

    fn test_perf() -> &'static DronePerfFeatures {
        Box::leak(Box::new(DronePerfFeatures::default()))
    }

    /// Test node that always returns a fixed status.
    #[derive(Debug)]
    struct FixedNode {
        status: BehaviorStatus,
        tick_count: usize,
    }

    impl FixedNode {
        fn new(status: BehaviorStatus) -> Self {
            FixedNode { status, tick_count: 0 }
        }
    }

    impl BehaviorNode for FixedNode {
        fn tick(&mut self, _ctx: &mut BehaviorContext) -> BehaviorStatus {
            self.tick_count += 1;
            self.status
        }

        fn reset(&mut self) {
            self.tick_count = 0;
        }

        fn name(&self) -> &str {
            "FixedNode"
        }
    }

    fn create_test_context() -> (State, Bounds) {
        let state = State {
            pos: Position::new(500.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        (state, bounds)
    }

    #[test]
    fn test_sequence_all_success() {
        let (state, bounds) = create_test_context();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        let mut seq = Sequence::new("test", vec![
            Box::new(FixedNode::new(BehaviorStatus::Success)),
            Box::new(FixedNode::new(BehaviorStatus::Success)),
        ]);

        assert_eq!(seq.tick(&mut ctx), BehaviorStatus::Success);
    }

    #[test]
    fn test_sequence_stops_on_failure() {
        let (state, bounds) = create_test_context();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        let mut seq = Sequence::new("test", vec![
            Box::new(FixedNode::new(BehaviorStatus::Success)),
            Box::new(FixedNode::new(BehaviorStatus::Failure)),
            Box::new(FixedNode::new(BehaviorStatus::Success)),
        ]);

        assert_eq!(seq.tick(&mut ctx), BehaviorStatus::Failure);
    }

    #[test]
    fn test_sequence_returns_running() {
        let (state, bounds) = create_test_context();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        let mut seq = Sequence::new("test", vec![
            Box::new(FixedNode::new(BehaviorStatus::Success)),
            Box::new(FixedNode::new(BehaviorStatus::Running)),
        ]);

        assert_eq!(seq.tick(&mut ctx), BehaviorStatus::Running);
    }

    #[test]
    fn test_selector_stops_on_success() {
        let (state, bounds) = create_test_context();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        let mut sel = Selector::new("test", vec![
            Box::new(FixedNode::new(BehaviorStatus::Failure)),
            Box::new(FixedNode::new(BehaviorStatus::Success)),
            Box::new(FixedNode::new(BehaviorStatus::Success)),
        ]);

        assert_eq!(sel.tick(&mut ctx), BehaviorStatus::Success);
    }

    #[test]
    fn test_selector_all_fail() {
        let (state, bounds) = create_test_context();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        let mut sel = Selector::new("test", vec![
            Box::new(FixedNode::new(BehaviorStatus::Failure)),
            Box::new(FixedNode::new(BehaviorStatus::Failure)),
        ]);

        assert_eq!(sel.tick(&mut ctx), BehaviorStatus::Failure);
    }

    #[test]
    fn test_selector_returns_running() {
        let (state, bounds) = create_test_context();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        let mut sel = Selector::new("test", vec![
            Box::new(FixedNode::new(BehaviorStatus::Failure)),
            Box::new(FixedNode::new(BehaviorStatus::Running)),
        ]);

        assert_eq!(sel.tick(&mut ctx), BehaviorStatus::Running);
    }

    #[test]
    fn test_parallel_threshold() {
        let (state, bounds) = create_test_context();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        // Need 2 successes
        let mut par = Parallel::new("test", vec![
            Box::new(FixedNode::new(BehaviorStatus::Success)),
            Box::new(FixedNode::new(BehaviorStatus::Success)),
            Box::new(FixedNode::new(BehaviorStatus::Failure)),
        ], 2);

        assert_eq!(par.tick(&mut ctx), BehaviorStatus::Success);
    }

    #[test]
    fn test_parallel_fails_when_threshold_impossible() {
        let (state, bounds) = create_test_context();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        // Need 2 successes but only 1 can succeed
        let mut par = Parallel::new("test", vec![
            Box::new(FixedNode::new(BehaviorStatus::Success)),
            Box::new(FixedNode::new(BehaviorStatus::Failure)),
            Box::new(FixedNode::new(BehaviorStatus::Failure)),
        ], 2);

        assert_eq!(par.tick(&mut ctx), BehaviorStatus::Failure);
    }

    #[test]
    fn test_reset_propagates() {
        let mut seq = Sequence::new("test", vec![
            Box::new(FixedNode::new(BehaviorStatus::Success)),
        ]);

        let (state, bounds) = create_test_context();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        seq.tick(&mut ctx);
        seq.reset();

        // After reset, internal state should be cleared
        // (current index = 0, children reset)
    }
}
