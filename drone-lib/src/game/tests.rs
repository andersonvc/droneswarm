#[cfg(test)]
mod tests {
    use crate::agent::DroneAgent;
    use crate::game::config::GameConfig;
    use crate::game::engine::GameEngine;
    use crate::game::result::GameResult;
    use crate::game::state::{GameDrone, TargetState};
    use crate::types::{Bounds, Heading, Position};

    fn make_engine(world_size: f32, group_split_id: usize) -> GameEngine {
        let bounds = Bounds::new(world_size, world_size).unwrap();
        GameEngine::new(bounds, group_split_id)
    }

    fn make_drone(id: usize, group: u32, pos: Position, bounds: Bounds) -> GameDrone {
        let hdg = Heading::new(0.0);
        let agent = DroneAgent::new(id, pos, hdg, bounds);
        let mut d = GameDrone { id, agent, group };
        d.agent.set_group(group);
        d
    }

    #[test]
    fn test_engine_detonation_blast_radius() {
        let mut engine = make_engine(2500.0, 2);
        let config = GameConfig::default();

        // Place drone 0 (group 0) and drone 2 (group 1) within blast radius.
        engine.drones.push(make_drone(0, 0, Position::new(500.0, 500.0), engine.bounds));
        engine.drones.push(make_drone(2, 1, Position::new(500.0 + config.detonation_radius * 0.5, 500.0), engine.bounds));

        // Place drone 3 (group 1) outside blast radius.
        engine.drones.push(make_drone(3, 1, Position::new(500.0 + config.detonation_radius * 2.0, 500.0), engine.bounds));

        // Trigger detonation for drone 0.
        engine.pending_detonations.insert(0);

        // Add empty targets so the engine doesn't trigger wins immediately.
        engine.targets_a.push(TargetState { pos: Position::new(100.0, 100.0), destroyed: false });
        engine.targets_b.push(TargetState { pos: Position::new(2000.0, 2000.0), destroyed: false });

        let result = engine.tick(0.05);

        // Drone 0 (detonator) and drone 2 (within blast) should be destroyed.
        assert!(result.destroyed_ids.contains(&0));
        assert!(result.destroyed_ids.contains(&2));

        // Drone 3 (outside blast) should survive.
        assert!(!result.destroyed_ids.contains(&3));
        assert_eq!(engine.drones.len(), 1);
        assert_eq!(engine.drones[0].id, 3);
    }

    #[test]
    fn test_engine_collision_detection() {
        let mut engine = make_engine(2500.0, 2);

        // Place two drones at the exact same position (collision).
        engine.drones.push(make_drone(0, 0, Position::new(500.0, 500.0), engine.bounds));
        engine.drones.push(make_drone(1, 0, Position::new(500.0, 500.0), engine.bounds));
        // Place a third drone far away.
        engine.drones.push(make_drone(2, 1, Position::new(2000.0, 2000.0), engine.bounds));

        engine.targets_a.push(TargetState { pos: Position::new(100.0, 100.0), destroyed: false });
        engine.targets_b.push(TargetState { pos: Position::new(2000.0, 2000.0), destroyed: false });

        let result = engine.tick(0.05);

        // Drones 0 and 1 should collide.
        assert!(result.destroyed_ids.contains(&0));
        assert!(result.destroyed_ids.contains(&1));
        // Drone 2 survives.
        assert!(!result.destroyed_ids.contains(&2));
    }

    #[test]
    fn test_engine_win_conditions() {
        let mut engine = make_engine(2500.0, 1);

        engine.drones.push(make_drone(0, 0, Position::new(500.0, 500.0), engine.bounds));
        engine.drones.push(make_drone(1, 1, Position::new(2000.0, 2000.0), engine.bounds));
        engine.targets_a.push(TargetState { pos: Position::new(100.0, 100.0), destroyed: false });
        engine.targets_b.push(TargetState { pos: Position::new(2400.0, 2400.0), destroyed: false });

        // InProgress: both sides alive.
        assert_eq!(engine.check_result(), GameResult::InProgress);

        // AWins: destroy all B targets.
        engine.targets_b[0].destroyed = true;
        assert_eq!(engine.check_result(), GameResult::AWins);

        // Draw: also destroy all A targets.
        engine.targets_a[0].destroyed = true;
        assert_eq!(engine.check_result(), GameResult::Draw);

        // BWins: restore B targets, keep A destroyed.
        engine.targets_b[0].destroyed = false;
        assert_eq!(engine.check_result(), GameResult::BWins);
    }

    #[test]
    fn test_engine_deterministic_same_seed() {
        // Two engines with identical setup should produce identical results.
        fn setup_engine() -> GameEngine {
            let mut engine = make_engine(2500.0, 3);

            for i in 0..3 {
                engine.drones.push(make_drone(i, 0, Position::new(300.0 + i as f32 * 50.0, 300.0), engine.bounds));
            }
            for i in 3..6 {
                engine.drones.push(make_drone(i, 1, Position::new(2000.0 + (i - 3) as f32 * 50.0, 2000.0), engine.bounds));
            }
            engine.targets_a.push(TargetState { pos: Position::new(200.0, 200.0), destroyed: false });
            engine.targets_b.push(TargetState { pos: Position::new(2300.0, 2300.0), destroyed: false });
            engine
        }

        let mut engine1 = setup_engine();
        let mut engine2 = setup_engine();

        for _ in 0..50 {
            engine1.tick(0.4);
            engine2.tick(0.4);
        }

        assert_eq!(engine1.drones.len(), engine2.drones.len());
        for (d1, d2) in engine1.drones.iter().zip(engine2.drones.iter()) {
            assert_eq!(d1.id, d2.id);
            let p1 = d1.agent.state().pos;
            let p2 = d2.agent.state().pos;
            assert!((p1.x() - p2.x()).abs() < 0.01, "Engine determinism failed");
            assert!((p1.y() - p2.y()).abs() < 0.01, "Engine determinism failed");
        }
    }

    #[test]
    fn test_engine_tick_count_increments() {
        let mut engine = make_engine(2500.0, 0);
        engine.targets_a.push(TargetState { pos: Position::new(100.0, 100.0), destroyed: false });
        engine.targets_b.push(TargetState { pos: Position::new(2000.0, 2000.0), destroyed: false });

        assert_eq!(engine.tick_count, 0);
        engine.tick(0.05);
        assert_eq!(engine.tick_count, 1);
        engine.tick(0.05);
        assert_eq!(engine.tick_count, 2);
    }

    #[test]
    fn test_engine_protected_zone_update() {
        let mut engine = make_engine(2500.0, 0);
        engine.targets_a.push(TargetState { pos: Position::new(100.0, 100.0), destroyed: false });
        engine.targets_a.push(TargetState { pos: Position::new(200.0, 200.0), destroyed: false });
        engine.targets_b.push(TargetState { pos: Position::new(2000.0, 2000.0), destroyed: false });

        engine.update_protected_zones();

        // Group 0 should have 2 protected zones (the A targets).
        assert_eq!(engine.protected_zones.get(&0).unwrap().len(), 2);
        // Group 1 should have 1 protected zone (the B target).
        assert_eq!(engine.protected_zones.get(&1).unwrap().len(), 1);

        // Destroy one A target, update zones.
        engine.targets_a[0].destroyed = true;
        engine.update_protected_zones();

        assert_eq!(engine.protected_zones.get(&0).unwrap().len(), 1);
    }

    #[test]
    fn test_engine_drone_loss_subtracts_from_targets() {
        use crate::game::result::check_win_condition;

        // B loses all drones. A has 2 drones, B has 3 targets.
        // Effective B targets = 3 - 2 = 1 > 0, but A still has drones → A wins.
        assert_eq!(
            check_win_condition(2, 3, 2, 0),
            GameResult::AWins,
        );

        // B loses all drones. A has 5 drones, B has 3 targets.
        // Effective B targets = 3 - 5 = 0. A still has targets → A wins.
        assert_eq!(
            check_win_condition(2, 3, 5, 0),
            GameResult::AWins,
        );

        // A loses all drones. B has 1 drone, A has 1 target.
        // Effective A targets = 1 - 1 = 0. B has targets → B wins.
        assert_eq!(
            check_win_condition(1, 2, 0, 1),
            GameResult::BWins,
        );

        // Both lose all drones. A has 3 targets, B has 3 targets.
        // Effective A = 3 - 0 = 3, Effective B = 3 - 0 = 3.
        // Both have targets, both have 0 drones → Draw.
        assert_eq!(
            check_win_condition(3, 3, 0, 0),
            GameResult::Draw,
        );

        // A loses all drones. B has 10 drones, A has 5 targets.
        // Effective A = 5 - 10 = 0. B has targets → B wins.
        assert_eq!(
            check_win_condition(5, 2, 0, 10),
            GameResult::BWins,
        );

        // Both lose all drones, both have 0 targets → Draw.
        assert_eq!(
            check_win_condition(0, 0, 0, 0),
            GameResult::Draw,
        );
    }
}
