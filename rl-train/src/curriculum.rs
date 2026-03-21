/// Curriculum stage definition.
#[derive(Clone)]
pub struct CurriculumStage {
    pub drones_per_side: u32,
    pub targets_per_side: u32,
    pub world_size: f32,
}

/// Win-rate-based curriculum that advances when the agent is competent,
/// and demotes when performance collapses.
///
/// Advances to the next stage when eval win rate exceeds the threshold.
/// Demotes to the previous stage when win rate drops below the demotion
/// threshold for too many consecutive evals.
/// During transitions, blends old/new stages across envs over a window
/// of training updates to avoid abrupt all-envs switches.
pub struct Curriculum {
    stages: Vec<CurriculumStage>,
    /// Current stage index (0-based).
    pub current_stage: usize,
    /// Win rate threshold to advance to the next stage.
    pub advance_threshold: f32,
    /// Number of consecutive evals above threshold before advancing.
    /// Prevents advancing on a lucky single eval.
    pub required_consecutive: usize,
    consecutive_above: usize,
    /// Updates since last stage advance (for blending).
    updates_since_advance: usize,
    /// Number of updates to blend over when advancing.
    blend_updates: usize,
    /// Win rate below which demotion counter increments.
    pub demotion_threshold: f32,
    /// Number of consecutive evals below demotion threshold before demoting.
    pub demotion_consecutive: usize,
    /// Counter of consecutive evals below demotion threshold.
    consecutive_below_demotion: usize,
}

impl Curriculum {
    pub fn new(final_drones: u32, final_targets: u32) -> Self {
        Curriculum {
            stages: vec![
                CurriculumStage { drones_per_side: 4,  targets_per_side: 2, world_size: 2500.0 },
                CurriculumStage { drones_per_side: 8,  targets_per_side: 4, world_size: 5000.0 },
                CurriculumStage { drones_per_side: 16, targets_per_side: 6, world_size: 7500.0 },
                CurriculumStage { drones_per_side: final_drones, targets_per_side: final_targets, world_size: 10000.0 },
            ],
            current_stage: 0,
            advance_threshold: 0.70,
            required_consecutive: 5,
            consecutive_above: 0,
            updates_since_advance: 100, // start past blend window
            blend_updates: 50,
            demotion_threshold: 0.30,
            demotion_consecutive: 10,
            consecutive_below_demotion: 0,
        }
    }

    /// Report an eval win rate.
    ///
    /// Returns:
    /// - `Some(true)` if the curriculum advanced to the next stage.
    /// - `Some(false)` if the curriculum demoted to the previous stage.
    /// - `None` if no stage change occurred.
    pub fn report_eval(&mut self, win_rate: f32) -> Option<bool> {
        // --- Advancement logic ---
        if self.current_stage < self.stages.len() - 1 {
            if win_rate >= self.advance_threshold {
                self.consecutive_above += 1;
            } else {
                self.consecutive_above = 0;
            }

            if self.consecutive_above >= self.required_consecutive {
                self.current_stage += 1;
                self.consecutive_above = 0;
                self.consecutive_below_demotion = 0;
                self.updates_since_advance = 0;
                return Some(true);
            }
        }

        // --- Demotion logic ---
        if win_rate < self.demotion_threshold {
            self.consecutive_below_demotion += 1;
        } else {
            self.consecutive_below_demotion = 0;
        }

        if self.consecutive_below_demotion >= self.demotion_consecutive && self.current_stage > 0 {
            self.current_stage -= 1;
            self.consecutive_above = 0;
            self.consecutive_below_demotion = 0;
            self.updates_since_advance = 0;
            return Some(false);
        }

        None
    }

    /// Call once per PPO update to track blend progress.
    pub fn tick_update(&mut self) {
        self.updates_since_advance += 1;
    }

    /// Get config for a specific env. During blending window after an advance,
    /// some envs still use the previous stage for smooth transition.
    pub fn config_for_env(&self, env_seed: u64) -> &CurriculumStage {
        if self.current_stage == 0 || self.updates_since_advance >= self.blend_updates {
            return &self.stages[self.current_stage];
        }

        // In blend window: gradually shift envs to new stage.
        let blend_pct = self.updates_since_advance as f32 / self.blend_updates as f32;
        if (env_seed % 1000) < (blend_pct * 1000.0) as u64 {
            &self.stages[self.current_stage]
        } else {
            &self.stages[self.current_stage - 1]
        }
    }

    /// Current primary stage config (for logging).
    pub fn current(&self) -> &CurriculumStage {
        &self.stages[self.current_stage]
    }

    /// Whether we're at the final stage.
    pub fn is_final(&self) -> bool {
        self.current_stage >= self.stages.len() - 1
    }

    /// Stage name for logging.
    pub fn stage_label(&self) -> String {
        let s = &self.stages[self.current_stage];
        format!("{}v{}/{}t", s.drones_per_side, s.drones_per_side, s.targets_per_side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_at_stage_0() {
        let c = Curriculum::new(24, 6);
        assert_eq!(c.current_stage, 0);
        assert_eq!(c.current().drones_per_side, 4);
    }

    #[test]
    fn advances_after_consecutive_wins() {
        let mut c = Curriculum::new(24, 6);
        // Four evals above threshold — not enough
        for _ in 0..4 {
            assert_eq!(c.report_eval(0.75), None);
            assert_eq!(c.current_stage, 0);
        }
        // Fifth consecutive — advances
        assert_eq!(c.report_eval(0.75), Some(true));
        assert_eq!(c.current_stage, 1);
        assert_eq!(c.current().drones_per_side, 8);
    }

    #[test]
    fn resets_consecutive_on_bad_eval() {
        let mut c = Curriculum::new(24, 6);
        assert_eq!(c.report_eval(0.75), None); // 1 consecutive
        assert_eq!(c.report_eval(0.40), None); // reset
        assert_eq!(c.report_eval(0.75), None); // 1 consecutive again
        assert_eq!(c.current_stage, 0);
    }

    #[test]
    fn stops_at_final_stage() {
        let mut c = Curriculum::new(24, 6);
        for _ in 0..20 {
            c.report_eval(0.75);
        }
        assert!(c.is_final());
        assert_eq!(c.current().drones_per_side, 24);
        // Further reports don't advance
        assert_eq!(c.report_eval(0.75), None);
    }

    #[test]
    fn blending_during_transition() {
        let mut c = Curriculum::new(24, 6);
        for _ in 0..5 {
            c.report_eval(0.75);
        }
        // advances to stage 1
        assert_eq!(c.current_stage, 1);
        assert_eq!(c.updates_since_advance, 0);

        // Right after advance: most envs still use old stage
        let old_count = (0..1000).filter(|&s| c.config_for_env(s).drones_per_side == 4).count();
        assert!(old_count > 900, "Most envs should still be old stage");

        // After blend window: all envs use new stage
        for _ in 0..50 { c.tick_update(); }
        let new_count = (0..1000).filter(|&s| c.config_for_env(s).drones_per_side == 8).count();
        assert_eq!(new_count, 1000);
    }

    #[test]
    fn test_demotion_after_collapse() {
        let mut c = Curriculum::new(24, 6);

        // Advance to stage 1.
        for _ in 0..5 {
            c.report_eval(0.75);
        }
        assert_eq!(c.current_stage, 1);
        assert_eq!(c.current().drones_per_side, 8);

        // 9 consecutive low win rates — not enough to demote yet.
        for _ in 0..9 {
            assert_eq!(c.report_eval(0.20), None);
            assert_eq!(c.current_stage, 1, "Should not demote before 10 consecutive");
        }

        // 10th consecutive low win rate — triggers demotion.
        assert_eq!(c.report_eval(0.20), Some(false));
        assert_eq!(c.current_stage, 0);
        assert_eq!(c.current().drones_per_side, 4);
    }

    #[test]
    fn demotion_resets_on_recovery() {
        let mut c = Curriculum::new(24, 6);

        // Advance to stage 1.
        for _ in 0..5 {
            c.report_eval(0.75);
        }
        assert_eq!(c.current_stage, 1);

        // 5 low evals, then one recovery, then 5 more low evals.
        for _ in 0..5 {
            c.report_eval(0.20);
        }
        c.report_eval(0.50); // above demotion threshold, resets counter
        for _ in 0..5 {
            c.report_eval(0.20);
        }
        // Only 5 consecutive (not 10), so no demotion.
        assert_eq!(c.current_stage, 1);
    }

    #[test]
    fn no_demotion_at_stage_0() {
        let mut c = Curriculum::new(24, 6);
        assert_eq!(c.current_stage, 0);

        // Report many low win rates at stage 0 — cannot demote below 0.
        for _ in 0..20 {
            assert_eq!(c.report_eval(0.10), None);
        }
        assert_eq!(c.current_stage, 0);
    }
}
