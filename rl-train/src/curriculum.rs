/// Curriculum stage definition.
struct CurriculumStage {
    /// Progress fraction where this stage begins.
    threshold: f32,
    drones_per_side: u32,
    targets_per_side: u32,
}

/// Build the fixed stage ladder scaled to the given final counts.
fn stages(final_drones: u32, final_targets: u32) -> [CurriculumStage; 4] {
    [
        CurriculumStage { threshold: 0.00, drones_per_side: 4, targets_per_side: 2 },
        CurriculumStage { threshold: 0.15, drones_per_side: 8, targets_per_side: 3 },
        CurriculumStage { threshold: 0.35, drones_per_side: 16, targets_per_side: 4 },
        CurriculumStage { threshold: 0.55, drones_per_side: final_drones, targets_per_side: final_targets },
    ]
}

const BLEND_WINDOW: f32 = 0.05;

/// Get curriculum config with smooth blending between stages.
///
/// When crossing a stage boundary, blend old/new stages over a 5% window.
/// Each env independently decides its stage based on:
///   `env_seed % 1000 < blend_pct * 1000`
///
/// Returns `(drones_per_side, targets_per_side)` for a specific env.
pub fn curriculum_config_smooth(
    progress: f32,
    env_seed: u64,
    final_drones: u32,
    final_targets: u32,
) -> (u32, u32) {
    let s = stages(final_drones, final_targets);

    // Find the index of the current (or most recent) stage.
    let mut idx = 0;
    for (i, stage) in s.iter().enumerate() {
        if progress >= stage.threshold {
            idx = i;
        }
    }

    // Before the first transition, always use the first stage.
    if idx == 0 {
        return (s[0].drones_per_side, s[0].targets_per_side);
    }

    let threshold = s[idx].threshold;
    let within_blend = progress < threshold + BLEND_WINDOW;

    if within_blend {
        // Compute blend percentage clamped to [0, 1].
        let blend_pct = ((progress - threshold) / BLEND_WINDOW).clamp(0.0, 1.0);

        if (env_seed % 1000) < (blend_pct * 1000.0) as u64 {
            // Use new stage
            (s[idx].drones_per_side, s[idx].targets_per_side)
        } else {
            // Use previous stage
            (s[idx - 1].drones_per_side, s[idx - 1].targets_per_side)
        }
    } else {
        // Fully past the blend window — use the new stage.
        (s[idx].drones_per_side, s[idx].targets_per_side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn before_first_transition() {
        let (d, t) = curriculum_config_smooth(0.0, 0, 24, 6);
        assert_eq!((d, t), (4, 2));

        let (d, t) = curriculum_config_smooth(0.10, 999, 24, 6);
        assert_eq!((d, t), (4, 2));
    }

    #[test]
    fn well_past_last_stage() {
        let (d, t) = curriculum_config_smooth(0.80, 0, 24, 6);
        assert_eq!((d, t), (24, 6));
    }

    #[test]
    fn blending_at_boundary() {
        // Right at the 0.15 threshold: blend_pct = 0.0, so always old stage.
        let (d, t) = curriculum_config_smooth(0.15, 0, 24, 6);
        assert_eq!((d, t), (4, 2));

        // Midway through blend window (0.15 + 0.025 = 0.175): blend_pct = 0.5
        // seed % 1000 == 0 < 500 => new stage
        let (d, t) = curriculum_config_smooth(0.175, 0, 24, 6);
        assert_eq!((d, t), (8, 3));

        // seed % 1000 == 999 >= 500 => old stage
        let (d, t) = curriculum_config_smooth(0.175, 999, 24, 6);
        assert_eq!((d, t), (4, 2));
    }

    #[test]
    fn fully_past_blend_window() {
        // 0.15 + 0.05 = 0.20, fully past the blend
        let (d, t) = curriculum_config_smooth(0.21, 999, 24, 6);
        assert_eq!((d, t), (8, 3));
    }

    #[test]
    fn final_stage_reached() {
        let (d, t) = curriculum_config_smooth(0.55, 0, 24, 6);
        // blend_pct = 0.0 at exact threshold, so previous stage
        assert_eq!((d, t), (16, 4));

        // Past the blend window
        let (d, t) = curriculum_config_smooth(0.61, 0, 24, 6);
        assert_eq!((d, t), (24, 6));
    }
}
