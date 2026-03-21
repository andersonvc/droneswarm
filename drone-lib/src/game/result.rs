/// Outcome of a simulation episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    /// Group A wins (has effective targets remaining, B does not).
    AWins,
    /// Group B wins (has effective targets remaining, A does not).
    BWins,
    /// Both sides have 0 effective targets, or time limit reached.
    Draw,
    /// Game is still in progress.
    InProgress,
}

/// Check win condition based on alive counts.
///
/// When a side loses all drones, the remaining enemy drones each destroy
/// one of that side's targets (down to 0). The winner is determined by
/// who still has effective targets. Draw if both reach 0.
pub fn check_win_condition(
    targets_a_alive: usize,
    targets_b_alive: usize,
    drones_a_alive: usize,
    drones_b_alive: usize,
) -> GameResult {
    // Game continues if both sides have drones and targets.
    if targets_a_alive > 0 && targets_b_alive > 0 && drones_a_alive > 0 && drones_b_alive > 0 {
        return GameResult::InProgress;
    }

    // When a side loses all drones, remaining enemy drones
    // effectively destroy that side's targets (1 drone = 1 target).
    let eff_a = if drones_a_alive == 0 {
        targets_a_alive.saturating_sub(drones_b_alive)
    } else {
        targets_a_alive
    };
    let eff_b = if drones_b_alive == 0 {
        targets_b_alive.saturating_sub(drones_a_alive)
    } else {
        targets_b_alive
    };

    let a_eliminated = eff_a == 0;
    let b_eliminated = eff_b == 0;

    if a_eliminated && b_eliminated {
        GameResult::Draw
    } else if b_eliminated && !a_eliminated {
        GameResult::AWins
    } else if a_eliminated && !b_eliminated {
        GameResult::BWins
    } else {
        // Both have effective targets. The side with more wins.
        if eff_a > eff_b {
            GameResult::AWins
        } else if eff_b > eff_a {
            GameResult::BWins
        } else {
            GameResult::Draw
        }
    }
}
