# Leader-Follower Formation with Waypoint Tracking

## Overview

The drone swarm implements a leader-follower formation system where:
- **Leader** (lowest ID drone): Follows waypoints independently using path planning
- **Followers**: Track dynamically-computed formation positions relative to the leader

```
                    Leader (follows waypoints)
                       ▲
                      / \
                     /   \
            Follower     Follower (track formation slots)
```

---

## 1. Leader Path Planning

### Algorithm: NLGL (Nonlinear Guidance Law)

The leader uses NLGL with Hermite spline smoothing for waypoint following.

**Location:** `drone-lib/src/missions/planner.rs`

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. Build Hermite spline from current position to waypoints │
│  2. Compute speed-adaptive lookahead distance               │
│  3. Find lookahead point on spline                          │
│  4. Steer toward lookahead point                            │
└─────────────────────────────────────────────────────────────┘
```

### Hermite Spline Construction

```rust
// Control points:
P0 = drone_position (origin)
P1 = waypoint_1

// Tangents:
T0 = drone_velocity_direction * tangent_scale  // Entry tangent
T1 = (waypoint_2 - waypoint_1).normalized * tangent_scale  // Exit tangent

// Hermite basis functions for point at parameter t ∈ [0,1]:
H00(t) = 2t³ - 3t² + 1    // P0 weight
H10(t) = t³ - 2t² + t     // T0 weight
H01(t) = -2t³ + 3t²       // P1 weight
H11(t) = t³ - t²          // T1 weight

Point(t) = H00*P0 + H10*T0 + H01*P1 + H11*T1
```

### Lookahead Calculation

```rust
// Speed-adaptive lookahead
lookahead_distance = base_lookahead + speed * speed_lookahead_factor
                   = 50.0 + speed * 1.0  // ~1 second ahead at current speed

// Convert to spline parameter
lookahead_t = (lookahead_distance / distance_to_waypoint).clamp(0.1, 0.9)

// Get lookahead point on spline
lookahead_point = hermite_point(P0, P1, T0, T1, lookahead_t)

// NLGL: simply steer toward lookahead point
desired_heading = lookahead_point.heading()
```

### Why NLGL?

| Feature | NLGL | Stanley Controller |
|---------|------|-------------------|
| Designed for | Aircraft | Ground vehicles |
| Tuning params | 1 (lookahead) | 2+ (gains) |
| Path convergence | Exponential | Can oscillate |
| Curved paths | Natural handling | Requires adaptation |

---

## 2. Follower Control

### Algorithm: Proportional Navigation

Followers use proportional navigation to intercept and track their formation slot.

**Location:** `drone-lib/src/agent/drone_agent.rs` (lines 108-198)

### Core Equation

```
desired_velocity = target_velocity + K × position_error
```

Where:
- `target_velocity` = leader's current velocity
- `position_error` = vector from follower to its formation slot
- `K` = proportional gain (varies with distance)

### Gain Schedule

```rust
let gain = if dist > 100.0 {
    0.8   // Aggressive catch-up
} else if dist > 30.0 {
    0.5   // Moderate approach
} else {
    0.3   // Gentle station-keeping
};
```

### Formation Slot Calculation

```rust
// Slot offset rotated by leader's heading
target_x = center.x + (slot.offset.x * cos(heading) - slot.offset.y * sin(heading))
target_y = center.y + (slot.offset.x * sin(heading) + slot.offset.y * cos(heading))
```

### Special Cases

#### 1. Follower Ahead of Formation
When a follower overshoots or spawns ahead:

```rust
// Detect: position_error · leader_direction < -10
if is_ahead {
    // Turn perpendicular and slow down
    desired_heading = formation_heading + π/2
    speed = target_speed * slowdown_factor  // Let formation catch up
}
```

#### 2. Leader Stopped
```rust
if dist < 10.0 && leader_speed < 5.0 {
    // Stop at position, match heading
    desired_heading = formation_heading
    desired_speed = 0.0
}
```

#### 3. Approaching Slowly
```rust
// Scale approach speed by distance for smooth deceleration
approach_speed = min(dist * 0.5, 30.0)
```

---

## 3. Platform Control

### Fixed-Wing Dynamics

**Location:** `drone-lib/src/platform/fixed_wing.rs`

The `apply_steering(heading, speed, dt)` function enforces physical constraints:

```rust
// Turn rate scales with velocity² (aerodynamic reality)
turn_rate = max(max_turn_rate * (speed/max_speed)², min_turn_rate)
actual_turn = clamp(heading_error, -turn_rate * dt, +turn_rate * dt)

// Acceleration is bounded (can accelerate AND decelerate)
speed_error = target_speed - current_speed
acceleration = clamp(speed_error, -max_acc, +max_acc)
new_speed = clamp(current_speed + acceleration * dt, 0, max_speed)

// Velocity follows heading
velocity = heading_to_vector(heading) * new_speed
position += velocity * dt
```

### Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_velocity` | 120 units/s | Top speed |
| `max_acceleration` | 21 units/s² | Accel/decel limit |
| `max_turn_rate` | 4 rad/s | Turn rate at max speed |
| `min_turn_rate` | 0.15 rad/s | Turn rate at zero speed |

---

## 4. Formation Coordination

**Location:** `wasm-lib/src/lib.rs`

### Update Flow (per tick)

```
┌──────────────────────────────────────────────────────────┐
│ 1. update_formation()                                    │
│    ├─ Get leader position, heading, velocity             │
│    └─ Broadcast to all followers via                     │
│       update_formation_reference(center, heading, vel)   │
│                                                          │
│ 2. For each drone: state_update(dt, swarm_info)          │
│    ├─ Leader: NLGL path planning to waypoints            │
│    └─ Follower: Proportional navigation to slot          │
│                                                          │
│ 3. If collisions detected:                               │
│    ├─ Remove destroyed drones                            │
│    └─ check_leader_succession()                          │
└──────────────────────────────────────────────────────────┘
```

### Leader Succession

When leader is destroyed:
1. Find lowest ID among remaining drones
2. Promote to leader (clear formation slot, set `is_formation_leader = true`)
3. Reassign formation slots to remaining followers

```rust
fn check_leader_succession(&mut self) {
    if !leader_alive {
        let new_leader = drones.iter().map(|d| d.id).min();
        self.set_formation(formation_type, new_leader);
    }
}
```

---

## 5. Formation Types

```rust
pub enum FormationType {
    Line { spacing: f32 },           // Single file
    Vee { spacing: f32, angle: f32 }, // V formation (default)
    Circle { radius: f32 },          // Circular orbit
    Grid { spacing: f32, cols: usize }, // Grid pattern
}
```

### Default Formation
```rust
// Enabled on swarm creation
Vee { spacing: 60.0, angle: π/4 }
```

---

## 6. Collision Avoidance

**Leader:** Does NOT avoid other swarm drones (followers avoid it)

**Followers:** Use velocity obstacle avoidance, but skip the leader

```rust
// In VelocityObstacleAvoid::tick()
if ctx.is_formation_leader {
    return BehaviorStatus::Success;  // Skip avoidance
}
```

---

## Summary Diagram

```
                     ┌─────────────────────┐
                     │   Waypoint Queue    │
                     └──────────┬──────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────┐
│                      LEADER DRONE                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │ Hermite     │───▶│ NLGL         │───▶│ Platform    │  │
│  │ Spline      │    │ Lookahead    │    │ Control     │  │
│  └─────────────┘    └──────────────┘    └─────────────┘  │
└───────────────────────────┬───────────────────────────────┘
                            │ broadcasts position,
                            │ heading, velocity
                            ▼
┌───────────────────────────────────────────────────────────┐
│                    FOLLOWER DRONES                        │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │ Formation   │───▶│ Proportional │───▶│ Platform    │  │
│  │ Slot Calc   │    │ Navigation   │    │ Control     │  │
│  └─────────────┘    └──────────────┘    └─────────────┘  │
│                                                           │
│  desired_vel = leader_vel + K × (slot_pos - my_pos)       │
└───────────────────────────────────────────────────────────┘
```

---

## 7. Comparison to Industry Standards

### Our Implementation Assessment

| Component | Our Approach | Industry Standard? |
|-----------|--------------|-------------------|
| Formation Method | Leader-Follower | ✅ Most common, widely used |
| Leader Path Planning | NLGL + Hermite Splines | ✅ Standard for fixed-wing UAVs |
| Follower Control | Proportional Navigation | ⚠️ Used, but simpler than some alternatives |
| Leader Succession | Lowest-ID election | ✅ Common approach |
| Collision Avoidance | Velocity Obstacles | ✅ Industry standard |

### Industry Formation Control Methods

Based on recent research (2024-2025), there are **5 main approaches** to UAV formation control:

#### 1. Leader-Follower (What We Use) ✅
> "The leader–follower method is one of the most common and basic methods in UAV formation control."

**Pros:** Simple, predictable, easy to implement, good for hierarchical tasks
**Cons:** Single point of failure, poor robustness if leader fails

**Our mitigation:** Leader succession via lowest-ID election

#### 2. Virtual Structure
> "The UAV formation is viewed as a virtual rigid structure, avoiding the problem of complete paralysis when the leader is incapacitated."

**Pros:** Robust to individual failures, good for rigid formations
**Cons:** Difficult to dynamically adjust, requires high-quality communication

**Consider if:** Formation shape must be strictly maintained (surveillance, mapping)

#### 3. Consensus-Based
> "The consensus method can overcome the shortcomings of the poor robustness of the leader–follower system... and is a predominant control strategy at present."

**Pros:** Decentralized, robust, highly scalable, integrates with other methods
**Cons:** More complex to implement, requires graph theory

**Consider if:** Large swarms (10+ drones), need high fault tolerance

#### 4. Artificial Potential Field (APF)
> "APF has the advantages of simple calculation and strong real-time performance... provides a balance between maintaining formation integrity and ensuring safety."

**Pros:** Natural obstacle avoidance, real-time performance, intuitive
**Cons:** Local minima traps, can cause oscillations

**Consider for:** Dynamic obstacle-rich environments

#### 5. Behavior-Based (Flocking/Boids)
> "The behavioral approach simulates biological response behavior and has good flexibility and robustness."

**Pros:** Emergent behavior, flexible, no central control
**Cons:** Hard to analyze mathematically, imprecise formation keeping

**Consider for:** Entertainment, art installations, loose formations

---

### Alternative Follower Control Methods to Consider

#### 1. Model Predictive Control (MPC)
> "Distributed MPC formulates and solves optimization problems by predicting future states."

```
Advantages over Proportional Navigation:
├─ Handles constraints explicitly (obstacles, velocity limits)
├─ Optimal trajectory over prediction horizon
└─ Better for aggressive maneuvers

Disadvantages:
├─ Computationally expensive
├─ Requires accurate system model
└─ May need powerful onboard computer
```

**Industry usage:** ETH Zurich's agile drones, research platforms

#### 2. Backstepping Sliding Mode Control
> "Solves problems of backstepping error and poor dynamic tracking in traditional PID."

**When to use:** High-precision formation keeping, aggressive maneuvers

#### 3. Deep Reinforcement Learning (DRL)
> "Deep RL has emerged as a core technique for autonomous decision-making in multi-agent systems."

**Approaches:**
- Deep Q-Network (DQN) for formation control
- Proximal Policy Optimization (PPO) for target tracking
- TD3 (Twin Delayed DDPG) for guidance law learning

**When to use:** Highly dynamic environments, adaptive behavior needed

---

### Recommended Enhancements

Based on industry research, consider these improvements:

#### Near-term (Low effort, high impact)

1. **Hybrid APF + Leader-Follower**
   > "The attractive field is replaced by the leader's attraction to followers, overcoming unreachable targets in APF."

   - Add repulsive potential fields between followers
   - Improves collision avoidance within formation

2. **Consensus Layer**
   - Add neighbor-to-neighbor communication
   - Each drone adjusts based on neighbors, not just leader
   - Improves robustness if leader signal is delayed

#### Medium-term (Moderate effort)

3. **Distributed Leader Election**
   > "Voting-based leader election inspired by Raft consensus."

   - Capability-aware leader selection (not just lowest ID)
   - Consider: battery level, sensor health, position

4. **Virtual Structure Hybrid**
   - Maintain virtual structure geometry
   - Leader navigates the structure, not individual drones
   - Better formation rigidity

#### Long-term (High effort, cutting-edge)

5. **MPC for Followers**
   - Replace proportional navigation with MPC
   - Explicit constraint handling
   - Optimal trajectories

6. **Learning-Based Adaptation**
   - Train RL policies for formation keeping
   - Adapt to wind, sensor noise, varying dynamics

---

## 8. References

### Research Papers
- [Advancement Challenges in UAV Swarm Formation Control](https://www.mdpi.com/2504-446X/8/7/320) - MDPI Drones, 2024
- [Leader–follower UAVs formation control based on Deep Q-Network](https://www.nature.com/articles/s41598-024-54531-w) - Nature Scientific Reports, 2024
- [Formation Control Algorithms for Drone Swarms](https://www.researchgate.net/publication/382403635) - ResearchGate, 2024
- [UAV swarms: research, challenges, and future directions](https://jeas.springeropen.com/articles/10.1186/s44147-025-00582-3) - Springer, 2025
- [Distributed MPC for UAVs and Vehicle Platoons](https://www.oaepublish.com/articles/ir.2024.19) - OAE Publishing, 2024

### Control Methods
- [Proportional Navigation for UAV Collision Avoidance](https://www.researchgate.net/publication/225859738) - ResearchGate
- [APF + Leader-Follower Integration](https://www.mdpi.com/2227-7390/12/7/954) - MDPI Mathematics, 2024
- [MPC Survey for Micro Aerial Vehicles](https://arxiv.org/pdf/2011.11104) - arXiv
- [Leader-Follower Formation via Feature Modelling](https://www.tandfonline.com/doi/full/10.1080/21642583.2023.2268153) - Taylor & Francis, 2023

### Open Source
- [UZH High MPC for Agile Drones](https://github.com/uzh-rpg/high_mpc) - ETH Zurich
