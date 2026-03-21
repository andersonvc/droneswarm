# RL Network Architecture

## Policy Network (PolicyNetV2)

Per-drone actor-critic with entity-attention. Each drone observes its own ego state and a variable number of entity tokens (other drones, targets), processes them through shared encoders and attention, then selects an action.

```mermaid
graph TD
    subgraph Inputs
        EGO["Ego Features<br/><i>25-dim</i><br/>position, velocity, heading,<br/>task type, global state"]
        ENT["Entity Tokens<br/><i>N × 10-dim</i><br/>dx, dy, dist, vx, vy, heading,<br/>type, alive, assignments, target"]
    end

    subgraph Encoders
        EGO_ENC["Ego Encoder<br/><i>Linear 25→64, ReLU</i>"]
        ENT_ENC["Entity Encoder<br/><i>Linear 10→64, ReLU</i><br/>applied per entity"]
    end

    subgraph Entity Attention
        ATTN1["Multi-Head Attention Layer 1<br/><i>4 heads, head_dim=16</i><br/>+ Residual + LayerNorm"]
        ATTN2["Multi-Head Attention Layer 2<br/><i>4 heads, head_dim=16</i><br/>+ Residual + LayerNorm"]
        POOL["Max Pool<br/><i>N × 64 → 64</i>"]
    end

    subgraph Trunk
        CONCAT["Concat<br/><i>64 + 64 = 128</i>"]
        FC1["FC1<br/><i>Linear 128→256, ReLU</i>"]
        FC2["FC2<br/><i>Linear 256→256, ReLU</i>"]
    end

    subgraph Heads
        ACTOR["Actor Head<br/><i>Linear 256→13</i><br/>+ Action Mask + Softmax"]
        VALUE["Local Value Head<br/><i>Linear 256→1</i>"]
    end

    EGO --> EGO_ENC
    ENT --> ENT_ENC
    EGO_ENC --> CONCAT
    ENT_ENC --> ATTN1 --> ATTN2 --> POOL --> CONCAT
    CONCAT --> FC1 --> FC2
    FC2 --> ACTOR
    FC2 --> VALUE

    style EGO fill:#e8f4f8,stroke:#2980b9
    style ENT fill:#e8f4f8,stroke:#2980b9
    style ATTN1 fill:#fef9e7,stroke:#f39c12
    style ATTN2 fill:#fef9e7,stroke:#f39c12
    style POOL fill:#fef9e7,stroke:#f39c12
    style ACTOR fill:#eafaf1,stroke:#27ae60
    style VALUE fill:#eafaf1,stroke:#27ae60
```

## Attention Block Detail

Each attention layer contains self-attention over entity embeddings with a residual connection and post-residual layer normalization.

```mermaid
graph TD
    IN["Entity Embeddings<br/><i>N × 64</i>"] --> WQ & WK & WV

    subgraph Projections
        WQ["W_Q<br/><i>64→64</i>"]
        WK["W_K<br/><i>64→64</i>"]
        WV["W_V<br/><i>64→64</i>"]
    end

    WQ --> SCORES
    WK --> SCORES["Q @ K^T / √d<br/><i>N × N per head</i>"]
    SCORES --> SOFTMAX["Softmax<br/><i>with NaN guard</i>"]
    SOFTMAX --> AV["Attn @ V<br/><i>N × 16 per head</i>"]
    WV --> AV
    AV --> WO["W_O<br/><i>64→64</i>"]
    WO --> ADD["+ Residual"]
    IN --> ADD
    ADD --> LN["LayerNorm<br/><i>learned γ, β</i>"]
    LN --> OUT["Output<br/><i>N × 64</i>"]

    style SCORES fill:#fef9e7,stroke:#f39c12
    style SOFTMAX fill:#fef9e7,stroke:#f39c12
    style ADD fill:#fadbd8,stroke:#e74c3c
    style LN fill:#fadbd8,stroke:#e74c3c
```

## MAPPO Centralized Critic

Sees each drone's trunk representation plus the mean of all drones' representations in the environment. Produces a centralized value estimate used for GAE advantage computation.

```mermaid
graph LR
    H2["Per-Drone h2<br/><i>256-dim</i>"]
    MP["Mean Pool<br/>all drones' h2<br/><i>256-dim</i>"]
    H2 --> CAT["Concat<br/><i>512-dim</i>"]
    MP --> CAT
    CAT --> C1["FC1<br/><i>512→256, ReLU</i>"]
    C1 --> C2["FC2<br/><i>256→256, ReLU</i>"]
    C2 --> CV["Value Head<br/><i>256→1</i>"]
    CV --> VOUT["V_centralized"]

    style H2 fill:#e8f4f8,stroke:#2980b9
    style MP fill:#e8f4f8,stroke:#2980b9
    style VOUT fill:#eafaf1,stroke:#27ae60
```

## QMIX Value Decomposition

Combines per-drone local values into a team value using hypernetwork-generated monotonic mixing weights, conditioned on global state.

```mermaid
graph TD
    subgraph Per-Drone
        LV["Local Values<br/><i>N scalars</i>"]
    end

    subgraph Global State
        GS["h2 Mean Pool + Game Features<br/><i>256 + 8 = 264-dim</i>"]
    end

    subgraph Hypernetworks
        HW1["Hyper W1<br/><i>264 → N×32</i>"]
        HB1["Hyper B1<br/><i>264 → 32</i>"]
        HW2["Hyper W2<br/><i>264 → 32</i>"]
        HB2["Hyper B2<br/><i>264 → 1</i>"]
    end

    GS --> HW1 & HB1 & HW2 & HB2
    LV --> MIX1["|W1| @ local + b1<br/>ELU"]
    HW1 --> MIX1
    HB1 --> MIX1
    MIX1 --> MIX2["|W2| · hidden + b2"]
    HW2 --> MIX2
    HB2 --> MIX2
    MIX2 --> TEAM["V_team"]

    style LV fill:#e8f4f8,stroke:#2980b9
    style GS fill:#e8f4f8,stroke:#2980b9
    style TEAM fill:#eafaf1,stroke:#27ae60
    style HW1 fill:#f5eef8,stroke:#8e44ad
    style HB1 fill:#f5eef8,stroke:#8e44ad
    style HW2 fill:#f5eef8,stroke:#8e44ad
    style HB2 fill:#f5eef8,stroke:#8e44ad
```

## Training Loop

```mermaid
graph TD
    RESET["Reset Envs<br/><i>curriculum stage</i>"] --> ROLLOUT

    subgraph ROLLOUT["Rollout Collection (1024 envs × 256 steps)"]
        OBS["Observe<br/><i>ego + entities per drone</i>"]
        ACT["Policy Forward<br/><i>batched with action masking</i>"]
        STEP["Env Step<br/><i>parallel via rayon</i>"]
        STORE["Store Transitions<br/><i>obs, action, reward, value, done</i>"]
        OBS --> ACT --> STEP --> STORE
        STORE -.-> OBS
    end

    ROLLOUT --> GAE["Per-Drone GAE<br/><i>centralized critic baseline</i><br/><i>death + truncation bootstrap</i>"]
    GAE --> NORM["Normalize Advantages"]

    NORM --> CRITIC_UPDATE

    subgraph CRITIC_UPDATE["Critic + QMIX Update (before PPO)"]
        CU["Centralized Critic<br/><i>batched forward + backward</i><br/><i>team-only returns</i>"]
        QU["QMIX Mixer<br/><i>per-timestep groups</i><br/><i>team-only returns</i>"]
    end

    CRITIC_UPDATE --> PPO

    subgraph PPO["PPO Update (4 epochs)"]
        SHUFFLE["Shuffle + Mini-Batch<br/><i>8192 samples</i>"]
        BACKWARD["Parallel Backward<br/><i>policy gradient + entropy</i>"]
        OPTIM["Adam Step<br/><i>gradient clipping</i>"]
        SHUFFLE --> BACKWARD --> OPTIM
    end

    PPO --> EVAL{"Evaluate?"}
    EVAL -->|yes| DOCTRINE_EVAL["Eval vs Doctrine<br/><i>50 episodes</i>"]
    DOCTRINE_EVAL --> CURRICULUM["Curriculum Check<br/><i>advance at 70% / demote at 30%</i>"]
    EVAL -->|no| RESET
    CURRICULUM --> RESET

    style GAE fill:#fef9e7,stroke:#f39c12
    style PPO fill:#eafaf1,stroke:#27ae60
    style CRITIC_UPDATE fill:#fadbd8,stroke:#e74c3c
```
