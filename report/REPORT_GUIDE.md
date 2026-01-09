# Report Writing Guide: Comparative Analysis of DRL Algorithms for BipedalWalker

## Overview

This guide provides detailed instructions for writing a 10-page academic report comparing TD3, SAC, and PPO algorithms on the BipedalWalker environment. **Focus is on SAC** (your section), with references to the other two algorithms for comparative analysis.

---

## Suggested Report Structure (10 pages)

| Section | Pages | Focus |
|---------|-------|-------|
| Abstract | 0.25 | Summary of findings |
| Introduction | 0.75 | Problem motivation, contributions |
| Problem Formulation | 1.0 | Environment, state/action spaces, variants |
| Algorithms | 1.5 | TD3, SAC, PPO theory |
| **Implementation Details** | 2.0 | Custom env, wrappers, optimizations |
| Evaluation Methodology | 0.75 | Metrics, training protocol |
| **Results & Analysis** | 2.5 | Training curves, comparison tables, insights |
| Discussion | 0.75 | RL landscape context |
| Conclusions | 0.5 | Key findings, recommendations |

---

## Section-by-Section Guide

### 1. Abstract (Current: needs conclusions)

**What to include:**
- Problem: Bipedal locomotion as continuous control benchmark
- Methods: Three algorithms (TD3, SAC, PPO) across three difficulty levels
- Key findings: SAC's balance of exploration/exploitation, natural walking quality
- Main contribution: Comprehensive comparison with reward engineering analysis

---

### 2. Introduction

**Current state:** Good foundation, expand on:

**Add:**
- Why BipedalWalker is a canonical benchmark (cite OpenAI Gym paper)
- Practical relevance to robotics (sim-to-real transfer potential)
- Gap in literature: Limited comparative studies across difficulty variants

**References to add:**
```bibtex
@article{towers2023gymnasium,
  title={Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author={Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan and others},
  journal={arXiv preprint arXiv:2407.17032},
  year={2024}
}

@article{raffin2021stable,
  title={Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  author={Raffin, Antonin and Hill, Ashley and others},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={268},
  pages={1--8},
  year={2021}
}
```

---

### 3. Problem Formulation

#### 3.1 Standard BipedalWalker-v3

**Codebase reference:** [src/envs/custom_walker.py](../src/envs/custom_walker.py) lines 104-175

**State Space (24D):**
```
obs[0]: hull_angle           # Hull orientation (radians)
obs[1]: hull_angular_vel     # Hull angular velocity
obs[2]: vel_x                # Horizontal velocity
obs[3]: vel_y                # Vertical velocity
obs[4-5]: joint1_angle, joint1_speed  # Hip joint leg 1
obs[6-7]: joint2_angle, joint2_speed  # Knee joint leg 1
obs[8]: leg1_ground_contact  # Binary contact flag
obs[9-10]: joint3_angle, joint3_speed # Hip joint leg 2
obs[11-12]: joint4_angle, joint4_speed # Knee joint leg 2
obs[13]: leg2_ground_contact # Binary contact flag
obs[14-23]: lidar[0-9]       # 10 rangefinder measurements
```

**Action Space (4D):** Motor torques ∈ [-1, 1] for:
- Hip 1, Knee 1, Hip 2, Knee 2

**Base Reward Function:**
```
r_t = 130 × x_progress - 5|θ_hull| - 0.00035 Σ|a_i| - 100 × 𝟙_fallen
```

**Codebase reference:** [src/envs/custom_walker.py](../src/envs/custom_walker.py) lines 700-750 (step function)

#### 3.2 Environment Variants

| Variant | Obstacles | Difficulty | Steps | Codebase |
|---------|-----------|------------|-------|----------|
| Easy | Flat + minor bumps | Low | 1600 | `hardcore=False` |
| Hardcore | Stumps, stairs, pits | High | 2000 | `hardcore=True` |
| Bridges | All + dynamic bridges | Extreme | 2000 | Custom walker + wrapper |

**Bridge Mechanism (Novel):**
- Bridges start vertical, lower after ~3 seconds
- Agent must learn to wait, then cross
- **Codebase:** [src/envs/custom_walker.py](../src/envs/custom_walker.py) lines 400-460

---

### 4. Algorithms

#### 4.1 SAC (Your Focus)

**Theory to explain:**
1. **Maximum Entropy RL Framework:**
   ```
   J(π) = Σ_t E[(r(s_t, a_t) + α H(π(·|s_t)))]
   ```
   
2. **Twin Critics (like TD3):** Prevent overestimation
   ```
   Q_target = r + γ(min(Q_1, Q_2)(s', a') - α log π(a'|s'))
   ```

3. **Automatic Entropy Tuning:**
   ```
   α* = argmin_α E[-α log π(a|s) - α H̄]
   ```
   Where H̄ = -dim(A) is target entropy

**Codebase reference:** [src/agents/sac_agent.py](../src/agents/sac_agent.py)
- Lines 21-82: Initialization with twin critics
- Lines 84-100: Action selection with reparameterization
- Lines 102-180: SAC update with entropy tuning

**Why SAC for BipedalWalker:**
- Stochastic policy → better exploration of action space
- Entropy regularization → avoids premature convergence to suboptimal gaits
- Off-policy → sample efficient for expensive locomotion episodes

**References:**
```bibtex
@inproceedings{haarnoja2018soft_algorithms,
  title={Soft Actor-Critic Algorithms and Applications},
  author={Haarnoja, Tuomas and Zhou, Aurick and others},
  booktitle={arXiv preprint arXiv:1812.05905},
  year={2018}
}
```

#### 4.2 TD3 (Brief for comparison)

**Codebase:** [src/agents/td3_agent.py](../src/agents/td3_agent.py)

Key mechanisms:
1. Twin critics: `min(Q_1, Q_2)` for target
2. Delayed policy updates: Actor updated every 2 critic steps
3. Target policy smoothing: Add clipped noise to target actions

#### 4.3 PPO (Brief for comparison)

**Codebase:** [src/agents/ppo_agent.py](../src/agents/ppo_agent.py)

Key mechanisms:
1. Clipped surrogate objective
2. GAE for advantage estimation
3. On-policy (less sample efficient)

---

### 5. Implementation Details (CRITICAL FOR SAC)

This is where your report should shine. Discuss the engineering that made SAC work.

#### 5.1 RL Zoo3 Optimizations

**Codebase:** [configs/sac_hardcore_gpu.yaml](../configs/sac_hardcore_gpu.yaml), [configs/sac_bridges_gpu.yaml](../configs/sac_bridges_gpu.yaml)

**Key hyperparameters from RL Zoo3 tuning:**

| Parameter | Value | Why It Matters |
|-----------|-------|----------------|
| `learning_rate` | 7.3e-4 | RL Zoo3 tuned for BipedalWalker |
| `gamma` | 0.98 | Lower than default (0.99) for natural walking |
| `tau` | 0.01 | Faster target updates than default (0.005) |
| `gradient_steps` | 4 | Multiple updates per env step |
| `net_arch` | [400, 300] | Wider first layer for perception |
| `buffer_size` | 1-2M | Large buffer for off-policy learning |

**Reference:**
```bibtex
@misc{rl-zoo3,
  author={Raffin, Antonin},
  title={RL Baselines3 Zoo},
  year={2020},
  howpublished={\url{https://github.com/DLR-RM/rl-baselines3-zoo}}
}
```

#### 5.2 Reward Engineering for Natural Walking

**Codebase:** [wrappers/bridge_balanced_wrapper.py](../wrappers/bridge_balanced_wrapper.py)

**Problem:** Default reward only optimizes forward progress → unnatural gaits

**Solution:** Custom reward shaping wrapper

```python
# Modified reward function
r_shaped = r_base - λ_smooth × |a_t - a_{t-1}|    # Smoothness penalty
                 - λ_hull × θ_hull²               # Hull angle penalty
                 - λ_angular × ω_hull²            # Angular velocity penalty
                 - λ_velocity × max(0, |v_joint| - v_max)  # Joint velocity limit
                 + λ_knee × 𝟙_{knee_bent_during_swing}     # Knee bend reward
```

**Parameters (from your successful natural walking config):**

| Component | Coefficient | Purpose |
|-----------|-------------|---------|
| `smoothness_coef` | 0.05 | Penalize jerky action changes |
| `hull_angle_coef` | 0.03 | Keep body upright |
| `hull_angular_vel_coef` | 0.015 | Prevent spinning |
| `max_joint_velocity` | 2.0 | Limit leg speed |
| `velocity_penalty` | 0.02 | Penalty for excess velocity |
| `knee_bend_reward` | 0.02 | Encourage natural knee lift |


#### 5.3 Bridge-Specific Reward Shaping

**Challenge:** Agent must learn non-obvious behavior (wait, then cross)

**Codebase:** [wrappers/bridge_balanced_wrapper.py](../wrappers/bridge_balanced_wrapper.py) lines 102-220

**Detection mechanism:**
```python
def _detect_bridge_in_lidar(self, obs):
    front_lidar = obs[14:19]  # 5 front beams
    close_beams = sum(1 for d in front_lidar if d < threshold)
    # Bridge = 3+ blocked beams after minimum progress
    return close_beams >= 3 and has_progress
```

**Bridge reward components:**
1. **Waiting bonus:** +0.02/step × 300 steps ≈ +6.0 total
2. **Crossing bonus:** +8.0 for successful crossing
3. **Total bridge reward:** ~14 (equivalent to 14 terrain sections)

**Design insight:** Rewards must be balanced - too high causes reward hacking, too low is ignored.

#### 5.4 VecNormalize and Environment Wrappers

**Codebase:** [training/train_sac.py](../training/train_sac.py) lines 166-180

```python
env = VecNormalize(
    env,
    norm_obs=True,   # Normalize observations
    norm_reward=True, # Normalize rewards (training only!)
    clip_obs=10.0,
    clip_reward=10.0,
)
```

**Critical:** Disable reward normalization during evaluation!

#### 5.5 Parallel Environment Training

**Codebase:** [training/train_sac.py](../training/train_sac.py) lines 158-165

```python
env = SubprocVecEnv([make_env(i, seed, config) for i in range(n_envs)])
```

- **8 parallel environments** for sample efficiency
- Frame skip = 4 for temporal abstraction

---

### 6. Evaluation Methodology

#### 6.1 Training Protocol

**Codebase:** [training/train_sac.py](../training/train_sac.py)

| Setting | Value |
|---------|-------|
| Total timesteps | 10M (hardcore/bridges) |
| Evaluation frequency | 25,000 steps |
| Evaluation episodes | 10 |
| Checkpoint frequency | 100,000 steps |
| Early stopping | 15 evals without improvement |

#### 6.2 Metrics

**Performance metrics:**
- Mean episode reward (± std)
- Success rate (reward ≥ 300)
- Episode length (longer = more robust)

**Learning metrics:**
- Convergence speed (steps to reach threshold)
- Stability (variance in learning curve)
- Sample efficiency (reward per million steps)

**Action quality (for natural walking):**
- Action smoothness: `mean(|a_t - a_{t-1}|)`
- Action entropy: Distribution spread

**Codebase:** [scripts/analyze_model.py](../scripts/analyze_model.py)

---

### 7. Results & Analysis (YOUR MAIN CONTRIBUTION)

#### 7.1 How to Generate Training Curves

**TensorBoard logs location:** `experiments/logs/`
- `sac_easy/` - Easy mode
- `sac_hardcore/` - Hardcore mode  
- `sac_bridges/` - Bridges mode

**Commands to visualize:**
```bash
tensorboard --logdir experiments/logs/
```

**Key plots to include:**
1. Episode reward over timesteps (all 3 variants)
2. Evaluation mean reward (smoothed)
3. Episode length evolution
4. Entropy coefficient (SAC-specific)

#### 7.2 Comparison Tables Template

**Table: Final Performance (after convergence)**

| Metric | Easy | Hardcore | Bridges |
|--------|------|----------|---------|
| Mean Reward | ±std | ±std | ±std |
| Success Rate | % | % | % |
| Mean Episode Length | steps | steps | steps |
| Convergence (steps) | K | K | K |

**Table: Algorithm Comparison (Hardcore)**

| Metric | SAC | TD3 | PPO |
|--------|-----|-----|-----|
| Mean Reward | | | |
| Sample Efficiency | | | |
| Training Stability | | | |
| Natural Gait Quality | | | |

#### 7.3 Analysis Points for SAC

**Strengths to highlight:**
1. **Exploration via entropy:** Prevents collapse to single gait
2. **Off-policy efficiency:** Replay buffer enables learning from past experience
3. **Automatic temperature tuning:** Adapts exploration throughout training

**Weaknesses to discuss:**
1. Slower initial convergence than TD3 (exploration overhead)
2. More hyperparameter-sensitive than TD3
3. Memory overhead from twin critics + replay buffer

**Bridge-specific insights:**
- SAC's stochastic policy helps discover waiting behavior
- Entropy bonus maintains exploration when stuck at bridge
- Compare bridge crossing success rate vs. TD3/PPO

#### 7.4 Natural Walking Quality Analysis

**Codebase for analysis:** [scripts/analyze_actions.py](../scripts/analyze_actions.py)

**Metrics to compute:**
1. Action autocorrelation (periodic = natural)
2. Action magnitude distribution (should be centered, not saturated)
3. Hull angle variance during episode (lower = more stable)

**Plot suggestions:**
- Action distribution histograms (compare agents)
- Hull angle over episode time
- Leg joint trajectories (phase plot)

---

### 8. Discussion

#### 8.1 RL Landscape Context

**Position your findings within:**

1. **Off-policy vs On-policy trade-off:**
   - SAC/TD3: Better sample efficiency
   - PPO: More stable but data-hungry

2. **Entropy regularization importance:**
   - Critical for exploration in continuous action spaces
   - SAC's automatic tuning > manual entropy coefficients

3. **Reward shaping debate:**
   - Pure RL purists: Avoid shaped rewards
   - Practical view: Necessary for complex behaviors (bridges)
   - Your evidence: Natural walking required explicit shaping

**References:**
```bibtex
@article{ng1999policy,
  title={Policy Invariance Under Reward Transformations},
  author={Ng, Andrew Y and others},
  journal={ICML},
  year={1999}
}

@article{andrychowicz2020matters,
  title={What Matters In On-Policy Reinforcement Learning?},
  author={Andrychowicz, Marcin and others},
  journal={arXiv preprint arXiv:2006.05990},
  year={2020}
}
```

#### 8.2 Practical Recommendations

Based on your results:

| Scenario | Recommended Algorithm | Why |
|----------|----------------------|-----|
| Quick prototyping | PPO | Simple, stable |
| Sample efficiency matters | TD3 | Fastest convergence |
| Exploration critical | SAC | Entropy regularization |
| Natural behaviors needed | SAC + reward shaping | Stochastic + shaped |

---

### 9. Conclusions

**Template for conclusions:**

1. **Main finding:** SAC achieves competitive performance with superior exploration characteristics
2. **Reward engineering:** Critical for natural walking - smoothness penalties and knee rewards essential
3. **Bridge challenge:** Demonstrates SAC's ability to discover non-obvious strategies
4. **Practical insight:** RL Zoo3 hyperparameters provide strong baselines

---

## Literature References (Complete List)

```bibtex
% Core algorithms
@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1587--1596},
  year={2018}
}

@inproceedings{haarnoja2018soft,
  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  pages={1861--1870},
  year={2018}
}

@article{schulman2017proximal,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}

% Environments and tools
@article{brockman2016openai,
  title={OpenAI Gym},
  author={Brockman, Greg and Cheung, Vicki and others},
  journal={arXiv preprint arXiv:1606.01540},
  year={2016}
}

@article{raffin2021stable,
  title={Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  author={Raffin, Antonin and Hill, Ashley and others},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={268},
  year={2021}
}

% Reward shaping and locomotion
@article{ng1999policy,
  title={Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping},
  author={Ng, Andrew Y and Harada, Daishi and Russell, Stuart},
  journal={ICML},
  year={1999}
}

@article{heess2017emergence,
  title={Emergence of Locomotion Behaviours in Rich Environments},
  author={Heess, Nicolas and Sriram, Srinivasan and others},
  journal={arXiv preprint arXiv:1707.02286},
  year={2017}
}

% Hyperparameter tuning
@misc{rl-zoo3,
  author={Raffin, Antonin},
  title={RL Baselines3 Zoo},
  year={2020},
  howpublished={\url{https://github.com/DLR-RM/rl-baselines3-zoo}}
}
```

---

## Appendix: Useful Commands

### Visualize trained agents
```bash
# SAC Easy
python evaluation/visualize.py --checkpoint experiments/checkpoints/sac_easy/best_model.zip --mode easy

# SAC Hardcore
python evaluation/visualize.py --checkpoint experiments/checkpoints/sac_hardcore/best_model.zip --mode hardcore

# SAC Bridges
python evaluation/visualize.py --checkpoint experiments/checkpoints/sac_bridges/best_model.zip --mode hardcore
```

### Generate evaluation metrics
```bash
python scripts/analyze_model.py --checkpoint experiments/checkpoints/sac_hardcore/best_model.zip --config configs/sac_hardcore_gpu.yaml
```

### Record videos
```bash
python evaluation/record_video.py --checkpoint experiments/checkpoints/sac_bridges/best_model.zip --episodes 3
```

### Compare training logs
```bash
python scripts/plot_results.py --logs experiments/logs/sac_easy experiments/logs/sac_hardcore
```

---

## Checklist Before Submission

- [ ] Abstract updated with actual results
- [ ] All hyperparameter tables filled with actual values
- [ ] Training curves included for all 3 SAC variants
- [ ] Comparison tables with TD3/PPO data from colleagues
- [ ] Natural walking quality analysis completed
- [ ] Bridge crossing statistics included
- [ ] All code references verified
- [ ] Bibliography complete and formatted
- [ ] Page count ≤ 10

---

## File Cross-Reference Quick Guide

| Topic | Primary File | Lines |
|-------|--------------|-------|
| SAC Algorithm | `src/agents/sac_agent.py` | All |
| Custom Environment | `src/envs/custom_walker.py` | 104-750 |
| Bridge Mechanism | `src/envs/custom_walker.py` | 400-460 |
| Reward Wrapper | `wrappers/bridge_balanced_wrapper.py` | All |
| SAC Training | `training/train_sac.py` | All |
| SAC Configs | `configs/sac_*_gpu.yaml` | All |
| Evaluation | `evaluation/evaluate.py` | All |
| Analysis Scripts | `scripts/analyze_model.py` | All |

Good luck with your report! 🎓
