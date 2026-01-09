# Custom BipedalWalker with BRIDGES

This directory contains a custom BipedalWalker environment that adds a new obstacle type: **BRIDGES** (dynamic drawbridges).

---

## What's Different?

### Standard BipedalWalker Hardcore
Obstacles: `GRASS`, `STUMP`, `STAIRS`, `PIT`

### Custom BipedalWalker Hardcore
Obstacles: `GRASS`, `STUMP`, `STAIRS`, `PIT`, **`BRIDGE`**

---

## Bridge Obstacle Details

### Physics Implementation

**File**: `custom_walker.py` (lines 396-465, 704-731)

**Structure**:
- **Gap**: 7 terrain steps wide (~3.27 units)
- **Bridge body**: Dynamic rigid body (Box2D physics)
- **Hinge**: Revolute joint at left edge
- **Motor**: Controlled lowering mechanism
- **Initial state**: Vertical/raised (blocking path)

**Activation Logic**:
```python
# When robot approaches within 10 units
if (bridge_x - robot_x) < 10.0:
    bridge['active'] = True

# Timing sequence
if timer < 300 steps (6 seconds):
    motorSpeed = 2.0  # Stay raised
else:
    motorSpeed = -2.0  # Lower bridge
```

**Challenge**:
- Robot must **approach bridge and wait** for it to lower
- Bridge takes **6 seconds** (300 steps) before lowering
- Must maintain balance while waiting
- Bridge lowers slowly - requires timing to cross safely

### Why This Is Harder

1. **Patience required**: Can't just run through like other obstacles
2. **Timing**: Must cross at the right moment (too early = fall, too late = waste time)
3. **Balance during wait**: Standing still for 6 seconds is unstable
4. **New physics**: Dynamic bridge body can swing/move when stepped on

---

## Training on Custom Walker

### Quick Start

```bash
# Train with Elite Hardcore configuration
python train_custom_walker.py --config configs/sac_elite_hardcore_gpu.yaml
```

This will:
- Use **custom_walker.py** instead of standard BipedalWalker-v3
- Apply **EliteHardcoreWrapper** (proven reward shaping)
- Train on all 5 obstacle types including **BRIDGES**
- Save to: `experiments/checkpoints/sac_elite_unified_hardcore_gpu_custom_bridges/`

### Training Time

- **Apple Silicon (M1/M2)**: ~2-3 hours for 10M steps
- **8 parallel environments**: ~2500 steps/second
- **Expected performance**: Similar to standard hardcore (250-300 reward)
- **Bridge success**: Will learn to wait and cross

### Resume Training

```bash
python train_custom_walker.py \
  --config configs/sac_elite_hardcore_gpu.yaml \
  --resume experiments/checkpoints/sac_elite_unified_hardcore_gpu_custom_bridges/sac_model_5000000_steps.zip
```

---

## Evaluating Trained Models

### Live Visualization

```bash
python visualize_custom_walker.py \
  --model experiments/checkpoints/sac_elite_unified_hardcore_gpu_custom_bridges/final_model/sac_model.zip \
  --vecnorm experiments/checkpoints/sac_elite_unified_hardcore_gpu_custom_bridges/final_model/vecnormalize.pkl \
  --episodes 5
```

**What you'll see**:
- Robot walking through standard obstacles (stumps, stairs, pits)
- Robot **approaching bridge** (vertical/raised)
- Robot **waiting** while bridge lowers (6 seconds)
- Bridge **lowering** to horizontal position
- Robot **crossing** the bridge

### Recording Videos

```bash
python visualize_custom_walker.py \
  --model experiments/checkpoints/sac_elite_unified_hardcore_gpu_custom_bridges/final_model/sac_model.zip \
  --vecnorm experiments/checkpoints/sac_elite_unified_hardcore_gpu_custom_bridges/final_model/vecnormalize.pkl \
  --episodes 5 \
  --record
```

Videos saved to: `experiments/videos/custom_bridges_TIMESTAMP-episode-*.mp4`

**Note**: Videos use correct 50 FPS (RecordVideo wraps BEFORE frame skip)

---

## Technical Implementation

### Environment Registration

The training script registers the custom environment:

```python
from gymnasium.envs.registration import register

register(
    id='CustomBipedalWalker-v3',
    entry_point='custom_walker:BipedalWalker',
    max_episode_steps=2000,
    reward_threshold=300,
)
```

Then creates it with:
```python
env = gym.make("CustomBipedalWalker-v3", hardcore=True)
```

### Wrapper Application

Same as standard Elite Hardcore:
```python
env = gym.make("CustomBipedalWalker-v3", hardcore=True)
env = EliteHardcoreWrapper(env,
    frame_skip=4,
    smoothness_coef=0.2,
    hull_angle_coef=0.1,
    hull_angular_vel_coef=0.05,
    knee_bend_reward=0.02,
    max_joint_velocity=2.0,
    velocity_penalty=0.02,
    early_steps_stability_bonus=0.01,
    early_steps_count=100,
)
```

**No changes needed to reward function!** Elite Hardcore wrapper works identically on custom walker.

---

## Expected Learning Behavior

### Early Training (0-2M steps)

- Learns basic walking on flat ground
- Starts navigating stumps and stairs
- **Falls into bridge gaps** (doesn't wait for bridge)
- Low reward (~50-150)

### Mid Training (2M-6M steps)

- Navigates most obstacles successfully
- **Starts learning to slow down before bridges**
- May stand/balance while waiting (unstable)
- Medium reward (~150-250)

### Late Training (6M-10M steps)

- Smooth navigation of all obstacle types
- **Learned strategy for bridges**:
  - Approaches bridge
  - Maintains balance while waiting
  - Crosses after bridge lowers
- High reward (~250-300)

### Potential Strategies Agent May Learn

1. **Slow approach**: Reduce speed when bridge detected ahead (LIDAR)
2. **Balanced waiting**: Small motor adjustments to maintain upright stance
3. **Timed crossing**: Wait for bridge angle to reach ~horizontal before stepping
4. **Emergency stop**: If approaching too fast, stop and wait

---

## Comparison: Standard vs Custom Walker

| Metric | Standard Hardcore | Custom with Bridges | Difference |
|--------|-------------------|---------------------|------------|
| Obstacle Types | 4 (GRASS, STUMP, STAIRS, PIT) | 5 (+ BRIDGE) | +1 new type |
| Episode Difficulty | Hard | **Harder** | Bridges add waiting challenge |
| Expected Reward | 250-300 | 200-280 | Slightly lower (bridges are hard) |
| Training Time | 10M steps | 10-15M steps | May need more training |
| Episode Length | 600-1200 steps | 700-1400 steps | Longer (waiting time) |
| Required Skills | Speed + agility | Speed + agility + **patience** | +Timing control |

---

## Troubleshooting

### Problem: Agent never crosses bridges

**Symptoms**: Falls into bridge gaps every time

**Causes**:
1. Not trained long enough (need >5M steps)
2. LIDAR not detecting bridge ahead
3. Reward function doesn't incentivize waiting

**Solutions**:
- Train longer (try 15M steps)
- Check LIDAR readings in logs
- Verify bridge activation distance (should be 10 units)

### Problem: Agent waits too long

**Symptoms**: Stands before bridge for >10 seconds, episode times out

**Causes**:
1. Over-conservative policy
2. Bridge lowering mechanism broken
3. Agent learned "standing is safe"

**Solutions**:
- Check bridge timer logic (should lower after 300 steps)
- Verify bridge angle reaches horizontal (< 0.02 rad)
- May need to add "timeout penalty" to reward

### Problem: Performance worse than standard

**Symptoms**: Custom walker gets <200 reward, standard gets >280

**Expected**: This is **normal**! Bridges are a new, harder obstacle type.

**What to do**:
- Train longer (15M instead of 10M)
- Check that bridge crossing is learned (watch videos)
- If bridge success rate is high but reward is low, that's OK - bridges add episode length

---

## Files Overview

| File | Purpose |
|------|---------|
| `custom_walker.py` | Custom BipedalWalker environment with bridges |
| `train_custom_walker.py` | Training script for custom walker |
| `visualize_custom_walker.py` | Visualization script for custom walker |
| `elite_hardcore_wrapper.py` | Reward shaping (works on both standard and custom) |
| `configs/sac_elite_hardcore_gpu.yaml` | Configuration (same for both) |

---

## Key Differences from Standard Training

### What's the Same
- SAC algorithm
- Elite Hardcore wrapper
- Hyperparameters
- Network architecture
- Normalization settings

### What's Different
- Environment: `CustomBipedalWalker-v3` instead of `BipedalWalker-v3`
- Obstacles: +BRIDGE type
- Checkpoint name: `_custom_bridges` suffix
- Expected performance: Slightly lower (bridges are hard)

---

## Advanced: Analyzing Bridge Behavior

### Logging Bridge Encounters

Modify `custom_walker.py` line 714 to add logging:

```python
if not bridge['active'] and (bridge_x - robot_x) < 10.0:
    bridge['active'] = True
    print(f"Bridge activated at x={bridge_x:.1f}, robot at x={robot_x:.1f}")
```

### Extracting Bridge Statistics

After training, analyze logs to see:
- How many bridges encountered per episode
- Bridge crossing success rate
- Average wait time before crossing

### Visualizing Bridge Timing

Use tensorboard to track:
- Episode length (should increase with bridges)
- Reward distribution (may be more variable)
- Success rate over training

```bash
tensorboard --logdir experiments/logs/sac_elite_unified_hardcore_gpu_custom_bridges/
```

---

## Future Improvements

### Potential Enhancements

1. **Variable bridge timing**: Random 200-400 step delays
2. **Swinging bridges**: Add oscillation to lowered bridge
3. **Multiple bridges per episode**: Increase frequency
4. **Bridge collapse**: Break after too much weight
5. **Moving bridges**: Horizontal translation during lowering

### Curriculum Learning

Train in stages:
1. Standard hardcore (no bridges) - 5M steps
2. Low bridge frequency (10%) - 3M steps
3. Normal bridge frequency (20%) - 2M steps
4. High bridge frequency (30%) - final training

---

## Credits

- **Base environment**: OpenAI Gym BipedalWalker-v3
- **Bridge implementation**: Custom modification (custom_walker.py)
- **Elite Hardcore wrapper**: Proven reward shaping for obstacle navigation
- **Training framework**: Stable-Baselines3 SAC

---

## Citation

If you use this custom environment in research:

```bibtex
@misc{custom_bipedal_walker_bridges,
  title={Custom BipedalWalker with Dynamic Bridges},
  author={TASI Project},
  year={2025},
  note={Extension of Gymnasium BipedalWalker-v3 with dynamic drawbridge obstacles}
}
```

---

**Happy training on bridges!** 🌉🤖
