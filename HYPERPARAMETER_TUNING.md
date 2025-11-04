# Hyperparameter Tuning Guide for Bipedal Walker

## Understanding Why Your Model Failed

Based on your results (mean reward: -50), the most likely issues are:

1. **Insufficient Exploration** - Agent didn't try enough diverse actions
2. **Learning Rate Issues** - Updates were too aggressive or too conservative
3. **Training Time** - May need more timesteps to learn
4. **Policy Collapse** - Agent learned a suboptimal strategy and got stuck

## Systematic Approach to Hyperparameter Tuning

### Step 1: Understand What Each Parameter Does

#### **Learning Rate** (`learning_rate`)
**What it does**: Controls how big the parameter updates are

- **Too High** (>1e-3):
  - Training becomes unstable
  - Policy can collapse
  - Large fluctuations in performance

- **Too Low** (<1e-5):
  - Training is very slow
  - May not converge in reasonable time

- **Sweet Spot**: 3e-4 to 1e-4 for PPO, 3e-4 for SAC

**Your issue**: Default 3e-4 might be too high for CPU training
**Recommendation**: Try 1e-4 or 5e-5

---

#### **Entropy Coefficient** (`entropy_coef` for PPO, `alpha` for SAC)
**What it does**: Encourages exploration by penalizing overconfident actions

- **Too Low** (<0.001):
  - Agent becomes too deterministic too quickly
  - Gets stuck in local optima
  - Limited exploration

- **Too High** (>0.1):
  - Agent stays too random
  - Never commits to good strategies
  - Training takes forever

- **Sweet Spot**: 0.01-0.02 for PPO, auto-tuned for SAC

**Your issue**: Default 0.01 might be too low
**Recommendation**: Try 0.02 or 0.03 for more exploration

---

#### **PPO Clip Epsilon** (`clip_epsilon`)
**What it does**: Limits how much the policy can change per update

- **Too Low** (<0.1):
  - Very conservative updates
  - Slow learning
  - Very stable but inefficient

- **Too High** (>0.3):
  - Large policy changes
  - Can cause instability
  - Risk of policy collapse

- **Sweet Spot**: 0.1-0.2

**Your issue**: Default 0.2 is standard
**Recommendation**: Keep at 0.2 or try 0.15 for more stability

---

#### **Discount Factor (Gamma)** (`gamma`)
**What it does**: How much the agent values future rewards

- **Too Low** (<0.95):
  - Agent is myopic (only considers immediate rewards)
  - Won't plan ahead

- **Too High** (>0.995):
  - Agent overvalues distant future
  - Harder to learn

- **Sweet Spot**: 0.99 for most tasks

**Your issue**: Default 0.99 is appropriate
**Recommendation**: Keep at 0.99

---

#### **GAE Lambda** (`gae_lambda` - PPO only)
**What it does**: Balances bias vs variance in advantage estimation

- **Too Low** (<0.9):
  - Lower variance but higher bias
  - More stable but less accurate

- **Too High** (>0.98):
  - Lower bias but higher variance
  - More accurate but noisier

- **Sweet Spot**: 0.95

**Your issue**: Default 0.95 is standard
**Recommendation**: Keep at 0.95

---

#### **Network Architecture** (`hidden_dims`)
**What it does**: Capacity of the neural network

- **Too Small** ([64, 64]):
  - Not enough capacity to learn complex behaviors
  - Underfitting

- **Too Large** ([1024, 1024]):
  - Overfitting risk
  - Slower training
  - Needs more samples

- **Sweet Spot**: [256, 256] or [400, 300]

**Your issue**: Default [256, 256] is appropriate
**Recommendation**: Keep at [256, 256]

---

#### **Batch Size** (`mini_batch_size` for PPO, `batch_size` for SAC)
**What it does**: Number of samples per gradient update

- **Too Small** (<32):
  - Noisy gradients
  - Unstable training
  - Slower convergence

- **Too Large** (>512):
  - Less frequent updates
  - May miss important patterns

- **Sweet Spot**: 64-256

**Your issue**: Default 64 (PPO) or 256 (SAC) is good
**Recommendation**: Keep as is, or increase if using GPU

---

#### **Training Timesteps** (`total_timesteps`)
**What it does**: Total amount of interaction with environment

- **Too Few** (<500K):
  - Not enough experience to learn
  - Agent hasn't seen enough scenarios

- **Too Many** (>10M):
  - Diminishing returns
  - Wasted compute

- **Sweet Spot**: 1-3M for normal, 3-5M for hardcore

**Your issue**: 2M might not be enough if learning is slow
**Recommendation**: Try 3M timesteps

---

## Diagnosis: What Went Wrong With Your Training

Based on your results:

### Symptoms:
- Mean reward: -50 (very poor)
- All episodes: 1600 steps (survives but doesn't progress)
- Consistent performance (low variance)

### Most Likely Causes:

1. **Premature Convergence**: Agent found a "safe" strategy (do nothing or minimal movement) and stuck with it
2. **Insufficient Exploration**: Didn't try enough diverse actions to discover walking
3. **Learning Rate**: Possibly too high, causing instability

### Recommended Fixes:

1. ✅ **Increase exploration** (entropy_coef: 0.01 → 0.02)
2. ✅ **Reduce learning rate** (learning_rate: 3e-4 → 1e-4)
3. ✅ **Train longer** (total_timesteps: 2M → 3M)
4. ✅ **Add learning rate schedule** (decay over time)

---

## Pre-Configured Solutions to Try

I've created several configurations for you. Try them in order:

### **Config 1: Conservative & Stable** (Recommended First)
File: `configs/ppo_conservative.yaml`
- Lower learning rate for stability
- Moderate exploration
- Longer training

**Use when**: You want stable, reliable training
**Expected time**: 9-15 hours on CPU

### **Config 2: Exploration-Focused**
File: `configs/ppo_exploration.yaml`
- Higher entropy for more exploration
- Standard learning rate
- Longer training

**Use when**: Agent seems stuck in local optima
**Expected time**: 9-15 hours on CPU

### **Config 3: Fast & Aggressive**
File: `configs/ppo_fast.yaml`
- Larger batches
- Higher learning rate
- Shorter training

**Use when**: Quick experiments, have GPU
**Expected time**: 2-4 hours on GPU, 6-8 hours on CPU

### **Config 4: SAC (Sample Efficient)**
File: `configs/sac_tuned.yaml`
- Different algorithm entirely
- More sample efficient
- Better for continuous control

**Use when**: PPO keeps failing
**Expected time**: 4-8 hours on CPU

---

## How to Use These Configs

### Step 1: Pick a configuration
```bash
# Try conservative first (recommended)
python train.py --config configs/ppo_conservative.yaml
```

### Step 2: Monitor training
```bash
# In another terminal
tensorboard --logdir experiments/logs
```

**Look for:**
- Episode reward should gradually increase
- Should see positive rewards within 500K-1M steps
- Mean reward should reach 100+ by 1.5M steps

### Step 3: Evaluate periodically
The training will auto-evaluate every 10K steps. Watch for:
- Eval mean reward increasing
- Less variance in rewards over time

### Step 4: If still failing
- Check TensorBoard for signs of learning
- Try next configuration
- Consider SAC if PPO keeps failing

---

## Advanced Tuning Strategies

### Strategy 1: Learning Rate Schedule
Gradually decrease learning rate during training:

```yaml
# Start high, end low
learning_rate: 3.0e-4  # Initial
# Decays to 3.0e-5 by end (implement in train.py)
```

### Strategy 2: Curriculum Learning
Start with easier task, then increase difficulty:

1. Train on flat terrain (if available)
2. Fine-tune on normal terrain
3. Finally, try hardcore

### Strategy 3: Reward Shaping
Modify rewards to guide learning:

```python
# In src/envs/env_wrapper.py
# Add shaping rewards:
# - Small positive reward for forward velocity
# - Penalty for falling
# - Bonus for maintaining balance
```

### Strategy 4: Increase Network Capacity
If agent consistently fails:

```yaml
hidden_dims: [512, 512]  # Instead of [256, 256]
```

But only if you have enough compute!

---

## Red Flags in Training

Watch out for these warning signs:

### 🚩 Policy Collapse
**Symptoms**:
- Sudden drop in performance
- All actions become the same
- Episode rewards flatline

**Fix**:
- Increase entropy coefficient
- Reduce learning rate
- Add gradient clipping

### 🚩 No Learning
**Symptoms**:
- Rewards stay constant
- No improvement after 500K steps
- Very low variance in actions

**Fix**:
- Increase learning rate
- Increase exploration
- Check environment is correct

### 🚩 Unstable Training
**Symptoms**:
- Wild swings in performance
- Rewards jump up and down
- Training crashes

**Fix**:
- Decrease learning rate
- Reduce batch size
- Increase value loss coefficient

### 🚩 Slow Convergence
**Symptoms**:
- Very gradual improvement
- Takes >3M steps to see progress

**Fix**:
- Increase learning rate slightly
- Increase batch size
- Try SAC instead

---

## Hyperparameter Tuning Checklist

Before retraining, ask yourself:

- [ ] Did I check TensorBoard logs from previous run?
- [ ] Do I understand why the previous run failed?
- [ ] Am I changing only 1-2 parameters at a time?
- [ ] Do I have enough compute/time for this run?
- [ ] Am I monitoring training in real-time?
- [ ] Have I tried the conservative config first?

---

## Quick Reference Table

| Parameter | Default | Conservative | Exploration | Fast | Notes |
|-----------|---------|--------------|-------------|------|-------|
| `learning_rate` | 3e-4 | 5e-5 | 1e-4 | 5e-4 | Lower = more stable |
| `entropy_coef` | 0.01 | 0.015 | 0.03 | 0.02 | Higher = more exploration |
| `clip_epsilon` | 0.2 | 0.15 | 0.2 | 0.25 | Lower = more stable |
| `mini_batch_size` | 64 | 64 | 64 | 128 | Larger = faster (needs GPU) |
| `total_timesteps` | 2M | 3M | 3M | 1.5M | More = better learning |
| `rollout_steps` | 2048 | 2048 | 4096 | 2048 | More = less frequent updates |

---

## When to Stop Tuning

You're done when:

1. ✅ Mean reward > 300 consistently
2. ✅ Success rate > 80%
3. ✅ Agent walks smoothly when you watch it
4. ✅ Performance is stable across runs

Don't over-optimize! Getting to 300+ is enough.

---

## Pro Tips

1. **Start Conservative**: Always try stable configs first
2. **One Change at a Time**: Don't change everything at once
3. **Use TensorBoard**: Monitor every run
4. **Be Patient**: RL training takes time
5. **Try Different Algorithms**: SAC might work better than PPO
6. **Consider Hardware**: GPU makes huge difference
7. **Keep Notes**: Document what works and what doesn't

---

## Expected Results by Configuration

| Config | Time (CPU) | Expected Reward | Success Rate |
|--------|------------|-----------------|--------------|
| Conservative | 9-15h | 250-350+ | 60-90% |
| Exploration | 9-15h | 200-350+ | 50-85% |
| Fast | 6-8h | 150-300 | 40-70% |
| SAC Tuned | 4-8h | 250-350+ | 65-90% |

These are estimates - actual results vary!

---

## Next Steps

1. **Choose a configuration** from the pre-configured ones I'll create
2. **Start training** with monitoring
3. **Check TensorBoard** regularly
4. **Evaluate at checkpoints** (every 500K steps)
5. **Adjust if needed** based on what you see
6. **Be patient** - good results take time!

Let's create those config files now! →

