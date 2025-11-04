# Quick Start: Hyperparameter Tuning

## Your Situation

**Previous Result**: Mean reward -50 (VERY POOR)
**Goal**: Achieve 300+ mean reward
**Problem**: Agent learned to do nothing/fall instead of walking

---

## What I've Created for You

### 📚 **5 Pre-Configured Files** (Ready to Use!)

1. **`configs/ppo_conservative.yaml`** ⭐ **START HERE**
   - Most stable, reliable option
   - Lower learning rate (5e-5 vs 3e-4)
   - More exploration (0.015 vs 0.01)
   - Longer training (3M vs 2M steps)
   - **Best first choice after failure**

2. **`configs/ppo_exploration.yaml`**
   - Maximum exploration focus
   - High entropy (0.03)
   - Longer rollouts (4096 steps)
   - Try if conservative doesn't explore enough

3. **`configs/ppo_fast.yaml`**
   - Quick experiments
   - Higher learning rate
   - Shorter training
   - Good for rapid iteration

4. **`configs/sac_tuned.yaml`**
   - Different algorithm (SAC)
   - More sample efficient
   - Better for continuous control
   - Try if PPO keeps failing

5. **`configs/ppo_hardcore.yaml`**
   - For hardcore mode
   - Only after mastering normal mode

### 📖 **Complete Guide**
- **`HYPERPARAMETER_TUNING.md`** - Detailed explanation of every parameter

---

## Recommended Action Plan

### **Step 1: Try Conservative Config** (Recommended)

```bash
source venv/bin/activate
python train.py --config configs/ppo_conservative.yaml
```

**Why this config?**
- ✅ Lower learning rate = more stable
- ✅ Slightly more exploration
- ✅ Longer training = more chance to learn
- ✅ Best balance of stability and learning

**Expected time**: 9-15 hours on CPU
**Expected result**: 250-350+ reward (if working)

---

### **Step 2: Monitor Training**

In another terminal:
```bash
tensorboard --logdir experiments/logs
```

Open: http://localhost:6006

**Watch for:**
- 📈 Episode reward should increase over time
- 🎯 Should see positive rewards by 500K-1M steps
- ✅ Mean reward should reach 100+ by 1.5M steps

**Red flags:**
- ❌ Rewards stay at -50 for >500K steps = not learning
- ❌ Wild swings = instability
- ❌ Sudden drops = policy collapse

---

### **Step 3: If Still Failing After 1M Steps**

Stop training (Ctrl+C) and try SAC:

```bash
python train.py --config configs/sac_tuned.yaml
```

**Why SAC?**
- Different algorithm, different approach
- More sample efficient
- Often works when PPO doesn't
- Better for continuous control tasks

---

### **Step 4: If Working But Slow**

If you see improvement but it's taking too long:

```bash
# Try exploration-focused config
python train.py --config configs/ppo_exploration.yaml
```

---

## Quick Config Comparison

| Config | Learning Rate | Exploration | Training Steps | Best For |
|--------|---------------|-------------|----------------|----------|
| **conservative** ⭐ | 5e-5 (LOW) | 0.015 | 3M | **After failure** |
| **exploration** | 1e-4 | 0.03 (HIGH) | 3M | Stuck in local optima |
| **fast** | 5e-4 (HIGH) | 0.02 | 1.5M | Quick experiments |
| **sac_tuned** | 3e-4 | Auto | 1.5M | **PPO keeps failing** |

---

## What Changed From Your Failed Run?

Your original config had:
```yaml
learning_rate: 3e-4      # Too high?
entropy_coef: 0.01       # Too low? (not enough exploration)
total_timesteps: 2M      # Not enough?
```

Conservative config changes:
```yaml
learning_rate: 5e-5      # ⬇️ 6x LOWER (more stable)
entropy_coef: 0.015      # ⬆️ 1.5x HIGHER (more exploration)
total_timesteps: 3M      # ⬆️ 1.5x MORE (more learning time)
```

**Why these changes?**
- **Lower LR**: Your agent might have been updating too aggressively
- **Higher entropy**: Need to try more diverse actions to find walking
- **More steps**: Need more time to discover good behaviors

---

## Commands Summary

```bash
# Activate environment
source venv/bin/activate

# TRY FIRST (Conservative)
python train.py --config configs/ppo_conservative.yaml

# Monitor
tensorboard --logdir experiments/logs

# If failing, try SAC
python train.py --config configs/sac_tuned.yaml

# Compare configs (optional)
python scripts/compare_configs.py

# Evaluate when done
python scripts/analyze_model.py \
    --checkpoint experiments/checkpoints/ppo_conservative/final_model.pt \
    --config configs/ppo_conservative.yaml

# Watch agent
python scripts/watch_agent.py \
    --checkpoint experiments/checkpoints/ppo_conservative/final_model.pt \
    --config configs/ppo_conservative.yaml
```

---

## Understanding the Configs

### 🟢 **Conservative** (Recommended Start)
**Philosophy**: Slow and steady wins the race
- Smaller steps (low learning rate)
- More exploration (higher entropy)
- More time to learn

**Use when**: You want reliable, stable training

### 🟡 **Exploration**
**Philosophy**: Try everything to find what works
- Maximum exploration
- Longer rollouts for diverse experience

**Use when**: Agent seems stuck in one strategy

### 🟠 **Fast**
**Philosophy**: Move fast and iterate
- Bigger steps (high learning rate)
- Shorter training

**Use when**: Quick experiments, have GPU

### 🟣 **SAC Tuned**
**Philosophy**: Different algorithm, better sample efficiency
- Off-policy learning
- Automatic exploration tuning

**Use when**: PPO keeps failing

---

## Expected Timeline

### Conservative Config (Recommended):
```
0-500K steps:   Mostly negative rewards, random behavior
500K-1M steps:  Start seeing positive rewards, some progress
1M-2M steps:    Consistent improvement, rewards 50-150
2M-3M steps:    Good performance, rewards 200-350+
```

### SAC Config:
```
0-200K steps:   Random exploration, very negative
200K-500K:      Start learning, rewards improving
500K-1M:        Good performance, rewards 200-300+
```

---

## When to Stop Training

**Good signs** (can stop):
- ✅ Mean reward consistently > 300
- ✅ Evaluation shows stable performance
- ✅ Agent walks smoothly when you watch it

**Keep training**:
- ⏳ Reward still increasing
- ⏳ Not reached 300 yet but improving

**Stop and retry**:
- ❌ No improvement after 1M steps
- ❌ Rewards stuck at -50
- ❌ Training looks unstable

---

## Pro Tips

1. **Be Patient**: RL training takes time, especially on CPU
2. **Monitor Early**: Check TensorBoard after first 100K steps
3. **Don't Give Up**: If conservative fails, try SAC
4. **Take Notes**: Document what works for future runs
5. **Use GPU**: If you have L40 access, change `device: "cuda"`

---

## Next Actions

### Right Now:
```bash
source venv/bin/activate
python train.py --config configs/ppo_conservative.yaml
```

### In Another Terminal:
```bash
tensorboard --logdir experiments/logs
```

### While Training:
- Read `HYPERPARAMETER_TUNING.md` for deep dive
- Check TensorBoard every 30 minutes
- Be patient!

---

## Need More Help?

1. **Detailed guide**: Read `HYPERPARAMETER_TUNING.md`
2. **Evaluation guide**: Read `MODEL_EVALUATION.md`
3. **Compare configs**: Run `python scripts/compare_configs.py`
4. **Check training**: Monitor TensorBoard

Good luck! The conservative config should work much better than your original training! 🚀
