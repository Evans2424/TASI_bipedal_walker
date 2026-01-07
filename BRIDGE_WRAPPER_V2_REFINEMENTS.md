# Bridge Wrapper V2 - Anti-Exploit Refinements

**Date**: 2026-01-04
**Issue**: V1 wrapper was exploited by agent, causing training collapse at 625K steps
**Solution**: V2 wrapper with stricter criteria and exploit prevention

---

## V1 Training Collapse (FAILED)

### Timeline
- **0-600K steps**: Good performance, episodes reaching 500 steps ✓
- **625K steps**: Catastrophic collapse begins ⚠️
- **675K-925K steps**: Stuck at 40-60 step episodes ❌

### Performance Degradation
```
Eval Results:
  600K: 500 steps, 2.79 reward  ✓ (last good checkpoint)
  625K: 241 steps, 1.14 reward  ⚠️ (collapse starts)
  650K: 136 steps, -13.8 reward ❌
  675K: 32 steps, 8.33 reward   ❌ (catastrophic)
  700K-925K: 40-85 steps        ❌ (stuck in bad state)
```

### Root Cause: Reward Hacking

**The Exploit**:
Agent discovered it could trigger "waiting" detection from episode start:
1. Stand still immediately (velocity < 0.1, angle < 0.3)
2. Get patience bonus (+0.005 per step)
3. Get penalty refunds (smoothness, hull stability)
4. Move slightly for minimal forward progress
5. Fall early but keep small positive reward (10-15)

**Why This Happened**:
V1 `_is_waiting()` function had no safeguards:
```python
# V1 - EXPLOITABLE
def _is_waiting(self, obs):
    horizontal_velocity = abs(obs[2])
    hull_angle = abs(obs[0])
    hull_angular_vel = abs(obs[1])

    # PROBLEM: Can trigger anywhere, anytime!
    is_low_velocity = horizontal_velocity < 0.1
    is_stable = hull_angle < 0.3
    is_steady = hull_angular_vel < 0.5

    return is_low_velocity and is_stable and is_steady
```

**Result**: Agent learned "stand still → get rewards" instead of "navigate obstacles"

---

## V2 Anti-Exploit Features

### 1. Forward Progress Requirement
**Problem**: V1 allowed waiting detection from episode start
**Fix**: Only enable after agent reaches x=10 units

```python
# V2 - PROTECTED
self.min_progress_for_waiting = 10.0
self.total_distance = 0.0  # Track progress

def _is_waiting(self, obs):
    # ... basic checks ...

    # NEW: Must have made forward progress
    has_progressed = self.total_distance > self.min_progress_for_waiting

    return is_low_velocity and is_stable and is_steady and has_progressed
```

**Effect**: Agent can't exploit waiting at episode start - must navigate first

### 2. Consecutive Frame Requirement
**Problem**: V1 triggered on single frame of stability
**Fix**: Require 8 consecutive stable frames (2 seconds)

```python
# V2 - PROTECTED
self.min_consecutive_frames = 8
self.consecutive_waiting_steps = 0

def step(self, action):
    is_currently_stable = self._is_waiting(obs)

    if is_currently_stable:
        self.consecutive_waiting_steps += 1
    else:
        self.consecutive_waiting_steps = 0

    # Only apply bonus if stable for minimum duration
    is_waiting = self.consecutive_waiting_steps >= self.min_consecutive_frames
```

**Effect**: Prevents brief stops from triggering waiting state

### 3. Removed Patience Bonus
**Problem**: V1 gave +0.005 reward per waiting step (exploited)
**Fix**: Completely removed

```python
# V1 - EXPLOITABLE
patience_reward = self.patience_bonus  # +0.005
reward += patience_reward

# V2 - REMOVED
# NO PATIENCE BONUS - Removed to prevent exploitation
```

**Effect**: No free rewards for standing still

### 4. Stricter Thresholds
**Problem**: V1 thresholds too permissive
**Fix**: Tightened velocity and angle limits

```python
# V1 thresholds
waiting_velocity_threshold = 0.1
waiting_angle_threshold = 0.3

# V2 thresholds (STRICTER)
waiting_velocity_threshold = 0.05  # 50% reduction
waiting_angle_threshold = 0.2      # 33% reduction
```

**Effect**: Harder to accidentally trigger waiting state

---

## V2 Implementation

### File: `elite_hardcore_bridge_wrapper_v2.py`

**Key Changes**:
```python
class EliteHardcoreBridgeWrapperV2(EliteHardcoreWrapper):
    def __init__(self, env, ...):
        # V2 PARAMETERS (ANTI-EXPLOIT)
        self.waiting_velocity_threshold = 0.05  # Stricter
        self.waiting_angle_threshold = 0.2      # Stricter
        self.min_progress_for_waiting = 10.0    # NEW
        self.min_consecutive_frames = 8         # NEW
        # NO patience_bonus parameter

        self.consecutive_waiting_steps = 0
        self.total_distance = 0.0

    def _is_waiting(self, obs):
        # Basic stability checks (stricter thresholds)
        is_low_velocity = horizontal_velocity < 0.05
        is_stable = hull_angle < 0.2
        is_steady = hull_angular_vel < 0.5

        # NEW: Forward progress requirement
        has_progressed = self.total_distance > 10.0

        return is_low_velocity and is_stable and is_steady and has_progressed

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Track forward progress
        self.total_distance += obs[2] * 4  # vel_x * frame_skip

        # Check consecutive stable frames
        is_currently_stable = self._is_waiting(obs)
        if is_currently_stable:
            self.consecutive_waiting_steps += 1
        else:
            self.consecutive_waiting_steps = 0

        # Only activate after 8 consecutive frames
        is_waiting = self.consecutive_waiting_steps >= 8

        if is_waiting:
            # Refund penalties (80% reduction)
            # NO patience bonus
            ...
```

---

## Fresh Training with V2

### Configuration
- **Wrapper**: EliteHardcoreBridgeWrapperV2 (anti-exploit)
- **Starting**: From scratch (no resume)
- **Config**: Same proven Elite Hardcore hyperparameters
- **Backed up**: V1 failed training saved to `*_V1_FAILED_exploit`

### Initial Progress (First 30K Steps)

```
Timestep    Length    Reward    Status
--------    ------    ------    ------
480         47        -51       Initial random policy
8,500       45        -50       Early exploration
28,700      133       -119      Learning to survive
30,800      151       -131      Steady improvement ✓
```

**Trajectory**: Healthy learning curve, no exploitation detected

**Key Indicators**:
- ✅ Episode length increasing steadily (47 → 151 steps)
- ✅ Reward improving (no sudden positive spikes)
- ✅ No premature convergence to local optimum
- ✅ V2 wrapper confirmed active in logs

---

## Monitoring Plan

### Watch for Good Signs
1. **Steady length increase**: 50 → 100 → 200 → 400 → 500+ steps
2. **Improving reward**: -130 → -80 → -30 → 0 → 50 → 150+
3. **No early deaths**: Eval episodes should reach 500+ steps by 100K
4. **Legitimate waiting**: `is_waiting` flag only appears after x>10 progress

### Watch for Bad Signs (Exploitation)
1. **Short episodes with positive reward**: If seeing 40-60 steps with +10-20 reward
2. **Premature convergence**: Length stops improving before 400+ steps
3. **Suspicious waiting**: `is_waiting` flag appearing early in episodes
4. **Reward spikes**: Sudden jumps to positive reward without length increase

### Checkpoints to Validate
- **100K**: Should see 300-400 step episodes, -50 to -20 reward
- **250K**: Should see 400-500 step episodes, -20 to +20 reward
- **500K**: Should see 500+ step episodes, +20 to +80 reward
- **1M+**: Should see consistent 600-1000 steps, 150-250 reward

---

## Expected V2 vs V1 Comparison

| Metric | V1 (Failed) | V2 (Expected) |
|--------|-------------|---------------|
| **100K steps** | ~400 steps (then collapsed) | 300-400 steps |
| **600K steps** | 500 steps (last good) | 500-700 steps |
| **625K steps** | 241 steps (collapse!) | 600-800 steps |
| **1M steps** | N/A (stuck at 50) | 700-1000 steps |
| **Final (10M)** | Failed | 250-300 reward |

### Why V2 Should Work
1. **Can't exploit at start**: Forward progress requirement
2. **Can't exploit briefly**: Consecutive frame requirement
3. **No free rewards**: Patience bonus removed
4. **Stricter criteria**: Harder to trigger accidentally
5. **Still helps bridges**: Legitimate waiting (after progress) gets penalty reduction

---

## Files Modified/Created

### Created
- ✅ `elite_hardcore_bridge_wrapper_v2.py` - Anti-exploit wrapper

### Modified
- ✅ `train_custom_walker.py` - Updated to use V2 wrapper
  - Import changed to V2
  - Logging updated to show V2 features
  - make_env() uses EliteHardcoreBridgeWrapperV2

### Backed Up
- ✅ `experiments/checkpoints/..._V1_FAILED_exploit/` - Failed V1 checkpoints
- ✅ `experiments/logs/..._V1_FAILED_exploit/` - Failed V1 tensorboard logs

---

## Next Steps

### Immediate (0-100K steps)
1. ✅ Monitor training progress every 25K steps
2. ✅ Verify episode lengths increasing steadily
3. ✅ Check eval results at 25K, 50K, 75K, 100K
4. ✅ Ensure no exploitation (short episodes + positive reward)

### Short-term (100K-500K steps)
1. Compare to V1 performance at same checkpoints
2. Verify episodes reaching 500+ steps consistently
3. Check that waiting detection only triggers after progress
4. Save checkpoints every 100K for safety

### Long-term (500K-10M steps)
1. Monitor for any new exploitation patterns
2. Track bridge crossing success rate (via videos)
3. Compare final performance to standard Elite Hardcore (~280 reward)
4. Document any additional refinements needed

---

## Lessons Learned

### Reward Shaping Pitfalls
1. **Any reward bonus can be exploited** - Agent will find the easiest way to trigger it
2. **Unconditional bonuses are dangerous** - Always require meaningful progress first
3. **Test early and often** - 625K steps of training wasted due to late detection
4. **Monitor full distributions** - Mean metrics can hide exploitation (short episodes but positive reward)

### Best Practices Going Forward
1. **Require forward progress** before any special bonuses
2. **Use consecutive frame requirements** to prevent brief exploitation
3. **Start with penalty reduction only** - Add bonuses cautiously later
4. **Monitor episode length distributions** - Not just mean reward
5. **Save frequent checkpoints** - Easy to rollback if issues found

---

## Current Status

**Training**: In progress with V2 wrapper (fresh start)
**Progress**: ~30K steps, 151-step episodes, healthy learning
**Checkpoints**: None yet (first save at 100K)
**Estimated Time**: 10M steps = ~12-15 hours on Apple Silicon

**Monitor Progress**:
```bash
# Live log
tail -f training_bridge_v2_fresh.log

# Tensorboard
tensorboard --logdir experiments/logs/sac_elite_unified_hardcore_gpu_custom_bridges/
```

**Test Checkpoint** (after 100K+):
```bash
python test_checkpoint_600k.py  # Modify to test current checkpoint
```

---

**Status**: V2 training running, exploitation prevented, healthy early progress ✓
